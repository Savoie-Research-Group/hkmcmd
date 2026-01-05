#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dylan.gilley@gmail.com


import sys, datetime
import numpy as np
import pandas as pd
from HkMCMD.hkmcmd import interactions, voxels, system, io, reaction


# Main argument
def main(argv):
    """Driver for conducting Hybrid MDMC simulation."""

    # Parse the command line arguments
    parser = io.HkmcmdArgumentParser()
    parser.add_argument("-temperature", type=float, default=0.852)
    parser.add_argument("-gamma", type=float, default=0.03)
    parser.add_argument("-total_steps", type=int, default=2000)
    parser.add_argument("-print_every", type=int, default=100)
    args = parser.parse_args()

    # set debugger
    my_debugger = lambda: None
    if args.debug is True:
        my_debugger = breakpoint

    system_data = system.SystemData(args.system, args.prefix, filename_json=args.filename_json)
    system_data.read_json()
    system_data.clean()
    for rxn in system_data.reactions:
        rxn.calculate_raw_rate(
            system_data.hkmcmd["temperature_rxn"], method="Arrhenius"
        )

    # Read the data file
    (
        atoms_list,
        bonds_list,
        angles_list,
        dihedrals_list,
        impropers_list,
        box,
        _,
    ) = io.parse_data_file(
        args.filename_data,
        atom_style=system_data.lammps["atom_style"],
        preserve_atom_order=False,
        preserve_bond_order=False,
        preserve_angle_order=False,
        preserve_dihedral_order=False,
        preserve_improper_order=False,
        tdpd_conc=[],
        unwrap=False,
    )

    # Create the Voxels object
    voxels_datafile = voxels.Voxels(box, system_data.scaling_diffusion["number_of_voxels"])

    # Create the SystemState instance
    system_state = system.SystemState(filename=args.filename_system_state)
    system_state.read_data_from_json()

    # Check for consistency between SystemState and Datafile atoms
    system_state.check_atoms_list(atoms_list)

    # Create molecules_list
    molecules_list = [
        interactions.Molecule(ID=ID)
        for ID in sorted(list(set([atom.molecule_ID for atom in atoms_list])))
    ]
    for molecule in molecules_list:
        molecule.fill_lists(
            atoms_list=atoms_list,
            bonds_list=bonds_list,
            angles_list=angles_list,
            dihedrals_list=dihedrals_list,
            impropers_list=impropers_list,
        )
        molecule.kind = system_state.get_molecule_kind(molecule.ID)
        molecule.assign_voxel_idx(voxels_datafile)

    for molecule in molecules_list:
        molecule.translate(voxels_datafile.centers[molecule.voxel_idx[0]])

    my_debugger()

    # Calculate number of chunks and chunk size
    chunk_size = args.print_every
    num_chunks = args.total_steps // chunk_size
    remaining_steps = args.total_steps % chunk_size
    if remaining_steps > 0:
        num_chunks += 1
    
    mol_kind_map = {"A": 1, "A2": 2}
    mol_kind_unmap = {1: "A", 2: "A2", 0: 0}
    
    # Track global simulation state
    current_time = 0.0
    global_step = 0

    # find face neighbors
    face_neighbors_dict = get_face_sharing_neighbors_with_partial_periodicity(voxels_datafile)

    # diffusion rates
    diffusion_rates = {}
    conversion = np.sqrt(system_data.lammps["LJ_energy"]) * system_data.lammps["LJ_distance"]/np.sqrt(system_data.lammps["LJ_mass"])
    voxel_length = (voxels_datafile.xbounds[1] - voxels_datafile.xbounds[0]) * system_data.lammps["LJ_distance"]
    diffusion_rates["A"] = args.temperature * args.gamma / 100 * conversion / (voxel_length**2)
    diffusion_rates["A2"] = args.temperature * args.gamma / 200 * conversion / (voxel_length**2)

    # Process simulation in chunks
    for chunk_idx in range(num_chunks):
        # Determine chunk size (handle last chunk which may be smaller)
        if chunk_idx == num_chunks - 1 and remaining_steps > 0:
            current_chunk_size = remaining_steps
        else:
            current_chunk_size = chunk_size
        
        # Initialize data arrays for this chunk
        data_time = np.empty((current_chunk_size + 1,), dtype=float)
        data_voxels = np.empty((current_chunk_size + 1, len(atoms_list)), dtype=int)
        data_types = np.empty((current_chunk_size + 1, len(atoms_list)), dtype=int)
        
        # Set initial state for this chunk
        data_time[0] = current_time
        for molecule in molecules_list:
            for atom in molecule.atoms:
                atom_idx = atom.ID - 1  # assuming atom IDs start from 1
                data_voxels[0, atom_idx] = molecule.voxel_idx[0]
                data_types[0, atom_idx] = mol_kind_map[molecule.kind]
        
        # Run simulation for this chunk
        for step in range(1, current_chunk_size + 1):
            # Declare events lists
            events_rates = []
            events_types = []
            events_extra = []

            # Loop over voxels
            for vidx in voxels_datafile.IDs:
                # pull molecules in voxel
                molecules_in_voxel = [
                    mol
                    for mol in molecules_list
                    if mol.voxel_idx[0] == vidx
                ]

                # Find A molecules
                A_molecules = [
                    mol for mol in molecules_in_voxel
                    if mol.kind == "A"
                ]

                # Find A2 molecules
                A2_molecules = [
                    mol for mol in molecules_in_voxel
                    if mol.kind == "A2"
                ]

                # Find face-neighbors
                face_neighbors = face_neighbors_dict[vidx]

                # speedup list appending
                number_of_events = (
                    len(A_molecules) * len(face_neighbors) + # A diffusion
                    len(A2_molecules) * len(face_neighbors) + # A2 diffusion
                    len(A_molecules) * (len(A_molecules) - 1) // 2 + # A + A -> A2
                    len(A2_molecules) # A2 -> A + A
                )
                rates_here = [None] * number_of_events
                types_here = [None] * number_of_events
                extra_here = [None] * number_of_events

                # Add A + A -> A2 reactions to events list
                event_idx = 0
                if len(A_molecules) >= 2:
                    for i in range(len(A_molecules)):
                        for j in range(i + 1, len(A_molecules)):
                            mol1 = A_molecules[i]
                            mol2 = A_molecules[j]
                            rates_here[event_idx] = system_data.reactions[0].rawrate  # assuming first reaction is A + A -> A2
                            types_here[event_idx] = "dimerization"
                            extra_here[event_idx] = (mol1.ID, mol2.ID, vidx)
                            event_idx += 1

                # Add A2 -> A + A reactions to events list
                for mol in A2_molecules:
                    rates_here[event_idx] = system_data.reactions[1].rawrate  # assuming second reaction is A2 -> A + A
                    types_here[event_idx] = "dissociation"
                    extra_here[event_idx] = (mol.ID, vidx)
                    event_idx += 1

                # Add diffusions to neighboring voxels to events list
                for mol in molecules_in_voxel:
                    for neighbor_vidx in face_neighbors:
                        rates_here[event_idx] = diffusion_rates[mol.kind]
                        types_here[event_idx] = "diffusion"
                        extra_here[event_idx] = (mol.ID, vidx, neighbor_vidx)
                        event_idx += 1

                # Append to global events list
                events_rates.extend(rates_here)
                events_types.extend(types_here)
                events_extra.extend(extra_here)

            # select event, calcualte dt, execute event
            events_rates = np.array(events_rates)
            u2 = 0
            while u2 == 0:
                u2 = np.random.random()
            dt = -np.log(u2) / np.sum(events_rates)
            u1 = np.random.random()
            event_idx = np.argwhere(np.cumsum(events_rates) >= np.sum(events_rates) * u1)[0][0]

            if events_types[event_idx] == "diffusion":
                mol_ID, from_vidx, to_vidx = events_extra[event_idx]
                molecule_idx = [i for i, mol in enumerate(molecules_list) if mol.ID == mol_ID][0]
                molecules_list[molecule_idx].voxel_idx = (to_vidx,)
                molecules_list[molecule_idx].translate(voxels_datafile.centers[to_vidx])
            elif events_types[event_idx] == "dimerization":
                mol1_ID, mol2_ID, vidx = events_extra[event_idx]
                mol1idx = [i for i, mol in enumerate(molecules_list) if mol.ID == mol1_ID][0]
                mol2idx = [i for i, mol in enumerate(molecules_list) if mol.ID == mol2_ID][0]
                reactive_event = interactions.Reaction(
                    ID=1,
                    kind=system_data.reactions[0].kind,
                    reactant_molecules=[molecules_list[mol1idx], molecules_list[mol2idx]],
                    translation=0.0,
                )
                reactive_event.create_product_molecules( system_data.reactions[0] )
                molecules_list = interactions.update_molecules_list_with_reaction(
                    molecules_list,
                    reactive_event,
                    box,
                    tolerance=None,
                    maximum_iterations=20
                    )
            elif events_types[event_idx] == "dissociation":
                mol_ID, vidx = events_extra[event_idx]
                molidx = [i for i, mol in enumerate(molecules_list) if mol.ID == mol_ID][0]
                reactive_event = interactions.Reaction(
                    ID=2,
                    kind=system_data.reactions[1].kind,
                    reactant_molecules=[molecules_list[molidx]],
                    translation=0.0,
                )
                reactive_event.create_product_molecules( system_data.reactions[1] )
                molecules_list = interactions.update_molecules_list_with_reaction(
                    molecules_list,
                    reactive_event,
                    box,
                    tolerance=None,
                    maximum_iterations=20
                    )

            # update data with new molecules list
            for molecule in molecules_list:
                molecule.assign_voxel_idx(voxels_datafile)
            current_time += dt
            global_step += 1
            data_time[step] = current_time
            for molecule in molecules_list:
                for atom in molecule.atoms:
                    atom_idx = atom.ID - 1  # assuming atom IDs start from 1
                    data_voxels[step, atom_idx] = molecule.voxel_idx[0]
                    data_types[step, atom_idx] = mol_kind_map[molecule.kind]

            if global_step % args.print_every == 0:
                my_debugger()
                print(f"\nStep {global_step} ({datetime.datetime.now()})")
                if events_types[event_idx] == "diffusion":
                    print(f"Event: {events_types[event_idx]} (molecule {events_extra[event_idx][0]}, {events_extra[event_idx][1]} --> {events_extra[event_idx][2]})")
                elif events_types[event_idx] == "dimerization":
                    print(f"Event: {events_types[event_idx]} (molecule {events_extra[event_idx][0]} + molecule {events_extra[event_idx][1]} in voxel {events_extra[event_idx][2]})")
                elif events_types[event_idx] == "dissociation":
                    print(f"Event: {events_types[event_idx]} (molecule {events_extra[event_idx][0]} in voxel {events_extra[event_idx][1]})")
        
        # Write chunk data to CSV file
        write_chunk_to_csv(args.prefix, data_time, data_voxels, data_types, 
                          atoms_list, mol_kind_unmap, chunk_idx == 0)

    return


def write_chunk_to_csv(prefix, data_time, data_voxels, data_types, atoms_list, mol_kind_unmap, is_first_chunk):
    """Write chunk data to CSV file."""
    data = {}
    data["time"] = data_time
    for idx in range(len(atoms_list)):
        atom_ID = atoms_list[idx].ID
        data[f"atom_{atom_ID}_voxel"] = data_voxels[:, idx]
        data[f"atom_{atom_ID}_type"] = [mol_kind_unmap[k] for k in data_types[:, idx]]
    
    df = pd.DataFrame(data, columns=["time"] +
                      [f"atom_{atom.ID}_voxel" for atom in atoms_list] +
                      [f"atom_{atom.ID}_type" for atom in atoms_list])
    
    # Write header only for first chunk, append for subsequent chunks
    if is_first_chunk:
        df.to_csv(f"{prefix}.3Dkmc.csv", index=False)
    else:
        df.to_csv(f"{prefix}.3Dkmc.csv", mode='a', header=False, index=False)


def get_face_sharing_neighbors_with_partial_periodicity(voxels):
    """
    Get face-sharing neighbors for each voxel with periodic y,z and non-periodic x.
    
    Parameters
    ----------
    voxels : Voxels
        Voxels object containing the grid information
        
    Returns
    -------
    neighbors_dict : dict
        Keys: voxel index
        Values: list of face-sharing neighbor voxel indices
    """
    nx, ny, nz = voxels.number_of_voxels
    neighbors_dict = {}
    
    for voxel_idx in voxels.IDs:
        # Convert linear index to 3D grid coordinates
        i = voxel_idx // (ny * nz)  # x index
        j = (voxel_idx % (ny * nz)) // nz  # y index  
        k = voxel_idx % nz  # z index
        
        neighbors = []
        
        # Check all 6 face-sharing directions
        directions = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
        
        for di, dj, dk in directions:
            ni, nj, nk = i + di, j + dj, k + dk
            
            # Handle x dimension (non-periodic)
            if ni < 0 or ni >= nx:
                continue  # Skip if outside x bounds
                
            # Handle y dimension (periodic)
            nj = nj % ny
            
            # Handle z dimension (periodic)  
            nk = nk % nz
            
            # Convert back to linear index
            neighbor_idx = ni * ny * nz + nj * nz + nk
            neighbors.append(int(neighbor_idx))
            
        neighbors_dict[int(voxel_idx)] = sorted(neighbors)
    
    return neighbors_dict


if __name__ == "__main__":
    main(sys.argv[1:])
