#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dylan.gilley@gmail.com


import sys
import numpy as np
from hkmcmd import interactions, voxels, system, io, reaction


# Main argument
def main(argv):
    """Driver for conducting Hybrid MDMC simulation."""

    # Parse the command line arguments
    parser = io.HkmcmdArgumentParser()
    args = parser.parse_args()

    # set debugger
    my_debugger = lambda: None
    if args.debug is True:
        my_debugger = breakpoint

    # Read and clean the system data file
    system_data = system.SystemData(args.system, args.prefix, filename_json=args.filename_json)
    system_data.read_json()
    system_data.clean()
    for rxn in system_data.reactions:
        rxn.calculate_raw_rate(
            system_data.hkmcmd["temperature_rxn"], method="Arrhenius"
        )

    my_debugger()

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

    my_debugger()

    # Read in the diffusion rate dictionary holding the diffusion rates matrix for each species
    diffusion_rates_dict_matrix = {
        sp: np.ones((np.prod(voxels_datafile.number_of_voxels), np.prod(voxels_datafile.number_of_voxels)))*np.inf
        for sp in [molecule.kind for molecule in system_data.species]
    }
    if system_data.scaling_diffusion["well-mixed"] is False:
        diffusion_rates_dict_matrix = io.parse_diffusion_file(args.filename_diffusion)

    # Begin the KMC loop
    moleculecount_starting = len(molecules_list)
    moleculecount_current = len(molecules_list)
    Reacting = True
    while Reacting:

        # Assemble DataFrames from the system_state instance.
        # Later, the functions that use these DataFrames can be updated to take the system_state instance.
        total_window = np.max([
            system_data.scaling_reaction["windowsize_slope"],
            system_data.scaling_reaction["windowsize_rxnselection"],
            system_data.scaling_reaction["windowsize_scalingpause"],
        ])
        reaction_scaling_df = system_state.assemble_reaction_scaling_df(total_window=total_window)
        progression_df = system_state.assemble_progression_df(
            sorted([molecule.kind for molecule in system_data.species]),
            total_window=total_window
        )

        # Perform reaction scaling, if requested.
        if system_data.hkmcmd["scale_rates"] is True:
            PSSrxns = reaction.get_PSSrxns(
                system_data.reactions,
                reaction_scaling_df,
                progression_df,
                system_data.scaling_reaction["windowsize_slope"],
                system_data.scaling_reaction["windowsize_rxnselection"],
                system_data.scaling_reaction["concentration_slope"],
                system_data.scaling_reaction["concentration_cycles"],
                system_data.scaling_reaction["reaction_selection_count"],
            )
            reaction_scaling_df = reaction.scalerxns(
                reaction_scaling_df,
                PSSrxns,
                system_data.scaling_reaction["windowsize_scalingpause"],
                system_data.scaling_reaction["scaling_factor"],
                system_data.scaling_reaction["scaling_minimum"],
                rxnlist="all",
            )

        my_debugger()

        # Assmeble a list of all possible reactions
        reactive_events_list = reaction.get_reactive_events_list(
            molecules_list,
            diffusion_rates_dict_matrix,
            reaction_scaling_df,
            system_data.reactions,
            system_data.hkmcmd["diffusion_cutoff"],
            avoid_double_counts=system_data.hkmcmd["avoid_double_counts"],
        )
        
        my_debugger()

        # Select a reaction
        reactive_event, dt = reaction.kMC_event_selection(reactive_events_list)
        reactive_event.create_product_molecules(
            [
                reaction
                for reaction in system_data.reactions
                if reaction.kind == reactive_event.kind
            ][0]
        )

        my_debugger()

        # Update the system with the selected reaction
        molecules_list = interactions.update_molecules_list_with_reaction(
            molecules_list,
            reactive_event,
            box,
            tolerance=0.0000_0010,
            maximum_iterations=None,
        )
        system_state.update_molecules(molecules_list)
        system_state.update_reactions([reactive_event.kind],reaction_scaling_df)
        system_state.update_steps(args.diffusion_cycle, dt)
        system_state.check_consistency()

        # Check for completion.
        moleculecount_current -= len(reactive_event.reactant_molecules)
        if (
            moleculecount_current
            < (1 - system_data.hkmcmd["change_threshold"]) * moleculecount_starting
        ):
            Reacting = False

        my_debugger()

    # Write the system_state file
    system_state.write_to_json(args.filename_system_state)

    # Write the LAMMPS data file
    atoms_list, bonds_list, angles_list, dihedrals_list, impropers_list = (
        interactions.get_interactions_lists_from_molcules_list(molecules_list)
    )

    my_debugger()

    io.write_lammps_data(
        args.prefix + ".in.data",
        atoms_list,
        bonds_list=bonds_list,
        angles_list=angles_list,
        dihedrals_list=dihedrals_list,
        impropers_list=impropers_list,
        box=box,
        num_atoms=len(atoms_list),
        num_bonds=len(bonds_list),
        num_angles=len(angles_list),
        num_dihedrals=len(dihedrals_list),
        num_impropers=len(impropers_list),
        num_atom_types=system_data.lammps.get("atom_types", None),
        num_bond_types=system_data.lammps.get("bond_types", None),
        num_angle_types=system_data.lammps.get("angle_types", None),
        num_dihedral_types=system_data.lammps.get("dihedral_types", None),
        num_improper_types=system_data.lammps.get("improper_types", None),
        masses=system_data.atomic_masses,
        charge=system_data.lammps.get("charged_atoms", None),
        wrap=False,
    )

    # Write the LAMMPS init file
    lammps_init = io.LammpsInitHandler(
        prefix=args.prefix,
        settings_file_name=args.filename_settings,
        data_file_name=args.prefix + ".in.data",
        **system_data.MD_cycling,
    )
    lammps_init.write()

    return


if __name__ == "__main__":
    main(sys.argv[1:])
