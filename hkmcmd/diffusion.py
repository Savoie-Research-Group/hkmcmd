#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dylan.gilley@gmail.com


import sys, os, datetime
import numpy as np
from copy import deepcopy
from typing import Union
from hkmcmd import io, voxels, interactions, system, utility


def main(argv):

    # Parse command line arguments.
    parser = io.HkmcmdArgumentParser()
    parser.add_argument(
        "-species",
        type=str,
        default="all",
        help="Species to calculate diffusion rates for.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files."
    )
    args = parser.parse_args()
    args.species = args.species.split(",")

    # Read system data.
    system_data = system.SystemData(
        args.system, args.prefix, filename_json=args.filename_json
    )
    system_data.read_json()
    system_data.clean()

    # Read the data file.
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

    # Create the Voxels object.
    voxels_datafile = voxels.Voxels(
        box, system_data.scaling_diffusion["number_of_voxels"]
    )

    # Create the SystemState instance.
    system_state = system.SystemState(filename=args.filename_system_state)
    system_state.read_data_from_json()

    # Check for consistency between SystemState and Datafile atoms.
    system_state.check_atoms_list(atoms_list)

    # Create molecules_list.
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

    # Create log file
    logfile = io.FileTracker(args.prefix + ".diffusion.log")

    # Create the Diffusion object.
    diffusion_handler = Diffusion(
        args.prefix,
        molecules_list,
        voxels_datafile,
        overwrite=args.overwrite,
        fuzz=system_data.scaling_diffusion["fuzz"],
        direct_transition_method=system_data.scaling_diffusion[
            "direct_transition_rates_method"
        ],
        global_rate_method=system_data.scaling_diffusion[
            "global_diffusion_rates_method"
        ],
        filename_trajectory=args.filename_trajectory,
        time_conversion=system_data.lammps["time_conversion"],
    )

    # Calculate the direct transition rates.
    diffusion_handler.calculate_direct_transition_rates(
        logfile=logfile,
        lammps_stepsize=system_data.lammps_timestep_size,
        start=system_data.scaling_diffusion["trj_parse_start"],
        end=system_data.scaling_diffusion["trj_parse_end"],
        every=system_data.scaling_diffusion["trj_parse_every"],
        average_across_voxel_neighbors=system_data.scaling_diffusion[
            "average_across_voxel_neighbors"
        ],
    )

    # Calculate the global diffusion rates.
    diffusion_handler.perform_random_walks(
        starting_position_idxs=None,
        number_of_steps=None,
        species=args.species,
    )
    diffusion_handler.calculate_average_first_time_between_positions(
        species=args.species,
        recursion_interval=system_data.scaling_diffusion["recursion_interval"],
    )
    global_diffusion_rates = diffusion_handler.calculate_global_diffusion_rates()

    # Write the global diffusion rates to a file.
    file_diffusion = io.FileTracker(args.filename_diffusion)
    file_diffusion.write_separation()
    file_diffusion.write(f"\nDiffusionStep 0\n")
    for k, v in sorted(global_diffusion_rates.items()):
        file_diffusion.write(f"\n\nDiffusion Rates for {k}\n")
        file_diffusion.write_array2df(v)

    return


class Diffusion:

    def __init__(
        self,
        name: str,
        molecules_list: list,
        voxels: voxels.Voxels,
        overwrite: bool = False,
        fuzz: float = 0.0,
        time_conversion: float = 1.0,
        direct_transition_method: Union[None, str] = None,
        global_rate_method: Union[None, str] = None,
        filename_trajectory: Union[None, str] = None,
    ):

        # Positional arguments
        self.name = name
        self.molecules_list = molecules_list
        self.voxels = voxels

        # Optional arguments
        self.overwrite = overwrite
        self.fuzz = fuzz
        self.time_conversion = time_conversion
        self.direct_transition_method = direct_transition_method
        if self.direct_transition_method is None:
            self.direct_transition_method = "fuzzy_boundary"
        self.global_rate_method = global_rate_method
        if self.global_rate_method is None:
            self.global_rate_method = "random_walk"
        self.filename_trajectory = filename_trajectory
        if self.filename_trajectory is None:
            self.filename_trajectory = f"{self.name}.diffusion.lammpstrj"

        # Attributes to be filled later
        self.direct_transition_rates = None
        self.random_walk_positions = None
        self.random_walk_times = None
        self.random_walk_average_first_time_between_positions = None
        self.global_diffusion_rates = None

        return

    def calculate_direct_transition_rates(
        self,
        logfile: Union[None, io.FileTracker] = None,
        time_conversion: Union[None, float] = None,
        lammps_stepsize: float = 1.0,
        start: int = 0,
        end: int = -1,
        every: int = 1,
        average_across_voxel_neighbors: bool = False,
    ):
        """Calculate the direct transition rates.

        This method calculates the direct transition rates for each voxel. The method used to
        calculate the direct transition rates is determined by the `local_rate_method` attribute of
        the instance. The direct transition rates are stored as an attribute of the class and returned
        by this method.
        """

        if time_conversion is None:
            time_conversion = self.time_conversion

        # Read in or calculate the mvabfa.
        if os.path.exists(self.name + ".mvabfa.txt") and self.overwrite is False:
            molecule_IDs, timesteps, mvabfa = read_mvabfa_file(
                self.name + ".mvabfa.txt"
            )
        elif self.direct_transition_method == "fuzzy_boundary":
            mvabfa_file = io.FileTracker(
                self.name + ".mvabfa.txt", overwrite=self.overwrite
            )
            molecule_IDs, timesteps, mvabfa = calculate_mvabfa_fuzzy_boundary(
                self.filename_trajectory,
                self.molecules_list,
                self.voxels.number_of_voxels,
                start=start,
                end=end,
                every=every,
                fuzz=self.fuzz,
                logfile=logfile,
            )
            mvabfa_file.write("MoleculeIDs\n")
            mvabfa_file.write_array(np.array(molecule_IDs).flatten(), as_str=True)
            mvabfa_file.write("Timesteps\n")
            mvabfa_file.write_array(np.array(timesteps).flatten(), as_str=True)
            mvabfa_file.write("MVABFA\n")
            mvabfa_file.write_array2df(mvabfa)
        else:
            raise ValueError(
                f"direct transition rate method {self.direct_transition_method} not recognized."
            )

        # Read in or calculate the direct transition rates.
        if (
            os.path.exists(self.name + ".direct_transition_rates.txt")
            and self.overwrite is False
        ):
            self.direct_transition_rates = read_direct_transition_rates_file(
                self.name + ".direct_transition_rates.txt"
            )
        else:
            self.direct_transition_rates = calculate_direct_transition_rates(
                np.prod(self.voxels.number_of_voxels),
                mvabfa,
                (timesteps[1] - timesteps[0]) * lammps_stepsize * time_conversion,
                self.molecules_list,
            )
            if average_across_voxel_neighbors is True:
                for voxel_idxs in self.voxels.voxel_idxs_by_distance_groupings:
                    for species in self.direct_transition_rates.keys():
                        self.direct_transition_rates[species][voxel_idxs] = np.mean(
                            self.direct_transition_rates[species][voxel_idxs]
                        )
            direct_transition_rates_file = io.FileTracker(
                self.name + ".direct_transition_rates.txt", overwrite=self.overwrite
            )
            for molecule_kind, rate in self.direct_transition_rates.items():
                direct_transition_rates_file.write(f"\n\ntype {molecule_kind}\n")
                direct_transition_rates_file.write_array2df(rate)
        return self.direct_transition_rates

    def perform_random_walks(
        self,
        starting_position_idxs: Union[None, list, np.ndarray] = None,
        number_of_steps: Union[None, int] = None,
        species: Union[str, list] = "all",
    ) -> tuple:
        """Perform a set of random walks using perform_random_walks function.

        M independent walks will be simulated, each taking N steps through the L possible positions
        (i.e. the L voxel indices). The transfer rates are the direct voxel transition rates, with
        a mask applied to remove transitions between nonneighboring voxels. If the starting
        positions are not specified, one walker is placed in all voxels. If the number of steps is
        not specified, the number of steps will be 4 times the number of voxels.

        Parameters
        ----------
        starting_position_idxs (list | np.ndarray, optional): The starting position indices for
            each walker. Default: None.
        number_of_steps (int, optional): The number of steps each walk will take. Default: None.
        species (str, optional): The species to perform the random walks for. Default: "all".

        Returns
        -------
        dict
            key (str): The molecule type.
            value (np.ndarray): NxM array, where an entry is the position index of a walker at a
                given step.
        dict
            key (str): The molecule type.
            value (np.ndarray): NxM array, where an entry is the time of a walker at a given step.
        """
        if starting_position_idxs is None:
            starting_position_idxs = np.arange(
                np.prod(self.voxels.number_of_voxels)
            ).flatten()
        starting_position_idxs = np.array(starting_position_idxs).flatten()
        if number_of_steps is None:
            number_of_steps = 4 * np.prod(self.voxels.number_of_voxels)
        if not isinstance(species, list):
            species = [species]
        if species[0] == "all":
            species = sorted(list(set([mol.kind for mol in self.molecules_list])))
        nonneighbor_mask = np.array(
            [
                [
                    j_idx in self.voxels.neighbors_dict[i_idx]
                    for j_idx in range(np.prod(self.voxels.number_of_voxels))
                ]
                for i_idx in range(np.prod(self.voxels.number_of_voxels))
            ]
        )
        self.random_walk_positions, self.random_walk_times = {}, {}
        for sp in species:
            transfer_rates = np.where(
                nonneighbor_mask == True, self.direct_transition_rates[sp], 0
            )
            self.random_walk_positions[sp], self.random_walk_times[sp] = (
                perform_random_walks(
                    transfer_rates, starting_position_idxs, number_of_steps
                )
            )
        return self.random_walk_positions, self.random_walk_times

    def calculate_average_first_time_between_positions(
        self,
        species: Union[str, list] = "all",
        recursion_interval: int = 1,
    ) -> dict:
        """Calculate the average first time between positions.

        Uses the calculate_average_first_time_between_positions function to calculate the average
        first time between voxels of the random walks previously preformed by the instance. The
        average times are stored as an attribute of the class and returned by this method.

        Parameters
        ----------
        species (str | list, optional): The species on which to operate. Default: "all".
        recursion_interval (int, optional): The interval at which to recurse through the position
            array. Default: 1.

        Returns
        -------
        dict
            key (str): The molecule type.
            value (np.ndarray): LxL array, where entry i,j is the average time a walker travels
                from voxel i to voxel j, for all L possible voxels.
        """

        if not isinstance(species, list):
            species = [species]
        if species[0] == "all":
            species = sorted(list(set([mol.kind for mol in self.molecules_list])))
        self.random_walk_average_first_time_between_positions = {}
        for sp in species:
            self.random_walk_average_first_time_between_positions[sp] = (
                calculate_average_first_time_between_positions(
                    self.random_walk_positions[sp],
                    self.random_walk_times[sp],
                    np.prod(self.voxels.number_of_voxels),
                    recursion_interval=recursion_interval,
                )
            )
        return self.random_walk_average_first_time_between_positions

    def calculate_global_diffusion_rates(
        self,
        starting_position_idxs: Union[None, list, np.ndarray] = None,
        number_of_steps: Union[None, int] = None,
        species: Union[str, list] = "all",
        recursion_interval: int = 1,
    ) -> dict:
        """Calculate the diffusion rates.

        Uses the calculate_diffusion_rates function to calculate the average diffusion rates
        between all voxels. The rates are stored as an attribute of the class and returned by this
        method. If the instance foes not have a random walk average first time between positions
        attribute, this method will call the perform_random_walks and
        calculate_average_first_time_between_positions methods.

        Parameters
        ----------
        starting_position_idxs (list | np.ndarray, optional): The starting position indices for
            each walker. Default: None.
        number_of_steps (int, optional): The number of steps each walk will take. Default: None.
        species (str, optional): The species to perform the random walks for. Default: "all".
        recursion_interval (int, optional): The interval at which to recurse through the position
            array. Default: 1.

        Returns
        -------
        dict
            key (str): The molecule type.
            value (float): Diffusion rate, where entry i,j is the average diffusion rate from voxel
                i to voxel j.
        """

        if not isinstance(species, list):
            species = [species]
        if species[0] == "all":
            species = sorted(list(set([mol.kind for mol in self.molecules_list])))
        if self.random_walk_average_first_time_between_positions is None:
            self.perform_random_walks(
                starting_position_idxs=starting_position_idxs,
                number_of_steps=number_of_steps,
                species=species,
            )
            self.calculate_average_first_time_between_positions(
                species=species, recursion_interval=recursion_interval
            )
        self.global_diffusion_rates = {}
        for sp in species:
            self.global_diffusion_rates[sp] = 1 / (
                self.random_walk_average_first_time_between_positions[sp]
            )
        return self.global_diffusion_rates


def read_mvabfa_file(filename):
    flag = None
    with open(filename, "r") as f:
        for line in f:
            fields = line.split()
            if len(fields) == 0:
                continue
            if fields[0] == "#":
                continue
            if fields[0] == "MoleculeIDs":
                flag = fields[0]
                continue
            if fields[0] == "Timesteps":
                flag = fields[0]
                continue
            if fields[0] == "MVABFA":
                flag = fields[0]
                mvabfa = []
                continue
            if flag == "MoleculeIDs":
                molecule_IDs = np.array(fields, dtype=int).tolist()
                continue
            if flag == "Timesteps":
                timesteps = np.array(fields, dtype=int).tolist()
                continue
            if flag == "MVABFA":
                mvabfa.append([int(_) for _ in fields])
                continue
    mvabfa = np.array(mvabfa)
    return molecule_IDs, timesteps, mvabfa


def read_direct_transition_rates_file(filename):
    parse, molecule = False, None
    rates = {}
    with open(filename, "r") as f:
        for line in f:
            fields = line.split()
            if len(fields) == 0:
                continue
            if fields[0] == "#":
                continue
            if fields[0] == "type":
                parse = True
                molecule = fields[1]
                rates[molecule] = []
                continue
            if parse is True:
                rates[molecule].append([float(_) for _ in fields])
                continue
    for molecule_kind, rate in rates.items():
        rates[molecule_kind] = np.array(rate)
    return rates


def calculate_direct_transition_rates(
    total_number_of_voxels: int,
    molecular_voxel_assignment_by_frame_array: np.ndarray,
    adjacent_transition_time: float,
    molecules_list: list,
) -> dict:
    """
    Calculate the direct voxel transition rates.

    This function calculates the direct voxel transition rates, which are the rates at which
    molecules transition from one voxel to another. This is done by counting the number of
    transitions from one voxel to another for each molecule type, and dividing by time between
    adjacent frames in the trajectory file.

    NOTE No neighbor mask is applied.
    NOTE Rates for a molecule to remain in a voxel (i.e. the matrix diagonal) ARE included.

    Parameters
    ----------
    total_number_of_voxels (int): Total number of voxels.
    molecular_voxel_assignment_by_frame_array (np.ndarray): Array of N frames by M molecules, where
        entry i,j is the voxel in which molecule j is located at frame i.
    adjacent_transition_time (float): Time between adjacent frames.
    molecules_datafile (MoleculeList): MoleculeList instance indexed to the column order of the
        molecular_voxel_assignment_by_frame_array.

    Returns
    -------
    dict
        key (str): The molecule type.
        value (np.ndarray): Direct voxel transition rates, where entry i,j is the average
            transition rate from voxel i to voxel j.
    """

    direct_transition_rates = {}
    direct_transition_counts = {
        _: np.zeros((total_number_of_voxels, total_number_of_voxels))
        for _ in set([mol.kind for mol in molecules_list])
    }
    for mol_kind, direct_transition_counts_here in direct_transition_counts.items():
        mol_idxs = [
            idx for idx, mol in enumerate(molecules_list) if mol.kind == mol_kind
        ]
        positions = molecular_voxel_assignment_by_frame_array[:, mol_idxs]
        for midx in mol_idxs:
            voxel_list = molecular_voxel_assignment_by_frame_array[:, midx].flatten()
            voxel_list_shifted = np.roll(voxel_list, -1)
            transitions = np.column_stack((voxel_list, voxel_list_shifted))
            to_from, count = np.unique(transitions[:-1, :], axis=0, return_counts=True)
            for idx, tf in enumerate((to_from)):
                direct_transition_counts_here[tf[0], tf[1]] += count[idx]
        starting_counts = np.zeros((total_number_of_voxels, 1))
        for idx, val in zip(*np.unique(positions[:-1, :], return_counts=True)):
            starting_counts[idx, 0] = val
        direct_transition_rates[mol_kind] = np.where(
            starting_counts != 0,
            direct_transition_counts_here / adjacent_transition_time / starting_counts,
            0,
        )
    return direct_transition_rates


def perform_random_walks(
    transfer_rates: np.ndarray,
    starting_position_idxs: np.ndarray,
    number_of_steps: int,
) -> tuple:
    """Perform random walks.

    Perform random walks on a set of transfer rates. All rates of "remaining in place" are removed
    (i.e. the diagonal on the transfer rates array is zeroed). M independent walks will be
    simulated, each taking N steps through L possible positions.

    Parameters
    ----------
    transfer_rates (np.ndarray): LxL array, where entry i,j is the average transition rate from
        position i to position j.
    starting_position_idxs (np.ndarray): Mx1 array, where an entry is the starting position index
        for each of M total independent walkers.
    number_of_steps (int): The N steps each walk will take.

    Returns
    -------
    np.ndarray
        NxM array, where entry i,j is the position index of walker j at step i.
    np.ndarray
        NxM array, where entry i,j is the time of walker j at step i.
    """

    np.fill_diagonal(transfer_rates, 0)
    transfer_rates_sum = np.sum(transfer_rates, axis=1)
    transfer_rates_cumsum = np.cumsum(transfer_rates, axis=1)
    walkers_position = np.zeros(
        (number_of_steps + 1, starting_position_idxs.shape[0]), dtype=int
    )
    walkers_position[0] = starting_position_idxs.flatten()
    walkers_time = np.zeros(
        (number_of_steps + 1, starting_position_idxs.shape[0]), dtype=np.float64
    )
    for step in range(number_of_steps):
        u1 = np.random.rand(1, walkers_position.shape[1]).flatten()
        u1 *= np.array([transfer_rates_sum[idx] for idx in walkers_position[step]])
        walkers_position[step + 1] = np.array(
            [
                np.argwhere(transfer_rates_cumsum[walkers_position[step, idx]] >= u)[0][
                    0
                ]
                for idx, u in enumerate(u1)
            ]
        ).flatten()
        u2 = 1 - np.random.rand(1, walkers_position.shape[1]).flatten()
        dt = np.array(
            [
                -np.log(u) / transfer_rates_sum[walkers_position[step, idx]]
                for idx, u in enumerate(u2)
            ]
        ).flatten()
        walkers_time[step + 1] = walkers_time[step] + dt
    return walkers_position, walkers_time


def calculate_average_first_time_between_positions(
    walkers_position: np.ndarray,
    walkers_time: np.ndarray,
    number_of_positions: int,
    recursion_interval: int = 1,
) -> np.ndarray:
    """Calculate the average first time between positions.

    Calculate the average first time between positions given position and time arrays of a set of M
    random walks of N steps each.

    NOTE Position "labels" are expected to be their indices, zero indexed.

    Parameters
    ----------
    walkers_position (np.ndarray): NxM array, where entry i,j is the position index of walker j at
        step j.
    walkers_time (np.ndarray): NxM array, where entry i,j is the time of walker j at step i.
    number_of_positions (int): The number of possible positions.
    recursion_interval (int, optional): The interval at which to recurse through the position
        array. Default: 1.

    Returns
    -------
    np.ndarray
        LxL array, where entry i,j is the average time a walker travels from position i to position
        j, for all L possible positions.
    """

    transition_counts = np.zeros((number_of_positions, number_of_positions))
    transition_times = np.zeros((number_of_positions, number_of_positions))
    for colidx in range(walkers_position.shape[1]):
        zipped_positiontime = list(
            zip(walkers_position[:, colidx], walkers_time[:, colidx])
        )

        for i_idx in range(0, len(zipped_positiontime) - 1, recursion_interval):
            i_value = np.array(zipped_positiontime[i_idx])
            remaining = np.array(zipped_positiontime[i_idx + 1 :])
            first_idx = [
                np.min(np.where(remaining[:, 0] == j_value))
                for j_value in np.unique(remaining[:, 0])
                if j_value != i_value[0]
            ]
            for j_value in remaining[first_idx]:
                transition_counts[int(i_value[0]), int(j_value[0])] += 1
                transition_times[int(i_value[0]), int(j_value[0])] += (
                    j_value[1] - i_value[1]
                )
    return (
        np.where(transition_counts == 0, np.inf, transition_times) / transition_counts
    )


def add_ghost_voxels(adjusters, ghost_voxels_dict, voxel_centers):
    add_array = deepcopy(voxel_centers)
    for idx, adj in enumerate(adjusters):
        add_array[:, idx] += adj
    ghost_voxels_dict.update(
        {
            (idx, len(ghost_voxels_dict) + idx): val.tolist()
            for idx, val in enumerate(add_array)
        }
    )
    return ghost_voxels_dict


def assemble_ghost_voxels_boundaries(voxel_boundaries, box, number_of_voxels):

    # Prep
    voxel_centers = np.mean(voxel_boundaries, axis=2)
    ghost_voxels = {}
    xsize = box[0][1] - box[0][0]
    ysize = box[1][1] - box[1][0]
    zsize = box[2][1] - box[2][0]

    # Add faces
    ghost_voxels = add_ghost_voxels([-xsize, 0, 0], ghost_voxels, voxel_centers)
    ghost_voxels = add_ghost_voxels([+xsize, 0, 0], ghost_voxels, voxel_centers)
    ghost_voxels = add_ghost_voxels([0, -ysize, 0], ghost_voxels, voxel_centers)
    ghost_voxels = add_ghost_voxels([0, +ysize, 0], ghost_voxels, voxel_centers)
    ghost_voxels = add_ghost_voxels([0, 0, -zsize], ghost_voxels, voxel_centers)
    ghost_voxels = add_ghost_voxels([0, 0, +zsize], ghost_voxels, voxel_centers)

    # Add edges
    ghost_voxels = add_ghost_voxels([-xsize, -ysize, 0], ghost_voxels, voxel_centers)
    ghost_voxels = add_ghost_voxels([-xsize, +ysize, 0], ghost_voxels, voxel_centers)
    ghost_voxels = add_ghost_voxels([+xsize, -ysize, 0], ghost_voxels, voxel_centers)
    ghost_voxels = add_ghost_voxels([+xsize, +ysize, 0], ghost_voxels, voxel_centers)
    ghost_voxels = add_ghost_voxels([-xsize, 0, -zsize], ghost_voxels, voxel_centers)
    ghost_voxels = add_ghost_voxels([-xsize, 0, +zsize], ghost_voxels, voxel_centers)
    ghost_voxels = add_ghost_voxels([+xsize, 0, -zsize], ghost_voxels, voxel_centers)
    ghost_voxels = add_ghost_voxels([+xsize, 0, +zsize], ghost_voxels, voxel_centers)
    ghost_voxels = add_ghost_voxels([0, -ysize, -zsize], ghost_voxels, voxel_centers)
    ghost_voxels = add_ghost_voxels([0, -ysize, +zsize], ghost_voxels, voxel_centers)
    ghost_voxels = add_ghost_voxels([0, +ysize, -zsize], ghost_voxels, voxel_centers)
    ghost_voxels = add_ghost_voxels([0, +ysize, +zsize], ghost_voxels, voxel_centers)

    # Add corners
    ghost_voxels = add_ghost_voxels(
        [-xsize, -ysize, -zsize], ghost_voxels, voxel_centers
    )
    ghost_voxels = add_ghost_voxels(
        [-xsize, -ysize, +zsize], ghost_voxels, voxel_centers
    )
    ghost_voxels = add_ghost_voxels(
        [-xsize, +ysize, -zsize], ghost_voxels, voxel_centers
    )
    ghost_voxels = add_ghost_voxels(
        [-xsize, +ysize, +zsize], ghost_voxels, voxel_centers
    )
    ghost_voxels = add_ghost_voxels(
        [+xsize, -ysize, -zsize], ghost_voxels, voxel_centers
    )
    ghost_voxels = add_ghost_voxels(
        [+xsize, -ysize, +zsize], ghost_voxels, voxel_centers
    )
    ghost_voxels = add_ghost_voxels(
        [+xsize, +ysize, -zsize], ghost_voxels, voxel_centers
    )
    ghost_voxels = add_ghost_voxels(
        [+xsize, +ysize, +zsize], ghost_voxels, voxel_centers
    )

    # Only keep ghost voxels touching the original voxels
    ghost_voxel_keys = sorted(ghost_voxels.keys())
    ghost_voxel_centers = np.array([v for k, v in sorted(ghost_voxels.items())])
    center_bounds = np.array(
        [np.min(voxel_centers, axis=0)] + [np.max(voxel_centers, axis=0)]
    )
    vsize_x = xsize / number_of_voxels[0]
    vsize_y = ysize / number_of_voxels[1]
    vsize_z = zsize / number_of_voxels[2]
    new = np.argwhere(
        np.logical_and.reduce(
            (
                ghost_voxel_centers[:, 0] >= center_bounds[0, 0] - vsize_x,
                ghost_voxel_centers[:, 0] <= center_bounds[1, 0] + vsize_x,
                ghost_voxel_centers[:, 1] >= center_bounds[0, 1] - vsize_y,
                ghost_voxel_centers[:, 1] <= center_bounds[1, 1] + vsize_y,
                ghost_voxel_centers[:, 2] >= center_bounds[0, 2] - vsize_z,
                ghost_voxel_centers[:, 2] <= center_bounds[1, 2] + vsize_z,
            )
        )
    ).flatten()
    ghost_idxs = [k[0] for idx, k in enumerate(ghost_voxel_keys) if idx in new]
    ghost_voxel_centers = ghost_voxel_centers[new]

    # Transform from voxel centers to boundaries
    ghost_voxel_minimums = ghost_voxel_centers - np.array(
        [vsize_x / 2, vsize_y / 2, vsize_z / 2]
    )
    ghost_voxel_maximums = ghost_voxel_centers + np.array(
        [vsize_x / 2, vsize_y / 2, vsize_z / 2]
    )
    ghost_voxel_boundaries = np.stack(
        [ghost_voxel_minimums, ghost_voxel_maximums], axis=2
    )

    return ghost_idxs, ghost_voxel_boundaries


def assign_points_to_voxels(points, voxels, fuzz=0.0):
    """Assign points to voxels.

    Parameters
    ----------
    points (np.ndarray): Nx3 array of points to assign to voxels.
    voxels (Voxels): Voxels object to assign points to.
    fuzz (float, optional): Fuzz factor to use when assigning points to voxels. Default: 0.0.
    """

    # Create items
    voxel_boundaries = np.array(deepcopy(voxels.boundaries))
    voxel_idxs = list(range(np.prod(voxels.number_of_voxels)))

    # If fuzzy boundaries are being applied, create ghost voxels
    if fuzz != 0.0:
        ghost_idxs, ghost_boundaries = assemble_ghost_voxels_boundaries(
            voxel_boundaries, voxels.box, voxels.number_of_voxels
        )
        voxel_boundaries = np.concatenate((voxel_boundaries, ghost_boundaries), axis=0)
        voxel_boundaries[:, :, 0] -= fuzz
        voxel_boundaries[:, :, 1] += fuzz
        voxel_idxs += ghost_idxs

    # Prep
    voxel_idxs = np.array(voxel_idxs)
    points = utility.wrap_coordinates(points, voxels.box)

    # Assign
    voxel_assignments = [
        sorted(
            list(
                set(
                    voxel_idxs[
                        np.argwhere(
                            (point[0] >= voxel_boundaries[:, 0, 0])
                            & (point[0] <= voxel_boundaries[:, 0, 1])
                            & (point[1] >= voxel_boundaries[:, 1, 0])
                            & (point[1] <= voxel_boundaries[:, 1, 1])
                            & (point[2] >= voxel_boundaries[:, 2, 0])
                            & (point[2] <= voxel_boundaries[:, 2, 1])
                        )
                    ]
                    .flatten()
                    .tolist()
                )
            )
        )
        for point in points
    ]
    return voxel_assignments


def clean_voxel_assignments_list(voxel_idxs):
    for idx, val in enumerate(voxel_idxs):
        if isinstance(val, int):
            continue
        if len(val) == 1:
            voxel_idxs[idx] = val[0]
            continue
        if len(val) > 1 and voxel_idxs[idx - 1] in val:
            voxel_idxs[idx] = voxel_idxs[idx - 1]
            continue
        voxel_idxs[idx] = np.random.choice(val)
    return voxel_idxs


def calculate_mvabfa_fuzzy_boundary(
    filename_trajectory: str,
    molecules_list: list,
    number_of_voxels: list,
    start: int = 0,
    end: int = -1,
    every: int = 1,
    fuzz: float = 0.0,
    logfile: Union[None, io.FileTracker] = None,
) -> tuple:

    molecules_list = deepcopy(molecules_list)
    atoms_list, _, _, _, _ = interactions.get_interactions_lists_from_molcules_list(
        molecules_list
    )
    adj_list = [
        [
            idx2
            for idx2, atom2 in enumerate(atoms_list)
            if idx2 != idx1 and atom2.molecule_ID == atom1.molecule_ID
        ]
        for idx1, atom1 in enumerate(atoms_list)
    ]
    atomID2idx = {atom.ID: idx for idx, atom in enumerate(atoms_list)}
    mvabfa_dict = {}
    timesteps = []
    generator_step = 1
    for atoms_list_thisframe, timestep, box_thisframe, _ in io.frame_generator(
        filename_trajectory,
        start=start,
        end=end,
        every=every,
        unwrap=True,
        adj_list=adj_list,
        return_prop=False,
    ):
        box_thisframe = box_thisframe.flatten()
        for molecule in molecules_list:
            for atom in molecule.atoms:
                atom.x = atoms_list_thisframe[atomID2idx[atom.ID]].x
                atom.y = atoms_list_thisframe[atomID2idx[atom.ID]].y
                atom.z = atoms_list_thisframe[atomID2idx[atom.ID]].z
            molecule.calculate_cog(box=box_thisframe)
        points = np.array(
            [molecule.cog.flatten().tolist() for molecule in molecules_list]
        )
        mvabfa_dict[int(timestep)] = assign_points_to_voxels(
            points, voxels.Voxels(box_thisframe, number_of_voxels), fuzz=fuzz
        )
        timesteps.append(int(timestep))
        if logfile is not None and generator_step % 20 == 0:
            logfile.write(
                f"    {generator_step} frames parsed... {datetime.datetime.now()}\n"
            )
        generator_step += 1
    mvabfa = np.array(
        [
            clean_voxel_assignments_list([mvabfa_dict[t][idx] for t in timesteps])
            for idx, mol in enumerate(molecules_list)
        ]
    )
    mvabfa = np.transpose(mvabfa)
    return (
        [molecule.ID for molecule in molecules_list],
        timesteps,
        mvabfa,
    )


if __name__ == "__main__":
    main(sys.argv)
