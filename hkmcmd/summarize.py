#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dylan.gilley@gmail.com


import sys, datetime
import numpy as np
from hkmcmd import io, system


def main(argv):
    parser = io.HkmcmdArgumentParser()
    args = parser.parse_args()
    summarize_system_state(args.system, args.prefix)
    return


def summarize_system_state(system_name, prefix):
    """
    Summarize the system data and state.
    
    Parameters:
    system (str): The name of the system.
    prefix (str): The prefix for the output files.
    
    Returns:
    None
    """

    # Read the system data and state
    system_data = system.SystemData(system_name, f"{prefix}")
    system_data.read_json()
    system_state = system.SystemState(f"{prefix}.system_state.json")
    system_state.read_data_from_json()
    system_state.clean()

    # Summarize the system state
    diffusion_cycles = np.array(system_state.diffusion_steps, dtype=int).reshape(-1,1)
    reaction_cycles = np.array(system_state.reactive_steps, dtype=int).reshape(-1,1)
    species_list = sorted([mlc.kind for mlc in system_data.species])
    count_array = np.zeros((system_state.molecule_kinds.shape[0], len(species_list)), dtype=int)
    for idx, kind in enumerate(species_list):
        count_array[:, idx] = np.sum(system_state.molecule_kinds == kind, axis=1)
    try:
        reaction_selections = np.array([system_state.reaction_kinds[np.argwhere(row == 1)[0]][0] for row in system_state.reaction_selections[1:]], dtype=int)
        reaction_selections = np.insert(reaction_selections, 0, 0).reshape(-1,1)
    except:
        reaction_selections = np.zeros((diffusion_cycles.shape))
    times = system_state.times.reshape(-1,1)
    data = np.hstack((diffusion_cycles, reaction_cycles, count_array, reaction_selections, times))

    # Write the summary to a file
    with open(f"{prefix}.summary.txt", "w") as f:
        np.savetxt(
            f,
            data, 
            fmt=["%d"]*2 + ["%d"]*len(species_list) + ["%d"] + ["%.16e"],
            header=f"File written {datetime.datetime.now()}\n" + "diffusion_cycles reaction_cycles " + " ".join(species_list) + " reaction_selections times",
        )
    return


if __name__ == "__main__":
    main(sys.argv[1:])
