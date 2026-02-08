#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dylan.gilley@gmail.com


import numpy as np
import scipy.spatial.distance as sds
from copy import deepcopy
from itertools import chain


def isfloat_str(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def find_dist_same(geo, box, periodic=[True, True, True]):
    """Calculates MIC distance between points.

    Calculates the minimum image convention (MIC) distance between points,
    as supplied in a single array (hence "same"). Distance is Minkowski
    distance of p=1.0.

    Parameters
    ----------
    geo: np.ndarray
        Nx3 array, where each row holds the x, y, and z coordinates.
    box: list of lists of floats
        [[ xmin, xmax ], ... ]

    Returns
    -------
    np.ndarray
        NxN array, where an entry is the distance between original point
        of row index and original point of column index.
    """

    rs = np.zeros((geo.shape[0], geo.shape[0]))
    for i in range(3):
        dist = sds.squareform(sds.pdist(geo[:, i : i + 1], "minkowski", p=1.0))
        l, l2 = box[i][1] - box[i][0], (box[i][1] - box[i][0]) / 2.0
        while not (dist <= l2).all() and periodic[i] is True:
            dist -= l * (dist > l2)
            dist = np.abs(dist)
        rs += dist**2
    rs = np.sqrt(rs)

    return rs


def flatten_nested_list(list_):
    while True in [isinstance(_, list) for _ in list_]:
        list_ = list(chain.from_iterable(list_))
    return list_


def histo_getbins(times, num_of_bins=100, tmin=None, tmax=None):
    if not tmax:
        tmax = np.min([np.max(_) for _ in times])
    if tmin is None:
        tmin = np.max([np.min(_) for _ in times])
    binsize = (tmax - tmin) / num_of_bins
    return (
        [[tmin + i * binsize, tmin + (i + 1) * binsize] for i in range(num_of_bins)],
        tmin,
        tmax,
    )


def combine_sample_set_statistics(observations, means, stds):
    observation_combined = np.sum(observations, axis=-1)
    mean_combined = np.sum(means * observations, axis=-1) / observation_combined
    variances = stds**2
    ds = (means - mean_combined[..., np.newaxis]) ** 2
    std_combined = np.sqrt(
        np.sum(observations * (variances + ds), axis=-1) / observation_combined
    )
    return observation_combined, mean_combined, std_combined


def calc_Ea(A, k, T):
    # A in same units as k; usually 1/s
    # k in same units as A; usually 1/s
    # T in K
    # Ea in kcal/mol
    R = 0.00198588  # kcal / mol K
    return -R * T * np.log(k / A / T)


def LJtimesteps_to_seconds(LJtimesteps, m, s, e, lammps_stepsize=0.001):
    return LJtimesteps * lammps_stepsize * np.sqrt(m) * s / np.sqrt(e)


def read_direct_voxel_transition_rates(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    direct_voxel_transition_rates = {}
    parsing = False
    for line in lines:
        if line.split() == []:
            continue
        if "Frame" in line:
            frame = int(line.split()[1])
            direct_voxel_transition_rates[frame] = []
            parsing = True
        elif parsing is True:
            direct_voxel_transition_rates[frame].append(
                [float(i) for i in line.split()]
            )
    return {frame: np.array(v) for frame, v in direct_voxel_transition_rates.items()}


def unwrap_coordinates(coordinates, adj_list, box):
    # Unwrap the molecules using the adjacency matrix
    # Loops over the individual atoms and if they haven't been unwrapped yet, performs a walk
    # of the molecular graphs unwrapping based on the bonds.
    box = np.array(box).transpose()
    crystal = []
    geo_final = deepcopy(coordinates)
    L = np.repeat(np.array([box[1, :] - box[0, :]]), coordinates.shape[0], axis=0)
    nL2, pL2 = -0.5 * L[0, :], 0.5 * L[0, :]

    #if adj_list == [[]]:  # Just a single bead
    #    return coordinates, crystal
    # Apply minimum image convension to wrap the coordinates
    unwrapped = []
    for count_i, i in enumerate(coordinates):
        # Skip if this atom has already been unwrapped
        if count_i in unwrapped:
            continue

        # Proceed with a walk of the molecular graph
        # The molecular graph is cumulatively built up in the "unwrap" list and is initially seeded with the current atom
        else:
            unwrap = [count_i]  # list of indices to unwrap (next loop)
            unwrapped += [
                count_i
            ]  # list of indices that have already been unwrapped (first index is left in place)
            for j in unwrap:

                # new holds the index in geo_final of bonded atoms to j that need to be unwrapped
                new = [k for k in adj_list[j] if k not in unwrapped]

                # unwrap the new atoms
                for k in new:
                    unwrapped += [k]
                    dgeo = geo_final[k, :] - geo_final[j, :]

                    check = dgeo < nL2
                    while (check).any():
                        geo_final[k, :][check] += L[k, check]
                        dgeo = geo_final[k, :] - geo_final[j, :]
                        check = dgeo < nL2

                    check = dgeo > pL2
                    while (check).any():
                        geo_final[k, :][check] -= L[k, check]
                        dgeo = geo_final[k, :] - geo_final[j, :]
                        check = dgeo > pL2

                # append the just unwrapped atoms to the molecular graph so that their connections can be looped over and unwrapped.
                unwrap += new

    return geo_final


def wrap_coordinates(coordinates, box):
    if type(coordinates) != np.ndarray:
        try:
            coordinates = np.array(coordinates)
        except:
            raise ValueError("coordinates must be array or convertible to an array")
    if coordinates.shape[1] != 3:
        raise ValueError("coordinates must be Nx3")
    box = flatten_nested_list(box)
    box = [[box[0], box[1]], [box[2], box[3]], [box[4], box[5]]]
    for d in range(3):
        while not np.all(coordinates[:, d] >= box[d][0]):
            coordinates[:, d] = np.where(
                coordinates[:, d] >= box[d][0],
                coordinates[:, d],
                coordinates[:, d] + (box[d][1] - box[d][0]),
            )
        while not np.all(coordinates[:, d] <= box[d][1]):
            coordinates[:, d] = np.where(
                coordinates[:, d] <= box[d][1],
                coordinates[:, d],
                coordinates[:, d] - (box[d][1] - box[d][0]),
            )
    return coordinates
