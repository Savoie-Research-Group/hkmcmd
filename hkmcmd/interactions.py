#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dylan.gilley@gmail.com


import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from hkmcmd import utility


class Atom:

    def __init__(
        self,
        ID=None,
        kind=None,
        lammps_type=None,
        taffi_type=None,
        element=None,
        mass=None,
        molecule_ID=None,
        molecule_kind=None,
        x=None,
        y=None,
        z=None,
        vx=None,
        vy=None,
        vz=None,
        fx=None,
        fy=None,
        fz=None,
        q1=None,
        q2=None,
        q3=None,
        q4=None,
        charge=None,
    ):
        if ID is not None and ID < 1:
            raise ValueError("Atom ID must be greater than or equal to 1")
        self.ID = ID
        self.kind = kind
        self.lammps_type = lammps_type
        self.taffi_type = taffi_type
        self.element = element
        self.mass = mass
        self.molecule_ID = molecule_ID
        self.molecule_kind = molecule_kind
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.q4 = q4
        self.charge = charge
        return

    def wrap(self, box):
        coordinates = np.array([self.x, self.y, self.z]).reshape(1, 3)
        coordinates = utility.wrap_coordinates(coordinates, box)
        self.x, self.y, self.z = coordinates[0]
        return

    def make_jsonable(self):
        return {k: v for k, v in self.__dict__.items()}


class IntraMode:

    def __init__(self, ID=None, kind=None, atom_IDs=None):
        self.ID = ID
        self.kind = kind
        self.atom_IDs = atom_IDs
        return

    def make_jsonable(self):
        return {k: v for k, v in self.__dict__.items()}


class Molecule:

    def __init__(
        self,
        ID=None,
        kind=None,
        atoms=None,
        bonds=None,
        angles=None,
        dihedrals=None,
        impropers=None,
        cog=None,
        voxel_idx=None,
    ):
        self.ID = ID
        self.kind = kind
        self.atoms = atoms
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals
        self.impropers = impropers
        self.cog = cog
        self.voxel_idx = voxel_idx
        self.check_for_json_inputs()
        return

    @property
    def atom_IDs(self):
        return [_.ID for _ in self.atoms]

    def check_for_json_inputs(self):
        if self.atoms is not None:
            if len(set([type(_) for _ in self.atoms])) > 1:
                raise ValueError("Atoms must all be the same type")
            if type(self.atoms[0]) != Atom:
                self.atoms = [Atom(**_) for _ in self.atoms]
        for attr in ["bonds", "angles", "dihedrals", "impropers"]:
            intramode = getattr(self, attr)
            if intramode is not None:
                if type(intramode) is list:
                    if len(intramode) == 0:
                        setattr(self, attr, None)
                        continue
                if len(set([type(_) for _ in intramode])) > 1:
                    raise ValueError(f"{attr} must all be the same type")
                if type(intramode[0]) != IntraMode:
                    setattr(self, attr, [IntraMode(**_) for _ in intramode])
        return

    def make_jsonable(self):
        cog = deepcopy(self.cog)
        if cog is not None:
            cog = cog.tolist()
        dict_ = {
            "ID": self.ID,
            "kind": self.kind,
            "cog": cog,
            "voxel_idx": self.voxel_idx,
        }
        if self.atoms is not None:
            dict_.update(
                {
                    "atoms": [_.make_jsonable() for _ in self.atoms],
                }
            )
        for attr in ["bonds", "angles", "dihedrals", "impropers"]:
            intramode = getattr(self, attr)
            if intramode is not None:
                dict_.update(
                    {
                        attr: [_.make_jsonable() for _ in intramode],
                    }
                )
        return dict_

    def fill_lists(
        self,
        atoms_list=None,
        bonds_list=None,
        angles_list=None,
        dihedrals_list=None,
        impropers_list=None,
    ):
        if atoms_list is not None:
            self.atoms = sorted(
                [atom for atom in atoms_list if atom.molecule_ID == self.ID],
                key=lambda atom: (atom.ID),
            )
        if (
            len(
                set(
                    [
                        atom.molecule_kind
                        for atom in self.atoms
                        if atom.molecule_kind is not None
                    ]
                )
            )
            > 1
        ):
            raise ValueError(
                "Molecule kind must be the same for all atoms in a molecule"
            )
        if self.kind is not None:
            if self.kind != self.atoms[0].molecule_kind:
                raise ValueError("Molecule kind must be the same as atom kind")
        else:
            self.kind = self.atoms[0].molecule_kind
        if bonds_list is not None:
            self.bonds = sorted(
                [
                    bond
                    for bond in bonds_list
                    if set(bond.atom_IDs).issubset(self.atom_IDs)
                ],
                key=lambda bond: (bond.ID),
            )
        if angles_list is not None:
            self.angles = sorted(
                [
                    angle
                    for angle in angles_list
                    if set(angle.atom_IDs).issubset(self.atom_IDs)
                ],
                key=lambda angle: (angle.ID),
            )
        if dihedrals_list is not None:
            self.dihedrals = sorted(
                [
                    dihedral
                    for dihedral in dihedrals_list
                    if set(dihedral.atom_IDs).issubset(self.atom_IDs)
                ],
                key=lambda dihedral: (dihedral.ID),
            )
        if impropers_list is not None:
            self.impropers = sorted(
                [
                    improper
                    for improper in impropers_list
                    if set(improper.atom_IDs).issubset(self.atom_IDs)
                ],
                key=lambda improper: (improper.ID),
            )
        return

    def unwrap_atomic_coordinates(self, box):
        box = utility.flatten_nested_list(box)
        box = [[box[0], box[1]], [box[2], box[3]], [box[4], box[5]]]
        geo = np.array([[_.x, _.y, _.z] for _ in self.atoms]).reshape(-1, 3)
        adj_list = [
            [idx2 for idx2 in list(range(len(self.atoms))) if idx2 != idx1]
            for idx1 in range(len(self.atoms))
        ]
        geo = utility.unwrap_coordinates(geo, adj_list, box)
        for idx, atom in enumerate(self.atoms):
            atom.x, atom.y, atom.z = geo[idx]
        return

    def calculate_cog(self, box=None, wrap=False):
        if box is not None:
            self.unwrap_atomic_coordinates(box)
        self.cog = np.array(
            [
                np.mean([_.x for _ in self.atoms]),
                np.mean([_.y for _ in self.atoms]),
                np.mean([_.z for _ in self.atoms]),
            ]
        ).reshape(1, 3)
        if wrap is True:
            self.cog = utility.wrap_coordinates(self.cog, box)
        return self.cog

    def assign_voxel_idx(self, voxel_instance):
        self.calculate_cog(box=voxel_instance.box, wrap=True)
        self.voxel_idx = list(range(len(voxel_instance.IDs)))
        for dimension in range(3):
            bounds = np.array(voxel_instance.boundaries)[:, dimension, :]
            self.voxel_idx = [
                vidx
                for vidx in self.voxel_idx
                if vidx
                in np.where(
                    (self.cog[0, dimension] >= bounds[:, 0])
                    & (self.cog[0, dimension] < bounds[:, 1])
                )[0].tolist()
            ]
        self.voxel_idx = tuple(sorted(self.voxel_idx))
        return

    def translate(self, new_cog):
        if self.cog is None:
            raise ValueError("Molecule must have a cog to translate atoms to it")
        difference = new_cog - self.cog
        for atom in self.atoms:
            atom.x += difference[0, 0]
            atom.y += difference[0, 1]
            atom.z += difference[0, 2]
        self.calculate_cog()
        return

    def rotate(self, angle_radians, axis):
        coordinates = np.array([[_.x, _.y, _.z] for _ in self.atoms]).reshape(-1, 3)
        coordinates = rotate_molecule(coordinates, angle_radians, axis)
        for idx, atom in enumerate(self.atoms):
            atom.x, atom.y, atom.z = coordinates[idx]
        return

    def adjust_ID(self, new_ID):
        self.ID = new_ID
        for atom in self.atoms:
            atom.molecule_ID = new_ID
        return

    def adjust_atom_IDs(self, map=None, offset=None):
        if map is None and offset is None:
            raise ValueError("Either offset or map must be provided")
        if map is None:
            map = {ID: ID + offset for ID in self.atom_IDs}
        for atom in self.atoms:
            atom.ID = map[atom.ID]
        if self.bonds is not None:
            for bond in self.bonds:
                bond.atom_IDs = [map[ID] for ID in bond.atom_IDs]
        if self.angles is not None:
            for angle in self.angles:
                angle.atom_IDs = [map[ID] for ID in angle.atom_IDs]
        if self.dihedrals is not None:
            for dihedral in self.dihedrals:
                dihedral.atom_IDs = [map[ID] for ID in dihedral.atom_IDs]
        if self.impropers is not None:
            for improper in self.impropers:
                improper.atom_IDs = [map[ID] for ID in improper.atom_IDs]
        return

    def adjust_intramode_IDs(self, map=None, offset=None, mode="bonds"):
        attr = getattr(self, mode)
        if attr is None:
            return
        if map is None and offset is None:
            raise ValueError("Either offset or map must be provided")
        if map is None:
            map = {intramode.ID: intramode.ID + offset for intramode in attr}
        for intramode in attr:
            intramode.ID = map[intramode.ID]
        return

    def clean_IDs(self, new_ID=1):
        self.adjust_ID(new_ID)
        if self.atoms is not None:
            IDmap = {atom.ID: idx + 1 for idx, atom in enumerate(self.atoms)}
            self.adjust_atom_IDs(map=IDmap)
        if self.bonds is not None:
            IDmap = {bond.ID: idx + 1 for idx, bond in enumerate(self.bonds)}
            self.adjust_intramode_IDs(map=IDmap, mode="bonds")
        if self.angles is not None:
            IDmap = {angle.ID: idx + 1 for idx, angle in enumerate(self.angles)}
            self.adjust_intramode_IDs(map=IDmap, mode="angles")
        if self.dihedrals is not None:
            IDmap = {
                dihedral.ID: idx + 1 for idx, dihedral in enumerate(self.dihedrals)
            }
            self.adjust_intramode_IDs(map=IDmap, mode="dihedrals")
        if self.impropers is not None:
            IDmap = {
                improper.ID: idx + 1 for idx, improper in enumerate(self.impropers)
            }
            self.adjust_intramode_IDs(map=IDmap, mode="impropers")
        return

    def print(self):
        print(f"Molecule ID: {self.ID}")
        print(f"Kind: {self.kind}")
        print(f"Center of geometry: {self.cog}")
        print(f"Voxel index: {self.voxel_idx}")

        # atoms
        if self.atoms is not None:
            print("Atoms:")
            for atom in self.atoms:
                print(f"    atom {atom.ID}")
                for k, v in atom.__dict__.items():
                    if v is not None:
                        print(f"        {k}: {v}")
        else:
            print("Atoms: None")

        # bonds
        if self.bonds is not None:
            print("Bonds:")
            for bond in self.bonds:
                print(f"    bond {bond.ID}")
                for k, v in bond.__dict__.items():
                    if v is not None:
                        print(f"        {k}: {v}")
        else:
            print("Bonds: None")

        # angles
        if self.angles is not None:
            print("Angles:")
            for angle in self.angles:
                print(f"    angle {angle.ID}")
                for k, v in angle.__dict__.items():
                    if v is not None:
                        print(f"        {k}: {v}")
        else:
            print("Angles: None")

        # dihedrals
        if self.dihedrals is not None:
            print("Dihedrals:")
            for dihedral in self.dihedrals:
                print(f"    dihedral {dihedral.ID}")
                for k, v in dihedral.__dict__.items():
                    if v is not None:
                        print(f"        {k}: {v}")
        else:
            print("Dihedrals: None")

        # impropers
        if self.impropers is not None:
            print("Impropers:")
            for improper in self.impropers:
                print(f"    improper {improper.ID}")
                for k, v in improper.__dict__.items():
                    if v is not None:
                        print(f"       {k}: {v}")
        else:
            print("Impropers: None")

        return


class Reaction:

    def __init__(
        self,
        ID=None,
        kind=None,
        reactant_molecules=None,
        product_molecules=None,
        Ea=None,
        Ea_units="kcal/mol",
        A=None,
        A_units="1/s",
        b=None,
        event_rate=None,
        event_rate_units="1/s",
        translation=0.5,
        rawrate=None,
    ):
        self.ID = ID
        self.kind = kind
        self.reactant_molecules = reactant_molecules
        self.product_molecules = product_molecules
        self.Ea = Ea
        self.Ea_units = Ea_units
        self.A = A
        self.A_units = A_units
        self.b = b
        self.event_rate = event_rate
        self.event_rate_units = event_rate_units
        self.translation = translation
        self.rawrate = rawrate
        self.check_for_json_inputs()
        return

    def check_for_json_inputs(self):
        if self.reactant_molecules is not None:
            #if len(set([type(_) for _ in self.reactant_molecules])) != 1:
             #   raise ValueError("Reactant molecules must all be the same type")
            if type(self.reactant_molecules[0]) == dict:
                self.reactant_molecules = [
                    Molecule(**_) for _ in self.reactant_molecules
                ]
        if self.product_molecules is not None:
            if len(set([type(_) for _ in self.product_molecules])) != 1:
                raise ValueError("Product molecules must all be the same type")
            if type(self.product_molecules[0]) != Molecule:
                self.product_molecules = [Molecule(**_) for _ in self.product_molecules]
        return

    def calculate_product_positions(self):
        """Calculate final position(s) of product molecule(s)

        NOTE all products are placed in the neighborhood of the position of the last reactant
        listed, according to the HKMCMD cannon
        """
        return [
            np.array(
                [
                    self.reactant_molecules[-1].cog[0, 0]
                    + np.random.uniform(-1, 1, None) * self.translation,
                    self.reactant_molecules[-1].cog[0, 1]
                    + np.random.uniform(-1, 1, None) * self.translation,
                    self.reactant_molecules[-1].cog[0, 2]
                    + np.random.uniform(-1, 1, None) * self.translation,
                ]
            ).reshape(1, -1)
            for _ in self.product_molecules
        ]

    def create_product_molecules(self, template_reaction):
        template_atom_IDs = utility.flatten_nested_list(
            [molecule.atom_IDs for molecule in template_reaction.reactant_molecules]
        )
        real_atom_IDs = utility.flatten_nested_list(
            [molecule.atom_IDs for molecule in self.reactant_molecules]
        )
        if len(set(template_atom_IDs)) != len(template_atom_IDs):
            raise ValueError(
                "Duplicate atom IDs in template reaction reactant molecules"
            )
        if len(set(real_atom_IDs)) != len(real_atom_IDs):
            raise ValueError("Duplicate atom IDs in real reaction reactant molecules")
        template_atom_ID2real_atom_ID = {
            template_ID: real_ID
            for template_ID, real_ID in zip(template_atom_IDs, real_atom_IDs)
        }
        self.product_molecules = deepcopy(template_reaction.product_molecules)
        new_cogs = self.calculate_product_positions()
        for idx, molecule in enumerate(self.product_molecules):

            # molecule info
            molecule.calculate_cog()
            molecule.translate(
                new_cogs[idx]
            )  # adjust molecule cog AND atomic positions

            # atom info
            for atom in molecule.atoms:
                atom.ID = template_atom_ID2real_atom_ID[atom.ID]

            # bond/angle/dihedral/improper info
            for attr in ["bonds", "angles", "dihedrals", "impropers"]:
                intramode_list = getattr(molecule, attr)
                if intramode_list is not None:
                    for intramode in intramode_list:
                        intramode.atom_IDs = [
                            template_atom_ID2real_atom_ID[_] for _ in intramode.atom_IDs
                        ]

        return

    def calculate_raw_rate(self, T, method="Arrhenius"):
        R = 0.00198588  # kcal/mol/K
        if method == "Arrhenius":
            self.rawrate = self.A * T**self.b * np.exp(-self.Ea / T / R)
        return

    def make_jsonable(self):
        dict_ = {}
        for k, v in self.__dict__.items():
            if v is None:
                dict_[k] = None
            elif k == "reactant_molecules":
                dict_[k] = [_.make_jsonable() for _ in v]
            elif k == "product_molecules":
                dict_[k] = [_.make_jsonable() for _ in v]
            else:
                dict_[k] = v
        return dict_


def unwrap_atoms_list(atoms_list, box, adj_list=None):
    if adj_list is None:
        if None in [atom.molecule_ID for atom in atoms_list]:
            raise ValueError(
                "Either adj_list must be provided, or all atoms must have assigned molecule_IDs"
            )
        adj_list = [
            [
                idx_j
                for idx_j, atom_j in enumerate(atoms_list)
                if idx_j != idx_i and atom_j.molecule_ID == atom_i.molecule_ID
            ]
            for idx_i, atom_i in enumerate(atoms_list)
        ]
    coordinates = np.array([[_.x, _.y, _.z] for _ in atoms_list]).reshape(-1, 3)
    coordinates = utility.unwrap_coordinates(coordinates, adj_list, box)
    for idx, atom in enumerate(atoms_list):
        atom.x, atom.y, atom.z = coordinates[idx]
    return atoms_list


def rotate_molecule(coordinates, angle_radians, axis):
    """
    Rotates a molecule around its center of geometry.

    Args:
        coordinates (np.ndarray): A NumPy array of shape (n, 3) representing the
                                 3D coordinates of the molecule's atoms.
        angle_degrees (float): The angle of rotation in degrees.
        axis (np.ndarray): A NumPy array of shape (3,) representing the rotation axis.

    Returns:
        np.ndarray: A NumPy array of shape (n, 3) representing the rotated coordinates.
    """

    if type(axis) == str:
        axis = {
            "x": np.array([1, 0, 0]),
            "y": np.array([0, 1, 0]),
            "z": np.array([0, 0, 1]),
        }.get(axis, None)
    if axis is None:
        raise ValueError("Invalid rotation axis")

    # Calculate the center of geometry
    center_of_geometry = np.mean(coordinates, axis=0)

    # Translate the molecule to the origin
    translated_coordinates = coordinates - center_of_geometry

    # Create rotation object
    rotation = R.from_rotvec(angle_radians * axis / np.linalg.norm(axis))

    # Apply the rotation
    rotated_coordinates = rotation.apply(translated_coordinates)

    # Translate the molecule back to its original position
    final_coordinates = rotated_coordinates + center_of_geometry

    return final_coordinates


def update_molecules_list_IDs(
    molecules_list,
    reset_molecule_IDs=True,
    reset_atom_IDs=False,
    reset_intramode_IDs=None,
    reset_bond_IDs=True,
    reset_angle_IDs=True,
    reset_dihedral_IDs=True,
    reset_improper_IDs=True,
):

    if reset_intramode_IDs is not None:
        reset_bond_IDs = reset_intramode_IDs
        reset_angle_IDs = reset_intramode_IDs
        reset_dihedral_IDs = reset_intramode_IDs
        reset_improper_IDs = reset_intramode_IDs

    atoms_count = 0
    bonds_count = 0
    angles_count = 0
    dihedrals_count = 0
    impropers_count = 0

    for idx, molecule in enumerate(molecules_list):

        # assign molecule ID
        if reset_molecule_IDs is True:
            molecule.adjust_ID(idx + 1)

        # assign atom IDs
        if reset_atom_IDs is True:
            IDmap = {
                atom.ID: 1 + idx + atoms_count
                for idx, atom in enumerate(molecule.atoms)
            }
            molecule.adjust_atom_IDs(map=IDmap)
            atoms_count += len(molecule.atoms)

        # assign bonds IDs
        if molecule.bonds is not None and reset_bond_IDs is True:
            IDmap = {
                bond.ID: 1 + idx + bonds_count
                for idx, bond in enumerate(molecule.bonds)
            }
            molecule.adjust_intramode_IDs(map=IDmap, mode="bonds")
            bonds_count += len(molecule.bonds)

        # assign angles IDs
        if molecule.angles is not None and reset_angle_IDs is True:
            IDmap = {
                angle.ID: 1 + idx + angles_count
                for idx, angle in enumerate(molecule.angles)
            }
            molecule.adjust_intramode_IDs(map=IDmap, mode="angles")
            angles_count += len(molecule.angles)

        # assign dihedrals IDs
        if molecule.dihedrals is not None and reset_dihedral_IDs is True:
            IDmap = {
                dihedral.ID: 1 + idx + dihedrals_count
                for idx, dihedral in enumerate(molecule.dihedrals)
            }
            molecule.adjust_intramode_IDs(map=IDmap, mode="dihedrals")
            dihedrals_count += len(molecule.dihedrals)

        # assign impropers IDs
        if molecule.impropers is not None and reset_improper_IDs is True:
            IDmap = {
                improper.ID: 1 + idx + impropers_count
                for idx, improper in enumerate(molecule.impropers)
            }
            molecule.adjust_intramode_IDs(map=IDmap, mode="impropers")
            impropers_count += len(molecule.impropers)

    return molecules_list


def find_overlaps(coordinates, tolerance=0.0000_0010):
    """
    Find overlapping coordinates in a 3D space.

    Parameters
    ----------
    coordinates : np.ndarray
        Array of shape (N, 3) containing coordinates of N points.
    tolerance : float
        The tolerance within which coordinates are considered overlapping.

    Returns
    -------
    np.ndarray
        Array of shape (M, 2) containing the indices of overlapping coordinates.
    """
    diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    overlaps = np.all(np.abs(diff) <= tolerance, axis=2)
    return np.where(np.triu(overlaps, k=1))


def squeeze_coordinates_into_box(coordinates, box, fudge=0.0000_0010):
    """Squeeze coordinates into a box.

    Parameters
    ----------
    coordinates: np.ndarray
        Array of shape (n_atoms, 3) containing the coordinates of the atoms.
    box: list
        [ [xmin,xmax], [ymin,ymax], [zmin,zmax] ]
    fudge: float, optional
        A small value to ensure coordinates are within the box boundaries
        (ensures remove_overlaps doesn't run forever in the extremely rare
        chance that two particles overlap at xmax, ymax, zmax).
    """
    box = np.array(box)
    for i in range(3):
        coordinates[:, i] = np.where(
            coordinates[:, i] > box[i][1],
            box[i][1] - fudge,
            coordinates[:, i],
        )
        coordinates[:, i] = np.where(
            coordinates[:, i] < box[i][0] + fudge,
            box[i][0] + fudge,
            coordinates[:, i],
        )
    return coordinates


def remove_overlaps(coordinates, box, tolerance=0.0000_0010, maximum_iterations=None):
    """Remove overlaps in the coordinates of atoms in a periodic box.

    Parameters
    ----------
    coordinates: np.ndarray
        Array of shape (n_atoms, 3) containing the coordinates of the atoms.
    box: list
        [ [xmin,xmax], [ymin,ymax], [zmin,zmax] ]
    tolerance: float, optional
        The tolerance within which coordinates are considered overlapping.
    maximum_iterations: int, optional
        The maximum number of iterations to attempt to remove overlaps.
    """
    if maximum_iterations is None:
        maximum_iterations = coordinates.shape[0]

    # Calculate box lengths
    box = np.array(box)
    box_lengths = np.abs(np.diff(box, axis=1)).flatten()

    # Calculate image flags for all atoms
    image_flags = np.zeros_like(coordinates)
    for i in range(3):
        image_flags[:, i] = np.where(
            coordinates[:, i] > box[i][1],
            np.ceil((coordinates[:, i] - box[i][1]) / box_lengths[i]),
            0,
        )
        image_flags[:, i] = np.where(
            coordinates[:, i] < box[i][0],
            np.floor((coordinates[:, i] - box[i][0]) / box_lengths[i]),
            image_flags[:, i],
        )

    # Catch atomic overlaps
    overlaps = True
    iteration = 0
    coordinates = squeeze_coordinates_into_box(coordinates, box, fudge=tolerance*1.01)
    while overlaps:
        iteration += 1
        if iteration > maximum_iterations:
            raise RuntimeError("Maximum iterations reached while removing overlaps.")

        # Wrap coordinates and check for overlaps
        wrapped_coords = coordinates - image_flags * box_lengths
        overlap_idxs_wrapped = find_overlaps(wrapped_coords, tolerance=tolerance)
        if overlap_idxs_wrapped[0].size > 0:
            print(
                f"Found overlapping atoms! Removing {len(overlap_idxs_wrapped[0])} overlaps..."
            )
            coordinates[overlap_idxs_wrapped[0]] += tolerance

        # Unwrap coordinates and check again
        unwrapped_coords = coordinates + image_flags * box_lengths
        overlap_idxs_unwrapped = find_overlaps(unwrapped_coords, tolerance=tolerance)
        if overlap_idxs_unwrapped[0].size > 0:
            print(
                f"Found overlapping atoms! Removing {len(overlap_idxs_unwrapped[0])} overlaps..."
            )
            coordinates[overlap_idxs_unwrapped[0]] += tolerance

        # Check if all overlaps are removed
        if overlap_idxs_wrapped[0].size == 0 and overlap_idxs_unwrapped[0].size == 0:
            overlaps = False

        coordinates = squeeze_coordinates_into_box(coordinates, box, fudge=tolerance*1.01)

    return coordinates


def remove_overlaps_from_molecules_list(
    molecule_list, box, tolerance=0.0000_0010, maximum_iterations=None
):
    atom_list = utility.flatten_nested_list(
        [molecule.atoms for molecule in molecule_list]
    )
    coordinates = np.array([[_.x, _.y, _.z] for _ in atom_list]).reshape(-1, 3)
    coordinates = remove_overlaps(
        coordinates, box, tolerance=tolerance, maximum_iterations=maximum_iterations
    )
    for idx, atom in enumerate(atom_list):
        atom.x, atom.y, atom.z = coordinates[idx]
    return molecule_list


def update_molecules_list_with_reaction(
    molecules_list,
    reactive_event,
    box,
    tolerance=0.0000_0010,
    maximum_iterations=None,
):
    molecules_list = [
        molecule
        for molecule in molecules_list
        if molecule.ID not in [_.ID for _ in reactive_event.reactant_molecules]
    ]
    molecules_list += reactive_event.product_molecules
    molecules_list = update_molecules_list_IDs(molecules_list, reset_atom_IDs=False)
    if tolerance is not None:
        molecules_list = remove_overlaps_from_molecules_list(
            molecules_list, box, tolerance=tolerance, maximum_iterations=maximum_iterations
        )
    return molecules_list


def get_interactions_lists_from_molcules_list(molecules_list):

    # Create interactions lists
    atoms_list = utility.flatten_nested_list(
        [molecule.atoms for molecule in molecules_list]
    )
    bonds_list = utility.flatten_nested_list(
        [molecule.bonds for molecule in molecules_list if molecule.bonds is not None]
    )
    angles_list = utility.flatten_nested_list(
        [molecule.angles for molecule in molecules_list if molecule.angles is not None]
    )
    dihedrals_list = utility.flatten_nested_list(
        [
            molecule.dihedrals
            for molecule in molecules_list
            if molecule.dihedrals is not None
        ]
    )
    impropers_list = utility.flatten_nested_list(
        [
            molecule.impropers
            for molecule in molecules_list
            if molecule.impropers is not None
        ]
    )

    # Sort lists by IDs
    atoms_list.sort(key=lambda atom: (atom.ID))
    bonds_list.sort(key=lambda bond: (bond.ID))
    angles_list.sort(key=lambda angle: (angle.ID))
    dihedrals_list.sort(key=lambda dihedral: (dihedral.ID))
    impropers_list.sort(key=lambda improper: (improper.ID))

    # Check for duplicates
    if len(atoms_list) != len(set([atom.ID for atom in atoms_list])):
        raise ValueError("Duplicate atom IDs")
    if len(bonds_list) != len(set([bond.ID for bond in bonds_list])):
        raise ValueError("Duplicate bond IDs")
    if len(angles_list) != len(set([angle.ID for angle in angles_list])):
        raise ValueError("Duplicate angle IDs")
    if len(dihedrals_list) != len(set([dihedral.ID for dihedral in dihedrals_list])):
        raise ValueError("Duplicate dihedral IDs")
    if len(impropers_list) != len(set([improper.ID for improper in impropers_list])):
        raise ValueError("Duplicate improper IDs")

    return atoms_list, bonds_list, angles_list, dihedrals_list, impropers_list
