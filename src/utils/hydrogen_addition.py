import functools
import pickle
import warnings
import numpy as np
import torch
from os.path import abspath, dirname, join
import itertools
import string
from biotite.structure import BondType
import biotite.structure as struc
import biotite.structure.io.pdb as pdb



class AtomNameLibrary:
    """
    A library for generating hydrogen atom names.

    For each molecule added to the :class:`AtomNameLibrary`,
    the hydrogen atom names are saved for each heavy atom in this
    molecule.

    If hydrogen atom names should be generated for a heavy atom,
    the library first looks for a corresponding entry in the library.
    If such entry is not found, since the molecule was never added to
    the library, the hydrogen atom names are guessed based on common
    hydrogen naming schemes.
    """

    def __init__(self):
        self._name_dict = {}

    @functools.cache
    @staticmethod
    def standard_library():
        """
        Get the standard :class:`AtomNameLibrary`.
        The library contains atom names for the most prominent molecules
        including amino acids and nucleotides.

        Returns
        -------
        library : AtomNameLibrary
            The standard library.
        """
        name_library = AtomNameLibrary()
        file_name = join(dirname(abspath(__file__)), "names.pickle")
        with open(file_name, "rb") as names_file:
            name_library._name_dict = pickle.load(names_file)
        return name_library
    
    def generate_hydrogen_names(self, heavy_res_name, heavy_atom_name):
        """
        Generate hydrogen atom names for the given residue and heavy
        atom name.

        If the residue is not found in the library, the hydrogen atom
        name is guessed based on common hydrogen naming schemes.
        """
        hydrogen_names = self._name_dict.get((heavy_res_name, heavy_atom_name))
        if hydrogen_names is not None:
            # Hydrogen names from library
            for i, hydrogen_name in enumerate(hydrogen_names):
                yield hydrogen_name
            try:
                base_name = hydrogen_name[:-1]
                for number in itertools.count(int(hydrogen_name[-1]) + 1):
                    # Proceed by increasing the atom number
                    # e.g. CB -> HB1, HB2, HB3, ...
                    yield f"{base_name}{number}"
            except ValueError:
                # Atom name has no number at the end
                # -> simply append number
                for number in itertools.count(1):
                    yield f"{hydrogen_name}{number}"

        else:
            if len(heavy_atom_name) == 0:
                # Atom array has no atom names
                # (loaded e.g. from MOL file)
                # -> Also no atom names for hydrogen atoms
                while True:
                    yield ""
            if heavy_atom_name[-1] in string.digits:
                # Atom name ends with number
                # -> assume ligand atom naming
                # C1 -> H1, H1A, H1B
                number = int("".join([c for c in heavy_atom_name if c.isdigit()]))
                heavy_atom_name[0]
                # C1 -> H1, H1A, H1B
                yield f"H{number}"
                for i in itertools.count():
                    yield f"H{number}{string.ascii_uppercase[i]}"
            elif len(heavy_atom_name) > 1:
                # e.g. CA -> HA, HA2, HA3, ...
                suffix = heavy_atom_name[1:]
                yield f"H{suffix}"
                for number in itertools.count(1):
                    yield f"H{suffix}{number}"

            else:
                # N -> H, H2, H3, ...
                yield "H"
                for number in itertools.count(1):
                    yield f"H{number}"


class FragmentLibrary:
    """
    A molecule fragment library for estimation of hydrogen positions.

    For each molecule added to the :class:`FragmentLibrary`,
    the molecule is split into fragments.
    Each fragment consists of

        - A central heavy atom,
        - bond order and position of its bonded heavy atoms and
        - and positions of bonded hydrogen atoms.

    The properties of the fragment (central atom element,
    central atom charge, order of connected bonds) are stored in
    a dictionary mapping these properties to heavy and hydrogen atom
    positions.

    If hydrogen atoms should be added to a target structure,
    the target structure is also split into fragments.
    Now the corresponding reference fragment in the library dictionary
    is accessed for each fragment.
    The corresponding atom coordinates of the reference fragment
    are superimposed [1]_ [2]_ onto the target fragment to obtain the
    hydrogen coordinates for the heavy atom.

    The constructor of this class creates an empty library.

    References
    ----------

    .. [1] W Kabsch,
       "A solution for the best rotation to relate two sets of vectors."
       Acta Cryst, 32, 922-923 (1976).

    .. [2] W Kabsch,
       "A discussion of the solution for the best rotation to relate
       two sets of vectors."
       Acta Cryst, 34, 827-828 (1978).
    """

    def __init__(self):
        self._frag_dict = {}

    @functools.cache
    @staticmethod
    def standard_library():
        """
        Get the standard :class:`FragmentLibrary`.
        The library contains fragments from all molecules in the
        *RCSB* *Chemical Component Dictionary*.

        Returns
        -------
        library : FragmentLibrary
            The standard library.
        """
        fragment_library = FragmentLibrary()
        file_name = join(dirname(abspath(__file__)), "fragments.pickle")
        with open(file_name, "rb") as fragments_file:
            fragment_library._frag_dict = pickle.load(fragments_file)
        return fragment_library
    
    def calculate_hydrogen_coord(self, coords, bonds, names, elements, res_name, device):
        assert len(coords) == len(names)
        # The target and reference heavy atom coordinates
        # for each fragment
        tar_frag_center_coord = torch.zeros((len(coords), 3), dtype=torch.float32, device=device)
        tar_frag_heavy_coord = torch.zeros((len(coords), 3, 3), dtype=torch.float32, device=device)
        ref_frag_heavy_coord = torch.zeros((len(coords), 3, 3), dtype=torch.float32, device=device)
        # The amount of hydrogens varies for each fragment
        # -> padding with NaN
        # The maximum number of bond hydrogen atoms is 4
        ref_frag_hydrogen_coord = torch.full(
            (len(coords), 4, 3), torch.nan, dtype=torch.float32, device=device
        )

        # Fill the coordinate arrays
        fragments = _fragment(coords, bonds, names, elements, res_name, device)
        for i, fragment in enumerate(fragments):
            if fragment is None:
                # This atom is not in mask
                continue
            (
                central_element,
                # central_charge,
                stereo,
                bond_types,
                center_coord,
                heavy_coord,
                _,
            ) = fragment
            tar_frag_center_coord[i] = center_coord
            tar_frag_heavy_coord[i] = heavy_coord
            # The hydrogen_coord can be ignored:
            # In the target structure are no hydrogen atoms
            hit = self._frag_dict.get(
                (central_element, 0, stereo, tuple(bond_types)) #(central_element, central_charge, stereo, tuple(bond_types))
            )
            if hit is None:
                warnings.warn(
                    f"Missing fragment for atom '{names[i]}' "
                    f"at position {i}"
                )
            else:
                _, _, ref_heavy_coord, ref_hydrogen_coord = hit
                ref_hydrogen_coord = torch.from_numpy(ref_hydrogen_coord)
                ref_heavy_coord = torch.from_numpy(ref_heavy_coord)
                ref_frag_heavy_coord[i] = ref_heavy_coord
                ref_frag_hydrogen_coord[i, : len(ref_hydrogen_coord)] = (
                    ref_hydrogen_coord
                )
                    
        tar_frag_heavy_coord = displacement(
            tar_frag_center_coord[:, None, :], tar_frag_heavy_coord
        )
                
        # Get the rotation matrix required for superimposition of
        # the reference coord to the target coord
        matrices = _get_rotation_matrices(tar_frag_heavy_coord, ref_frag_heavy_coord)
        
        # Rotate the reference hydrogen atoms, so they fit the
        # target heavy atoms
        tar_frag_hydrogen_coord = _rotate(ref_frag_hydrogen_coord, matrices)
        # Translate hydrogen atoms to the position of the
        # non-centered central heavy target atom
        tar_frag_hydrogen_coord = tar_frag_hydrogen_coord + tar_frag_center_coord[:, None, :]

        # Turn into list and remove NaN paddings
        tar_frag_hydrogen_coord = [
            # If the x-coordinate is NaN it is expected that
            # y and z are also NaN
            coord[~torch.isnan(coord[:, 0])]
            for coord in tar_frag_hydrogen_coord
        ]

        return tar_frag_hydrogen_coord
    
    def calculate_hydrogen_coord_batch(self, coords, bonds, names, elements, res_name, device, test=False):
        assert coords.shape[1] == len(names)
        atoms_num = coords.shape[1]
        # The target and reference heavy atom coordinates
        # for each fragment
        tar_frag_center_coord = torch.zeros((coords.shape[0], atoms_num, 3), dtype=torch.float32, device=device)
        tar_frag_heavy_coord = torch.zeros((coords.shape[0], atoms_num, 3, 3), dtype=torch.float32, device=device)
        ref_frag_heavy_coord = torch.zeros((coords.shape[0], atoms_num, 3, 3), dtype=torch.float32, device=device)
        # The amount of hydrogens varies for each fragment
        # -> padding with NaN
        # The maximum number of bond hydrogen atoms is 4
        ref_frag_hydrogen_coord = torch.full(
            (coords.shape[0], atoms_num, 4, 3), 0, dtype=torch.float32, device=device
        )

        # Fill the coordinate arrays
        fragments = _fragment_batch(coords, bonds, names, elements, res_name, device)
        for i, fragment in enumerate(fragments):
            if fragment is None:
                # This atom is not in mask
                continue
            (
                central_element,
                # central_charge,
                stereo,
                bond_types,
                center_coord,
                heavy_coord,
                _,
            ) = fragment
            
            temp = torch.zeros_like(tar_frag_center_coord)
            temp[:, i] = center_coord
            tar_frag_center_coord = tar_frag_center_coord + temp
            temp = torch.zeros_like(tar_frag_heavy_coord)
            temp[:, i] = heavy_coord
            tar_frag_heavy_coord = tar_frag_heavy_coord + temp
            # The hydrogen_coord can be ignored:
            # In the target structure are no hydrogen atoms
            hit = [self._frag_dict.get((central_element, 0, int(val), tuple(bond_types))) for val in stereo] 
            if None in hit:
                warnings.warn(
                    f"Missing fragment for atom '{names[i]}' "
                    f"at position {i}"
                )
            else:
                ref_heavy_coord = torch.from_numpy(np.stack([h[2] for h in hit])).to(device)
                ref_hydrogen_coord = torch.from_numpy(np.stack([h[3] for h in hit]))
                # if ref_hydrogen_coord.shape == torch.Size([6, 0]):
                #     ref_hydrogen_coord = ref_hydrogen_coord.reshape(6, 0, 3)
                temp = torch.zeros_like(ref_frag_heavy_coord)
                temp[:, i] = ref_heavy_coord
                ref_frag_heavy_coord = ref_frag_heavy_coord + temp
                temp = torch.full(
                    (coords.shape[0], atoms_num, 4, 3), 0, dtype=torch.float32, device=device
                )
                temp[:, i, : ref_hydrogen_coord.shape[1]]= (ref_hydrogen_coord)
                mask = torch.zeros_like(ref_frag_hydrogen_coord)
                mask[:, i, : ref_hydrogen_coord.shape[1]] = 1
                ref_frag_hydrogen_coord = torch.where(mask.to(torch.bool), temp, ref_frag_hydrogen_coord)
                
                    
        tar_frag_heavy_coord = displacement(
            tar_frag_center_coord[:, :, None, :], tar_frag_heavy_coord
        )
                
        # Get the rotation matrix required for superimposition of
        # the reference coord to the target coord
        matrices = _get_rotation_matrices_batch(tar_frag_heavy_coord, ref_frag_heavy_coord, test)

        # Rotate the reference hydrogen atoms, so they fit the
        # target heavy atoms
        tar_frag_hydrogen_coord = _rotate_batch(ref_frag_hydrogen_coord, matrices)

        # Translate hydrogen atoms to the position of the
        # non-centered central heavy target atom
        mask = (tar_frag_hydrogen_coord.abs().sum(dim=-1) != 0)
        expanded_center_coord = tar_frag_center_coord[:, :, None, :] 
        tar_frag_hydrogen_coord = torch.where(mask.unsqueeze(-1), tar_frag_hydrogen_coord + expanded_center_coord, tar_frag_hydrogen_coord)

        tar_frag_hydrogen_coord = list(tar_frag_hydrogen_coord)
        # Turn into list and remove NaN paddings
        tar_frag_hydrogen_coord = [[
            # If the x-coordinate is NaN it is expected that
            # y and z are also NaN
            coord[~(coord == 0).all(dim=1)]
            for coord in sample] for sample in tar_frag_hydrogen_coord]

        naming_sample = tar_frag_hydrogen_coord[0]
        tar_frag_hydrogen_coord = torch.stack([torch.cat(s, dim=0) for s in tar_frag_hydrogen_coord], dim=0)
        return tar_frag_hydrogen_coord, naming_sample
    
def _fragment(coords, bonds, names, elements, res_name, device):
    """
    Create fragments for the input structure/molecule.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,)
        The structure to be fragmented.
        The structure must have an associated :class:`BondList`.
        The structure must also include the *charge* annotation
        array, depicting the formal charge for each atom.
    mask : ndarray, shape=(n,), dtype=bool
        A boolean mask that is true for each heavy atom for which a
        fragment should be created.

    Returns
    -------
    fragments : list of tuple(str, int, int, ndarray, ndarray, ndarray), length=n
        The fragments.
        The tuple elements are

            #. the central atom element,
            #. the central atom charge,
            #. the enantiomer for stereocenter
               (``-1`` and ``1`` based on a custom nomenclature),
            #. the :class:`BondType` for each bonded heavy atom,
            #. the coordinates of the central atom,
            #. 3 coordinates of bonded heavy atoms (includes padding
               values, if there are not enough heavy atoms),
            #. the coordinates of bonded hydrogen atoms.

        ``None`` for each atom not included by the `mask`.
    """

    fragments = [None] * len(coords)

    all_bond_indices, all_bond_types = bonds.get_all_bonds()
    # Always convert to upper case to make the fragment matching case-insensitive
    elements = np.char.upper(elements)
    # charges = atoms.charge

    for i in range(len(coords)):
        if elements[i] == "H":
            # Only create fragments for heavy atoms
            continue
        bond_indices = all_bond_indices[i]
        bond_types = all_bond_types[i]
        bond_indices = bond_indices[bond_indices != -1]
        bond_types = bond_types[bond_types != -1]

        heavy_mask = elements[bond_indices] != "H"
        heavy_indices = bond_indices[heavy_mask]
        heavy_types = bond_types[heavy_mask]
        if (heavy_types == BondType.ANY).any():
            warnings.warn(
                f"Atom '{names[i]}' in '{res_name[i]}' has an "
                f"undefined bond type and is ignored"
            )
            continue

        # Order the bonded atoms by their bond types
        # to remove atom order dependency in the matching step
        order = np.argsort(heavy_types)
        heavy_indices = heavy_indices[order]
        heavy_types = heavy_types[order]
        # heavy_types = torch.from_numpy(heavy_types)

        hydrogen_mask = ~heavy_mask
        hydrogen_coord = coords[bond_indices[hydrogen_mask]]

        # Special handling of nitrogen as central atom:
        # There are cases where the free electron pair can form
        # a partial double bond.
        # Although the bond order is formally 1 in this case,
        # it would enforce planar hydrogen positionioning
        # Therefore, a partial double bond is handled as bond type 7
        if elements[i] == "N":
            for j, remote_index in enumerate(heavy_indices):
                if heavy_types[j] != 1:
                    # This handling only applies to single bonds
                    continue
                rem_bond_indices = all_bond_indices[remote_index]
                rem_bond_indices = rem_bond_indices[rem_bond_indices != -1]
                rem_bond_types = all_bond_types[remote_index]
                rem_bond_types = rem_bond_types[rem_bond_types != -1]
                for rem_rem_index, bond_type in zip(rem_bond_indices, rem_bond_types):
                    # If the adjacent atom has a double bond
                    # the partial double bond condition is fulfilled
                    if (
                        bond_type == BondType.AROMATIC_DOUBLE
                        or bond_type == BondType.DOUBLE
                    ):
                        heavy_types[j] = 7

        n_heavy_bonds = np.count_nonzero(heavy_mask)
        if n_heavy_bonds == 0:
            # The orientation is arbitrary
            # -> The fragment coord is the coord of the central atom
            # 4 times repeated
            heavy_coord = torch.repeat(coords[None, i, :], 3, axis=0)
            stereo = 0
        elif n_heavy_bonds == 1:
            # Include one atom further away
            # to get an unambiguous fragment
            remote_index = heavy_indices[0]
            rem_bond_indices = all_bond_indices[remote_index]
            rem_bond_indices = rem_bond_indices[rem_bond_indices != -1]
            rem_heavy_mask = elements[rem_bond_indices] != "H"
            rem_heavy_indices = rem_bond_indices[rem_heavy_mask]
            # Use the coord of any heavy atom bonded to the remote
            # atom
            rem_rem_index = rem_heavy_indices[0]
            # Include the directly bonded atom two times, to give it a
            # greater weight in superimposition
            heavy_coord = coords[[remote_index, remote_index, rem_rem_index]]
            stereo = 0
        elif n_heavy_bonds == 2:
            heavy_coord = coords[[heavy_indices[0], heavy_indices[1], i]]
            stereo = 0
        elif n_heavy_bonds == 3:
            heavy_coord = coords[heavy_indices]
            center = coords[i]
            # Determine the enantiomer of this stereocenter
            # For performance reasons, the result does not follow the
            # R/S nomenclature, but a custom -1/1 based one, which also
            # unambiguously identifies the enantiomer
            n = torch.cross(heavy_coord[0] - center, heavy_coord[1] - center)
            stereo = int(torch.sign(torch.dot(heavy_coord[2] - center, n)).item())  
        elif n_heavy_bonds == 4:
            # The fragment is irrelevant, as there is no bonded hydrogen
            # -> The fragment coord is the coord of the central atom
            # 4 times repeated
            heavy_coord = torch.repeat(coords[None, i, :], 3, axis=0)
            stereo = 0
        else:
            warnings.warn(
                f"Atom '{names[i]}' in "
                f"'{res_name[i]}' has more than 4 bonds to "
                f"heavy atoms ({n_heavy_bonds}) and is ignored"
            )
            heavy_coord = torch.repeat(coords[None, i, :], 3, axis=0)
            hydrogen_coord = torch.zeros((0, 3), dtype=torch.float32, device=device)
            stereo = 0
        central_coord = coords[i]
        fragments[i] = (
            elements[i],
            # charges[i],
            stereo,
            heavy_types,
            central_coord,
            heavy_coord,
            hydrogen_coord,
        )
    return fragments


def _get_rotation_matrices(fixed, mobile):
    """
    Get the rotation matrices to superimpose the given mobile
    coordinates into the given fixed coordinates, minimizing the RMSD.

    Uses the *Kabsch* algorithm.

    Parameters
    ----------
    fixed : torch.Tensor, shape=(m,n,3), dtype=torch.float32
        The fixed coordinates.
    mobile : torch.Tensor, shape=(m,n,3), dtype=torch.float32
        The mobile coordinates.

    Returns
    -------
    matrices : torch.Tensor, shape=(m,3,3), dtype=torch.float32
        The rotation matrices.
    """
    # Calculate cross-covariance matrices
    cov = torch.sum(fixed[:, :, :, None] * mobile[:, :, None, :], dim=1)
    v, s, w = torch.linalg.svd(cov, full_matrices=True, driver="gesvdj")
    
    # Remove possibility of reflected atom coordinates
    reflected_mask = torch.det(v) * torch.det(w) < 0
    v[reflected_mask, :, -1] = v[reflected_mask, :, -1]* -1
    matrices = torch.matmul(v, w)
    
    return matrices

def _get_rotation_matrices_batch(fixed, mobile, test):
    """
    Get the rotation matrices to superimpose the given mobile
    coordinates into the given fixed coordinates, minimizing the RMSD.

    Uses the *Kabsch* algorithm.

    Parameters
    ----------
    fixed : torch.Tensor, shape=(m,n,3), dtype=torch.float32
        The fixed coordinates.
    mobile : torch.Tensor, shape=(m,n,3), dtype=torch.float32
        The mobile coordinates.

    Returns
    -------
    matrices : torch.Tensor, shape=(m,3,3), dtype=torch.float32
        The rotation matrices.
    """
    # Calculate cross-covariance matrices
    cov = torch.sum(fixed[:, :, :, :, None] * mobile[:, :, :, None, :], dim=2)
    if test:
        dev = cov.device
        cov = cov.detach().cpu().numpy()
        v, s, w = np.linalg.svd(cov)
        v = torch.from_numpy(v).to(dev)
        w = torch.from_numpy(w).to(dev) 
    else:
        v, s, w = torch.linalg.svd(cov, full_matrices=True, driver="gesvdj")
    
    # Remove possibility of reflected atom coordinates
    reflected_mask = torch.det(v) * torch.det(w) < 0
    # Flip the last column of v wherever reflected_mask is True
    
    temp = torch.ones_like(v)
    t = temp[reflected_mask]
    t[:,:,-1] = -1
    temp[reflected_mask] = t
    v = v * temp
    matrices = torch.matmul(v, w)
    
    # sanity check
    # matrices = []
    # for f,m in zip(fixed, mobile):
    #     matrices+= [_get_rotation_matrices(f,m)]
    # matrices = torch.stack(matrices, dim=0)
    
    return matrices


def _rotate(coord, matrices):
    """
    Apply a rotation on given coordinates.

    Parameters
    ----------
    coord : torch.Tensor, shape=(m,n,3), dtype=torch.float32
        The coordinates.
    matrices : torch.Tensor, shape=(m,3,3), dtype=torch.float32
        The rotation matrices.
    """
    return torch.transpose(
        torch.matmul(matrices, torch.transpose(coord, 1, 2)), 1, 2
    )
    
def _rotate_batch(coord, matrices):
    """
    Apply a rotation on given coordinates.

    Parameters
    ----------
    coord : torch.Tensor, shape=(m,n,3), dtype=torch.float32
        The coordinates.
    matrices : torch.Tensor, shape=(m,3,3), dtype=torch.float32
        The rotation matrices.
    """
    return torch.transpose(
        torch.matmul(matrices, torch.transpose(coord, 2, 3)), 2, 3
    )

def displacement(v1, v2):
    """
    Measure the displacement vector, i.e. the vector difference, from
    one array of atom coordinates to another array of coordinates.

    Returns
    -------
    disp : ndarray, shape=(m,n,3) or ndarray, shape=(n,3) or ndarray, shape=(3,)
        The displacement vector(s). The shape is equal to the shape of
        the input `atoms` with the highest dimensionality.

    index_displacement
    """
    # Decide subtraction order based on shape, since an array can be
    # only subtracted by an array with less dimensions
    if len(v1.shape) <= len(v2.shape):
        diff = v2 - v1
    else:
        diff = -(v1 - v2)
    return diff


    
    
def _fragment_batch(coords, bonds, names, elements, res_name, device):
    """
    Create fragments for the input structure/molecule.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,)
        The structure to be fragmented.
        The structure must have an associated :class:`BondList`.
        The structure must also include the *charge* annotation
        array, depicting the formal charge for each atom.
    mask : ndarray, shape=(n,), dtype=bool
        A boolean mask that is true for each heavy atom for which a
        fragment should be created.

    Returns
    -------
    fragments : list of tuple(str, int, int, ndarray, ndarray, ndarray), length=n
        The fragments.
        The tuple elements are

            #. the central atom element,
            #. the central atom charge,
            #. the enantiomer for stereocenter
               (``-1`` and ``1`` based on a custom nomenclature),
            #. the :class:`BondType` for each bonded heavy atom,
            #. the coordinates of the central atom,
            #. 3 coordinates of bonded heavy atoms (includes padding
               values, if there are not enough heavy atoms),
            #. the coordinates of bonded hydrogen atoms.

        ``None`` for each atom not included by the `mask`.
    """

    fragments = [None] * coords.shape[1]

    all_bond_indices, all_bond_types = bonds.get_all_bonds()
    # Always convert to upper case to make the fragment matching case-insensitive
    elements = np.char.upper(elements)
    # charges = atoms.charge

    for i in range(coords.shape[1]):
        if elements[i] == "H":
            # Only create fragments for heavy atoms
            continue
        bond_indices = all_bond_indices[i]
        bond_types = all_bond_types[i]
        bond_indices = bond_indices[bond_indices != -1]
        bond_types = bond_types[bond_types != -1]

        heavy_mask = elements[bond_indices] != "H"
        heavy_indices = bond_indices[heavy_mask]
        heavy_types = bond_types[heavy_mask]
        if (heavy_types == BondType.ANY).any():
            warnings.warn(
                f"Atom '{names[i]}' in '{res_name[i]}' has an "
                f"undefined bond type and is ignored"
            )
            continue

        # Order the bonded atoms by their bond types
        # to remove atom order dependency in the matching step
        order = np.argsort(heavy_types)
        heavy_indices = heavy_indices[order]
        heavy_types = heavy_types[order]

        hydrogen_mask = ~heavy_mask
        hydrogen_coord = coords[:, bond_indices[hydrogen_mask]]

        # Special handling of nitrogen as central atom:
        # There are cases where the free electron pair can form
        # a partial double bond.
        # Although the bond order is formally 1 in this case,
        # it would enforce planar hydrogen positionioning
        # Therefore, a partial double bond is handled as bond type 7
        if elements[i] == "N":
            for j, remote_index in enumerate(heavy_indices):
                if heavy_types[j] != 1:
                    # This handling only applies to single bonds
                    continue
                rem_bond_indices = all_bond_indices[remote_index]
                rem_bond_indices = rem_bond_indices[rem_bond_indices != -1]
                rem_bond_types = all_bond_types[remote_index]
                rem_bond_types = rem_bond_types[rem_bond_types != -1]
                for rem_rem_index, bond_type in zip(rem_bond_indices, rem_bond_types):
                    # If the adjacent atom has a double bond
                    # the partial double bond condition is fulfilled
                    if (
                        bond_type == BondType.AROMATIC_DOUBLE
                        or bond_type == BondType.DOUBLE
                    ):
                        heavy_types[j] = 7

        n_heavy_bonds = np.count_nonzero(heavy_mask)
        if n_heavy_bonds == 0:
            # The orientation is arbitrary
            # -> The fragment coord is the coord of the central atom
            # 4 times repeated
            heavy_coord = torch.repeat(coords[:, None, i, :], 3, axis=0)
            stereo = torch.zeros((coords.shape[0]))
        elif n_heavy_bonds == 1:
            # Include one atom further away
            # to get an unambiguous fragment
            remote_index = heavy_indices[0]
            rem_bond_indices = all_bond_indices[remote_index]
            rem_bond_indices = rem_bond_indices[rem_bond_indices != -1]
            rem_heavy_mask = elements[rem_bond_indices] != "H"
            rem_heavy_indices = rem_bond_indices[rem_heavy_mask]
            # Use the coord of any heavy atom bonded to the remote
            # atom
            rem_rem_index = rem_heavy_indices[0]
            # Include the directly bonded atom two times, to give it a
            # greater weight in superimposition
            heavy_coord = coords[:, [remote_index, remote_index, rem_rem_index]]
            stereo = torch.zeros((coords.shape[0]))
        elif n_heavy_bonds == 2:
            heavy_coord = coords[:, [heavy_indices[0], heavy_indices[1], i]]
            stereo = torch.zeros((coords.shape[0]))
        elif n_heavy_bonds == 3:
            heavy_coord = coords[:, heavy_indices]
            center = coords[:, i]
            # Determine the enantiomer of this stereocenter
            # For performance reasons, the result does not follow the
            # R/S nomenclature, but a custom -1/1 based one, which also
            # unambiguously identifies the enantiomer
            n = torch.cross(heavy_coord[:, 0] - center, heavy_coord[:, 1] - center)
            sign = torch.sign(((heavy_coord[:, 2] - center) * n).sum(dim=-1))
            stereo = sign.detach().cpu()
        elif n_heavy_bonds == 4:
            # The fragment is irrelevant, as there is no bonded hydrogen
            # -> The fragment coord is the coord of the central atom
            # 4 times repeated
            heavy_coord = torch.repeat(coords[:, None, i, :], 3, axis=0)
            stereo = torch.zeros((coords.shape[0]))
        else:
            warnings.warn(
                f"Atom '{names[i]}' in "
                f"'{res_name[i]}' has more than 4 bonds to "
                f"heavy atoms ({n_heavy_bonds}) and is ignored"
            )
            heavy_coord = torch.repeat(coords[:, None, i, :], 3, axis=0)
            hydrogen_coord = torch.zeros((0, 3), dtype=torch.float32, device=device)
            stereo = torch.zeros((coords.shape[0]))
        central_coord = coords[:, i]
        fragments[i] = (
            elements[i],
            # charges[i],
            stereo,
            heavy_types,
            central_coord,
            heavy_coord,
            hydrogen_coord,
        )
    return fragments


# hydrogen_coords batch size is 0 and shped like naming sample
# only for evaluation gradients don't pass
def update_hydrogens_in_atom_array(atoms, hydrogen_coord, hydrogen_names):
    count = 0
    for coord in hydrogen_coord:
        count += len(coord)
        
    hydrogenated_atoms = struc.AtomArray(atoms.array_length() + count)
    original_atom_mask = np.zeros(hydrogenated_atoms.array_length(), dtype=bool)
    
    # Add all annotation categories of the original AtomArray
    for category in atoms.get_annotation_categories():
        if category not in hydrogenated_atoms.get_annotation_categories():
            original_annotation = atoms.get_annotation(category)            
            new_shape = (hydrogenated_atoms.array_length(),) + original_annotation.shape[1:]
            hydrogenated_atoms.add_annotation(category, dtype=original_annotation.dtype)
            hydrogenated_atoms.set_annotation(category, np.zeros(new_shape, dtype=original_annotation.dtype))


    # Fill the combined AtomArray residue for residue
    # Stores covalent bonds between a heavy atom and its hydrogen atoms
    hydrogen_bonds = []
    residue_starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    index_mapping = np.zeros(atoms.array_length(), dtype=np.uint32)
    p = 0
    i_name = 0
    for i in range(len(residue_starts) - 1):
        # Set annotation and coordinates from input AtomArray
        start = residue_starts[i]
        stop = residue_starts[i + 1]
        res_length = stop - start
        index_mapping[start:stop] = np.arange(p, p + res_length)
        original_atom_mask[p : p + res_length] = True
        hydrogenated_atoms.coord[p : p + res_length] = atoms.coord[:, start:stop]
        for category in atoms.get_annotation_categories():
            hydrogenated_atoms.get_annotation(category)[p : p + res_length] = (
                atoms.get_annotation(category)[start:stop]
            )
        p += res_length
        # Set annotation and coordinates for hydrogen atoms
        for j in range(start, stop):
            hydrogen_coord_for_atom = hydrogen_coord[j]
            for coord in hydrogen_coord_for_atom:
                hydrogenated_atoms.coord[p] = coord.detach().cpu().numpy()
                hydrogenated_atoms.chain_id[p] = atoms.chain_id[j]
                hydrogenated_atoms.res_id[p] = atoms.res_id[j]
                hydrogenated_atoms.ins_code[p] = atoms.ins_code[j]
                hydrogenated_atoms.res_name[p] = atoms.res_name[j]
                hydrogenated_atoms.hetero[p] = atoms.hetero[j]
                hydrogenated_atoms.atom_name[p] = hydrogen_names[i_name][-1]
                hydrogenated_atoms.element[p] = "H"
                heavy_index = index_mapping[j]
                hydrogen_index = p
                hydrogen_bonds.append((heavy_index, hydrogen_index))
                p += 1
                i_name += 1
                
    # Add bonds to combined AtomArray
    original_bonds = atoms.bonds.as_array()
    bond_indices = index_mapping[original_bonds[:, :2]]
    heavy_bonds = np.stack(
        [
            bond_indices[:, 0],
            bond_indices[:, 1],
            # The bond types
            original_bonds[:, 2],
        ],
        axis=-1,
    )
    hydrogen_bonds = np.array(hydrogen_bonds, dtype=np.uint32).reshape(-1, 2)
    # All bonds to hydrogen atoms are single bonds
    hydrogen_bonds = np.stack(
        [
            hydrogen_bonds[:, 0],
            hydrogen_bonds[:, 1],
            np.ones(len(hydrogen_bonds), dtype=np.uint32),
        ],
        axis=-1,
    )
    hydrogenated_atoms.bonds = struc.BondList(
        hydrogenated_atoms.array_length(), np.concatenate([heavy_bonds, hydrogen_bonds])
    )
    return hydrogenated_atoms, original_atom_mask


def batch_hydrogen_to_atom_stack_array(hydrogen_atoms, hydrogen_coords, mask_non_hydrogen_atoms):
    atom_stack = struc.AtomArrayStack(hydrogen_coords.shape[0], hydrogen_atoms.shape[0])

    # Copy all attributes from atom_array to atom_stack
    for category in hydrogen_atoms.get_annotation_categories():
        atom_stack.add_annotation(category, dtype=hydrogen_atoms.get_annotation(category).dtype)
        atom_stack.set_annotation(category, hydrogen_atoms.get_annotation(category))

    batched_atom_coords = np.zeros((hydrogen_coords.shape[0], hydrogen_atoms.shape[0], 3), dtype=hydrogen_atoms.coord.dtype)
    batched_atom_coords[:, mask_non_hydrogen_atoms] = hydrogen_atoms.coord[mask_non_hydrogen_atoms][None].repeat(6, axis=0)
    batched_atom_coords[:, ~mask_non_hydrogen_atoms] = hydrogen_coords.detach().cpu().numpy()
    atom_stack.coord = batched_atom_coords
    
    return atom_stack

def save_hydrogen_array(hydrogen_array, path="test.pdb"):
    hydrogen_array.chain_id = [chain[:1] for chain in hydrogen_array.chain_id]
    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(hydrogen_array) 
    pdb_file.write(path)
    
def get_hydrogen_names(atom_array, hydrogen_coords, name_library):
        hydrogen_names = []
        residue_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        for i in range(len(residue_starts) - 1):
            start = residue_starts[i]
            stop = residue_starts[i + 1]
            already_used_names = set()
            for j in range(start, stop):
                hydrogen_coord_for_atom = hydrogen_coords[j]
                hydrogen_name_generator = name_library.generate_hydrogen_names(
                    atom_array.res_name[j], atom_array.atom_name[j]
                )
                for coord in hydrogen_coord_for_atom:
                    for hydrogen_name in hydrogen_name_generator:
                        if hydrogen_name == "" or hydrogen_name not in already_used_names:
                            already_used_names.add(hydrogen_name)
                            break
                    hydrogen_names += [(atom_array.res_id[j],atom_array.res_name[j],hydrogen_name)]
        return hydrogen_names
    
def add_hydrogen_to_pdb(in_path, out_path=None):
    if out_path is None:
        out_path = in_path
    pdb_file = pdb.PDBFile.read(in_path)
    atom_array = pdb.get_structure(pdb_file, include_bonds=True)
    
    bonds = atom_array.bonds
    names = atom_array.atom_name
    elements = atom_array.element
    res_name = atom_array.res_name
    coords = torch.tensor(atom_array.coord)
    fragment_library = FragmentLibrary.standard_library()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hydrogen_atoms_batch, naming_sample = fragment_library.calculate_hydrogen_coord_batch(coords, bonds, names, elements, res_name, device)
    
    name_library = AtomNameLibrary.standard_library()
    hydrogen_names = get_hydrogen_names(atom_array, naming_sample, name_library)
    hydrogen_array,_ = update_hydrogens_in_atom_array(atom_array, naming_sample, hydrogen_names)
    save_hydrogen_array(hydrogen_array, out_path)
