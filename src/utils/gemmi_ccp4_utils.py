import gemmi
import numpy as np
from typing import Tuple
from tqdm import tqdm
from scipy.spatial import cKDTree
import torch

def split_into_chunks(arr, chunk_size=1000):
    return [arr[i:i+chunk_size] for i in range(0, len(arr), chunk_size)]

def hkl_to_density(miller_array, amplitudes, phases, space_group, density_grid_size, volume):
    fourier_grid = np.zeros(density_grid_size, dtype=np.complex128)
    for operation in space_group.operations().sym_ops:
        new_hkl = (((np.array(operation.rot) / operation.DEN)[None] @ miller_array[...,None]).squeeze(-1)).astype(np.int32)
        phase_shift = (-2 * np.pi / operation.DEN) * ((miller_array * np.array(operation.tran)[None]).sum(axis=-1))
        theta = phase_shift + phases

        fourier_grid[tuple(new_hkl.T)] = (amplitudes * (np.cos(theta) + 1j * np.sin(theta)))

    friedel_indexes = -np.mgrid[:fourier_grid.shape[0], :fourier_grid.shape[1], :fourier_grid.shape[2]]
    friedel_mates = fourier_grid[tuple(friedel_indexes)].conj()
    friedel_mask = fourier_grid == 0
    fourier_grid[friedel_mask] = friedel_mates[friedel_mask]
    fourier_grid = fourier_grid.real - fourier_grid.imag * 1j # this is equal to flipping axeses

    normalization_factor = 1 / volume
    return (np.fft.ifftn(fourier_grid, norm="forward").real * normalization_factor)

def mtz_to_density_ccp4_map(mtz_path, sample_rate=3.0):
    mtz = gemmi.read_mtz_file(mtz_path)
    density_grid_size = mtz.get_size_for_hkl(sample_rate=sample_rate)

    density_grid = gemmi.FloatGrid()
    density_grid.unit_cell = mtz.cell
    density_grid.spacegroup = mtz.spacegroup
    density_grid.set_size(*density_grid_size)

    miller_array = mtz.make_miller_array()
    amplitudes = mtz.column_with_label("FWT").array
    phases = (mtz.column_with_label("PHWT").array / 360) * 2 * np.pi

    density_grid.array[:] = hkl_to_density(miller_array, amplitudes, phases, mtz.spacegroup, density_grid.array.shape, density_grid.unit_cell.volume)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = density_grid
    ccp4.update_ccp4_header()
    return ccp4

def calcualte_min_distance_between_cloud_points(cloud_a, cloud_b, batch_size=1000000):
    """
        for each value in cloud a, get the min distance in cloud_b
    """
    cloud_a_shape = cloud_a.shape 
    cloud_a = cloud_a.reshape(-1,3)
    cloud_b = cloud_b.reshape(-1,3)

    kd_tree = cKDTree(cloud_b)
    cloud_a_chunks = split_into_chunks(cloud_a, batch_size)
    min_distances = []
    for chunk in tqdm(cloud_a_chunks, "calcualting distances to region of interest"):
        min_distances.append(kd_tree.query(chunk, k=1, workers=-1)[0])
    min_distances = np.concatenate(min_distances, axis=0).reshape(cloud_a_shape[:-1])
    return min_distances

def apply_translation(m: gemmi.Ccp4Map, di: int, dj: int, dk: int) -> gemmi.Ccp4Map:
    m = copy_ccp4map(m)
    m.setup(0)
    # Read the old starts
    old_i = m.header_i32(5)
    old_j = m.header_i32(6)
    old_k = m.header_i32(7)

    # Write the new starts
    m.set_header_i32(5, old_i + di)
    m.set_header_i32(6, old_j + dj)
    m.set_header_i32(7, old_k + dk)

    return m

def slice_density(m: gemmi.Ccp4Map, range_x: Tuple[int, int], range_y: Tuple[int, int], range_z: Tuple[int, int]) -> gemmi.Ccp4Map:
    m = copy_ccp4map(m)
    m.setup(0)
    old_grid = m.grid
    old_cell = old_grid.unit_cell
    old_nx, old_ny, old_nz = old_grid.nu, old_grid.nv, old_grid.nw

    # Read the original start-indices (NCSTART, NRSTART, NSSTART)
    orig_i = m.header_i32(5)
    orig_j = m.header_i32(6)
    orig_k = m.header_i32(7)

    # 2. Define your slice in voxel indices
    i0, i1 = range_x   # I‐range
    j0, j1 = range_y  # J‐range
    k0, k1 = range_z    # K‐range

    # 3. Pull out the block as a NumPy array
    arr = np.array(old_grid, copy=False)       # shape = (nz, ny, nx)
    arr[np.isnan(arr)] = 0
    sub_arr = arr[k0:k1, j0:j1, i0:i1]          # shape = (new_nz, new_ny, new_nx)
    new_nx, new_ny, new_nz = sub_arr.shape

    # 4. Compute the new unit-cell lengths (angles stay the same)
    a_new = old_cell.a * (new_nx / old_nx)
    b_new = old_cell.b * (new_ny / old_ny)
    c_new = old_cell.c * (new_nz / old_nz)
    alpha, beta, gamma = old_cell.alpha, old_cell.beta, old_cell.gamma
    new_cell = gemmi.UnitCell(a_new, b_new, c_new, alpha, beta, gamma)

    # 5. Build a new FloatGrid from your sliced data
    new_grid = gemmi.FloatGrid(
        sub_arr.astype(np.float32),
        cell=new_cell,
        spacegroup=gemmi.SpaceGroup("P1")
    )

    # 6. Pack into a fresh Ccp4Map and regenerate the header
    out = gemmi.Ccp4Map()
    out.grid = new_grid
    out.update_ccp4_header(2)

    # 7. Now bump the start‐words so voxel (0,0,0) is placed correctly
    out.set_header_i32(5, orig_i + k0)  # NCSTART
    out.set_header_i32(6, orig_j + j0)  # NRSTART
    out.set_header_i32(7, orig_k + i0)  # NSSTART

    return out

def expand_to_p1(m: gemmi.Ccp4Map) -> gemmi.Ccp4Map:
    m = copy_ccp4map(m)
    m.setup(0)
    grid = m.grid   # gemmi.FloatGrid
    grid.symmetrize_max()
    grid.spacegroup = gemmi.SpaceGroup("P1")
    arr = np.array(grid, copy=False)       # shape = (nz, ny, nx)
    arr[np.isnan(arr)] = 0
    # 4. Write out the extended map
    out = gemmi.Ccp4Map()
    out.grid = grid
    out.update_ccp4_header(2)   
    return out  

def expand(m: gemmi.Ccp4Map) -> gemmi.Ccp4Map:
    """
    Tile a full P1 unit cell m.grid by translations only, NX×NY×NZ times,
    but place the original cell in the center of the supercell.
    """

    # 1. Ensure metadata is populated
    m = copy_ccp4map(m)
    m.setup(0)
    grid = m.grid
    cell = grid.unit_cell
    old_nx, old_ny, old_nz = grid.nu, grid.nv, grid.nw

    # original CCP4 start indices
    orig_i = m.header_i32(5)
    orig_j = m.header_i32(6)
    orig_k = m.header_i32(7)

    # 2. Slice out the raw data and tile
    arr = np.array(grid, copy=False)  # shape = (nz, ny, nx)
    tiled = np.tile(arr, (3, 3, 3))
    new_nx, new_ny, new_nz = tiled.shape

    # 3. Build the new cell: lengths scaled, angles unchanged
    new_cell = gemmi.UnitCell(cell.a * 3,
                              cell.b * 3,
                              cell.c * 3,
                              cell.alpha, cell.beta, cell.gamma)

    # 4. Create the new FloatGrid with explicit dims
    new_grid = gemmi.FloatGrid(new_nx, new_ny, new_nz)
    new_grid.unit_cell = new_cell
    new_grid.spacegroup = gemmi.find_spacegroup_by_number(1)  # now P1
    new_grid.array[:] = tiled

    # 5. Pack into a new Ccp4Map and regenerate header
    out = gemmi.Ccp4Map()
    out.grid = new_grid
    out.update_ccp4_header(2)

    # 7. Shift the start indices so that the “center” copy
    #    corresponds to the original spatial location
    out.set_header_i32(5, orig_i - old_nx)
    out.set_header_i32(6, orig_j - old_ny)
    out.set_header_i32(7, orig_k - old_nz)

    return out

def copy_ccp4map(m: gemmi.Ccp4Map) -> gemmi.Ccp4Map:
    m.setup(0)
    grid_clone = m.grid.clone()
    clone = gemmi.Ccp4Map()
    clone.grid = grid_clone
    clone.update_ccp4_header(2)
    for w in range(1, 257):
        try:
            value = m.header_i32(w)
            clone.set_header_i32(w, value)
        except RuntimeError:
            # some words aren’t valid ints—skip those
            pass
    return clone


# def get_density_voxel_center_locations(density_map: gemmi.Ccp4Map):
#     g = density_map.grid
#     nx, ny, nz = g.nu, g.nv, g.nw

#     # Get voxel-space offset (header words 5, 6, 7)
#     # These are the starting indices of the grid along each axis
#     orig_u = density_map.header_i32(5)
#     orig_v = density_map.header_i32(6)
#     orig_w = density_map.header_i32(7)
#     density_map.setup(0)

#     # Grid indices
#     us = np.arange(nx)
#     vs = np.arange(ny)
#     ws = np.arange(nz)
#     uu, vv, ww = np.meshgrid(us, vs, ws, indexing='ij')

#     # Convert to fractional coordinates using offset and center shift
#     frac_u = (uu + orig_u + 0.5) / g.nu
#     frac_v = (vv + orig_v + 0.5) / g.nv
#     frac_w = (ww + orig_w + 0.5) / g.nw
#     frac = np.stack([frac_u, frac_v, frac_w], axis=-1)  # shape: (nx, ny, nz, 3)

#     # Convert fractional to orthogonal using unit cell matrix
#     uc_mat = np.array(g.unit_cell.orthogonalization_matrix)
#     positions = frac @ uc_mat.T  # shape: (nx, ny, nz, 3)

#     return positions

def get_density_voxel_center_locations(density_map: gemmi.Ccp4Map):
    density_extent = density_map.get_extent()
    extent_minimum, extent_maximum = [np.array(list(density_extent.minimum)), np.array(list(density_extent.maximum))]
    extent_size = extent_maximum - extent_minimum
    # density locations are between 0 and 1
    density_locations = np.mgrid[:density_map.grid.shape[0], :density_map.grid.shape[1], :density_map.grid.shape[2]].transpose(1,2,3,0) / np.array(density_map.grid.shape)[None, None, None]
    # density locations are between minimum extent and maximum extent
    density_locations = (density_locations * extent_size[None,None,None]) + extent_minimum[None,None,None]
    voxel_size = extent_size / np.array(density_map.grid.shape)
    # moving the density locations to the center of the voxel and not the bottom left
    density_locations = density_locations + voxel_size[None, None, None] / 2
    # projects the fractional coordinates to absoulte coordiantes
    density_locations = (np.array(density_map.grid.unit_cell.orth.mat)[None, None, None] @ density_locations[..., None]).squeeze(-1)
    density_locations = torch.tensor(density_locations, device="cpu", dtype=torch.float32)
    density_locations = density_locations.reshape(-1,3)
    return density_locations
    

class Bounds:
    def __init__(self, bounds_x, bounds_y, bounds_z):
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.bounds_z = bounds_z
    
    def __contains__(self, other):
        """
            check if the bounds of other are in the bounds of self
        """
        in_x = (other.bounds_x[0] > self.bounds_x[0]) and (other.bounds_x[1] < self.bounds_x[1])
        in_y = (other.bounds_y[0] > self.bounds_y[0]) and (other.bounds_y[1] < self.bounds_y[1])
        in_z = (other.bounds_z[0] > self.bounds_z[0]) and (other.bounds_z[1] < self.bounds_z[1])
        return in_x and in_y and in_z
    
    def __str__(self):
        return f"{self.bounds_x}, {self.bounds_y}, {self.bounds_z}"

def get_density_bounds(density: gemmi.Ccp4Map):
    unit_cell_sizes = np.array(density.grid.unit_cell.parameters[:3])
    dx, dy, dz = unit_cell_sizes / np.array(density.grid.shape)
    offset_x = density.header_i32(5) * dx
    offset_y = density.header_i32(6) * dy
    offset_z = density.header_i32(7) * dz
    size_x = (offset_x, offset_x + unit_cell_sizes[0])
    size_y = (offset_y, offset_y + unit_cell_sizes[1])
    size_z = (offset_z, offset_z + unit_cell_sizes[2])
    return Bounds(size_x, size_y, size_z)

def wrap_density_around(atom_locations: np.ndarray, density_object: gemmi.Ccp4Map, padding=5) -> gemmi.Ccp4Map:
    """
        this function will return a new density that is wrapped around the atom locations
    """
    min_atom_locations = atom_locations.min(axis=0) - padding
    max_atom_locations = atom_locations.max(axis=0) + padding
    atom_locations_bounds = Bounds(*[(min_atom_locations[i], max_atom_locations[i]) for i in range(3)])
    density_object = expand_to_p1(density_object)
    while not atom_locations_bounds in get_density_bounds(density_object):
        density_object = expand(density_object)
    density_centeroids = get_density_voxel_center_locations(density_object).reshape(list(density_object.grid.shape) + [3])
    min_distances = calcualte_min_distance_between_cloud_points(density_centeroids, atom_locations)
    mask = min_distances < padding

    nonzero = mask.nonzero()
    ranges = [(nonzero[i].min(), nonzero[i].max()) for i in range(3)]
    sliced_density = slice_density(density_object, ranges[2], ranges[1], ranges[0])
    return sliced_density


if __name__ == "__main__":
    density = gemmi.read_ccp4_map("1lu4_box_xyz.ccp4")
    pdb = gemmi.read_pdb("pipline_inputs/pdbs/1lu4/1lu4_chain_A_altloc_A_fixed.pdb")
    atom_locations = np.array([list(atom.pos) for res in pdb[0][0] for atom in res])
    wrapped_density = wrap_density_around(atom_locations, density)
    wrapped_density.write_ccp4_map("wrapped_density.ccp4")
    a = 2