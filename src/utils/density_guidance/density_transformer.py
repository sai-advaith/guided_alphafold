import numpy as np
import torch

from .density_atomsf import ATOM_STRUCTURE_FACTORS, ELECTRON_SCATTERING_FACTORS, ALL_ATOM_TENSOR, ATOM_TO_INDEX

def to_tensor(data, dtype=torch.float32, device='cuda', requires_grad=True):
    if isinstance(data, (np.ndarray, list, tuple)):
        tensor = torch.tensor(data, dtype=dtype, device=device)
    elif isinstance(data, torch.Tensor):
        tensor = data.to(device=device, dtype=dtype).clone()
        if tensor.requires_grad != requires_grad:
            tensor = tensor.detach().requires_grad_(requires_grad)
    else:
        tensor = torch.tensor(data, dtype=dtype, device=device)
    return tensor

class ElectronDensityCalculator:
    def __init__(
        self,
        structure_coordinates,
        structure_q,
        structure_e,
        structure_b,
        xmap,
        smin=None,
        smax=None,
        rmax=3.0,
        rstep=0.01,
        simple=False,
        em=False,
        dtype=torch.float32,
        device=torch.device("cuda:0")
    ):
        self.dtype = dtype
        self.device = device
        self.pi = torch.tensor(np.pi, device=self.device)
        self.four_pi2 = 4 * self.pi * self.pi

        # Structure
        self.structure_coor = structure_coordinates
        self.structure_b = structure_b
        self.structure_active = torch.ones(self.structure_coor.shape[0], dtype=torch.long, device=self.device)
        self.structure_q = structure_q
        self.structure_e = structure_e

        self.xmap = xmap
        self.smin = smin
        self.smax = smax
        self.rmax = rmax
        self.rstep = rstep
        self.simple = simple
        self.em = em
        self.asf_range = 6
        if self.em == True:
            self.asf_range = 5
            self._asf = ELECTRON_SCATTERING_FACTORS
        else:
            # self._asf = ATOM_STRUCTURE_FACTORS
            self._asf = ALL_ATOM_TENSOR.to(dtype=self.dtype, device=self.device)
            
            self.atom_to_index = ATOM_TO_INDEX
            self.atom_to_index_map = np.vectorize(lambda x: self.atom_to_index.get(x, self.atom_to_index['C']))

        self._initialized = False

        if not simple and smax is None and self.xmap.resolution.high is not None:
            self.smax = 1 / (2 * self.xmap.resolution.high)
        if not simple:
            rlow = self.xmap.resolution.low
            if rlow is None:
                rlow = 1000
            self.smin = 1 / (2 * rlow)

        # Calculate transforms
        uc = xmap.unit_cell
        self.lattice_to_cartesian = uc.frac_to_orth / uc.abc
        self.cartesian_to_lattice = uc.orth_to_frac * uc.abc.reshape(3, 1)
        self.grid_to_cartesian = self.lattice_to_cartesian * self.xmap.voxelspacing
        self._grid_coor = torch.zeros_like(self.structure_coor)
        self._grid_coor_rot = torch.zeros_like(self.structure_coor)

    def _coor_to_grid_coor(self):
        coor = self.structure_coor + 0.0
        if not torch.allclose(self.xmap.origin, torch.zeros_like(self.xmap.origin)):
            coor = coor - self.xmap.origin

        self._grid_coor = torch.matmul(coor, self.cartesian_to_lattice.T)
        self._grid_coor = self._grid_coor / self.xmap.voxelspacing
        self._grid_coor = self._grid_coor - self.xmap.offset

    def reset(self, rmax=None, full=False):
        if full:
            self.xmap.array.fill_(0.0)
            self._grid_coor = torch.zeros_like(self.structure_coor)
        else:
            self.mask(rmax=rmax, value=0.0)

    def compute_density(self, distance_squared, bfactors, elements):
        """
        Use structure factors and densities to compute 
        """
        indices = self.atom_to_index_map(elements)
        asf_data = self._asf[indices]

        bfactors = bfactors.expand(-1, self.asf_range)
        divisor = asf_data[:, 1, :] + bfactors

        bw = torch.where(divisor > 1e-4, -self.four_pi2 / divisor, 0.0)
        aw = asf_data[:, 0, :] * (-bw / self.pi) ** 1.5

        exp_factor = distance_squared.unsqueeze(-1) * bw.unsqueeze(1)
        density = torch.sum(aw.unsqueeze(1) * torch.exp(exp_factor), dim=-1)
        return density

    def dilate_array(self, points, active, q, lmax, rmax, grid_to_cartesian, out):
        import time
        rmax2 = rmax * rmax
        out_shape = out.shape
        grid_to_cartesian_t = grid_to_cartesian.t()

        out_slice = out_shape[2] * out_shape[1]
        out_size = out_slice * out_shape[0]

        # Pre-compute ranges for each dimension
        amin, bmin, cmin = torch.ceil(points - lmax).long().unbind(-1)
        amax, bmax, cmax = torch.floor(points + lmax).long().unbind(-1)

        # Determine the range for the voxel grid and determine rows with same shape
        diffs = torch.stack((amax-amin+1, bmax-bmin+1, cmax-cmin+1)).t()
        unique_idx, inverse_indices = torch.unique(diffs.cpu(), dim=0, return_inverse=True)
        inverse_indices = inverse_indices.numpy()

        # Create batch over meshes with the same shape
        for i, key in enumerate(unique_idx):
            indices = np.where(inverse_indices == i)[0]
            indices = np.atleast_1d(indices)

            # Create batch of points with same shape
            batch_points = points[indices]
            batch_q = q[indices]
            batch_elements = np.atleast_1d(self.structure_e[indices])
            batch_elements_indices = torch.tensor(self.atom_to_index_map(batch_elements), device=self.device)
            batch_bfactors = self.structure_b[indices].view(-1, 1)

             # Vectorized range creation
            a_ranges = torch.arange(amin[indices].min(), amax[indices].max() + 1, device=self.device)
            b_ranges = torch.arange(bmin[indices].min(), bmax[indices].max() + 1, device=self.device)
            c_ranges = torch.arange(cmin[indices].min(), cmax[indices].max() + 1, device=self.device)

            # Create expanded ranges for each index
            a_expanded = a_ranges.unsqueeze(0).expand(len(indices), -1)
            b_expanded = b_ranges.unsqueeze(0).expand(len(indices), -1)
            c_expanded = c_ranges.unsqueeze(0).expand(len(indices), -1)

            # Create masks for valid ranges
            a_mask = (a_expanded >= amin[indices].unsqueeze(1)) & (a_expanded <= amax[indices].unsqueeze(1))
            b_mask = (b_expanded >= bmin[indices].unsqueeze(1)) & (b_expanded <= bmax[indices].unsqueeze(1))
            c_mask = (c_expanded >= cmin[indices].unsqueeze(1)) & (c_expanded <= cmax[indices].unsqueeze(1))

            # Apply masks to get valid ranges for each index (vectorized)
            a_ranges = a_expanded[a_mask].split(a_mask.sum(1).tolist())
            b_ranges = b_expanded[b_mask].split(b_mask.sum(1).tolist())
            c_ranges = c_expanded[c_mask].split(c_mask.sum(1).tolist())

            # import ipdb; ipdb.set_trace()

            # Stack up to get all coordinates every mesh
            cc, bb, aa = zip(*[torch.meshgrid(c, b, a, indexing='ij') for c, b, a in zip(c_ranges, b_ranges, a_ranges)])
            cc, bb, aa = torch.stack(cc), torch.stack(bb), torch.stack(aa)
            all_meshes = torch.stack((aa, bb, cc), dim=-1).view(len(indices), -1, 3)

            # Compute the values in cartesian space TODO vmap 159 - 167
            diff = all_meshes - batch_points.unsqueeze(1)
            diff_cartesian = torch.bmm(diff, grid_to_cartesian_t.unsqueeze(0).repeat((len(indices), 1, 1)))

            # Distance squared
            d2_zyx_flat = torch.sum(diff_cartesian**2, dim=-1)

            # Compute density and determine which are a certain radius apart
            density_values = self.compute_density(d2_zyx_flat, batch_bfactors, batch_elements)

            mask = d2_zyx_flat <= rmax2

            # Determine index in density grid
            ind_c = (cc * out_slice) % out_size
            ind_cb = ((bb * out_shape[2]) % out_slice) + ind_c
            final_indices = ind_cb + (aa % out_shape[2])

            # Populate the grid
            radial_values = (density_values * batch_q.unsqueeze(1))[mask]
            scatter_indices = final_indices[mask.view(mask.shape[0], *cc.shape[1:])]
            # out = out.view(-1).scatter_add(0, scatter_indices, radial_values).view(out_shape)
            out.view(-1).scatter_add_(0, scatter_indices, radial_values)
        return out

    def density(self):
        self._coor_to_grid_coor()
        lmax = self.rmax / self.xmap.voxelspacing
        active, q = self.structure_active, self.structure_q
        for symop in self.xmap.unit_cell.space_group.symop_list:
            R_torch = to_tensor(symop.R, dtype=self.dtype, device=self.device, requires_grad=False)
            t_torch = to_tensor(symop.t, dtype=self.dtype, device=self.device, requires_grad=False)

            self._grid_coor_rot = torch.matmul(self._grid_coor, R_torch.T)
            self._grid_coor_rot = self._grid_coor_rot + t_torch * torch.tensor(self.xmap.shape[::-1],
                                                                               dtype=self.dtype,
                                                                               device=self.device)
            self.xmap.array = self.dilate_array(self._grid_coor_rot, active,
                                                q, lmax, self.rmax,
                                                self.grid_to_cartesian,
                                                self.xmap.array) + 0.0
