import gemmi
import numpy as np 
import argparse
import torch
import math

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
    return np.fft.ifftn(fourier_grid, norm="forward").real * normalization_factor


def hkl_to_density_torch_original(miller_array, amplitudes, phases, space_group, density_grid_size, volume):
    device = miller_array.device
    dtype = torch.complex64

    fourier_grid = torch.zeros(density_grid_size, dtype=dtype, device=device)

    for operation in space_group.operations().sym_ops:
        rot = torch.tensor(operation.rot, dtype=torch.float32, device=device) / operation.DEN
        tran = torch.tensor(operation.tran, dtype=torch.float32, device=device)

        new_hkl = torch.round((miller_array @ rot.T)).to(torch.long)

        phase_shift = (-2 * math.pi / operation.DEN) * (miller_array * tran).sum(dim=-1)
        theta = phase_shift + phases
        values = amplitudes * torch.exp(1j * theta)

        # Assign to grid (note: assumes no collisions)
        idx = tuple(new_hkl.T)
        fourier_grid[idx] = values

    # Fill Friedel mates
    grids = [torch.arange(s, device=device) for s in density_grid_size]
    mesh = torch.meshgrid(*grids, indexing='ij')
    friedel_idx = [(-g % s).long() for g, s in zip(mesh, density_grid_size)]

    friedel_mates = fourier_grid[friedel_idx].conj()
    mask = (fourier_grid == 0)
    fourier_grid = torch.where(mask, friedel_mates, fourier_grid)

    # Hermitian flip (ensures real result)
    fourier_grid = fourier_grid.real - 1j * fourier_grid.imag

    # Inverse FFT
    density = torch.fft.ifftn(fourier_grid, norm="forward").real
    return density * (1 / volume)

def hkl_to_density_torch(miller_array, amplitudes, phases, space_group, density_grid_size, volume):
    """
    miller_array: [N, 3], torch.float32
    amplitudes: [B, N], torch.float32
    phases: [B, N], torch.float32
    Returns: [B, D, H, W] real-valued densities
    """
    device = miller_array.device
    B, N = amplitudes.shape
    D, H, W = density_grid_size
    dtype = torch.complex64

    fourier_grid = torch.zeros((B, D, H, W), dtype=dtype, device=device)
    count_grid = torch.zeros((B, D, H, W), dtype=torch.float32, device=device)

    for operation in space_group.operations().sym_ops:
        rot = torch.tensor(operation.rot, dtype=torch.float32, device=device) / operation.DEN
        tran = torch.tensor(operation.tran, dtype=torch.float32, device=device)

        new_hkl = torch.round((miller_array @ rot.T)).to(torch.long)  # [N, 3]
        new_hkl = new_hkl % torch.tensor([D, H, W], device=device)  # wrap negative indices

        phase_shift = (-2 * math.pi / operation.DEN) * (miller_array * tran).sum(dim=-1)  # [N]
        theta = phase_shift[None, :] + phases  # [B, N]
        values = amplitudes * torch.exp(1j * theta)  # [B, N]

        # Flatten grid for scatter
        flat_idx = (new_hkl[:, 0] * H * W + new_hkl[:, 1] * W + new_hkl[:, 2])  # [N]
        flat_idx = flat_idx[None, :].expand(B, N)  # [B, N]
        values_flat = values  # [B, N]

        grid_flat = fourier_grid.view(B, -1)
        count_flat = count_grid.view(B, -1)

        grid_flat.scatter_add_(dim=1, index=flat_idx, src=values_flat)
        count_flat.scatter_add_(dim=1, index=flat_idx, src=torch.ones_like(values_flat.real))

    # Avoid divide-by-zero
    count_grid = count_grid.clamp(min=1.0)
    fourier_grid = fourier_grid / count_grid

    # Fill Friedel mates
    coords = torch.meshgrid(
        [torch.arange(s, device=device) for s in (D, H, W)],
        indexing='ij'
    )
    friedel_idx = [(-g % s).long() for g, s in zip(coords, (D, H, W))]
    friedel_mates = fourier_grid[:, friedel_idx[0], friedel_idx[1], friedel_idx[2]].conj()
    mask = (fourier_grid == 0)
    fourier_grid = torch.where(mask, friedel_mates, fourier_grid)

    # Hermitian flip
    fourier_grid = fourier_grid.real - 1j * fourier_grid.imag

    # Inverse FFT
    density = torch.fft.ifftn(fourier_grid, dim=(-3, -2, -1), norm="forward").real
    return density * (1 / volume)

def save_density_ccp4(mtz_path, density, output_ccp4_path):
    mtz = gemmi.read_mtz_file(mtz_path)
    density_grid_size = density.shape

    density_grid = gemmi.FloatGrid()
    density_grid.unit_cell = mtz.cell
    density_grid.spacegroup = mtz.spacegroup
    density_grid.set_size(*density_grid_size)

    density_grid.array[:] = density
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = density_grid
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(output_ccp4_path)

def mtz_to_density_ccp4_map(mtz_path):
    mtz = gemmi.read_mtz_file(mtz_path)
    density_grid_size = mtz.get_size_for_hkl(sample_rate=3.0) # TODO: make this a hyper parameter ? 

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

def main(mtz_path, output_file='density.ccp4'):
    ccp4 = mtz_to_density_ccp4_map(mtz_path)
    ccp4.write_ccp4_map(output_file)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mtz_path",type=str,default="/nfs/scistore20/bronsgrp/nsellam/proteinx_guidance/pipeline_inputs/mtzs/1lu4/1lu4.mtz" , required=False, help="the path to the mtz file you want to convert to ccp4")
    argparser.add_argument("--output_file",type=str, default="converted.ccp4", required=False, help="the output ccp4 file from mtz")
    args = argparser.parse_args()
    main(args.mtz_path, args.output_file)