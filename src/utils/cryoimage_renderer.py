from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from cryoforward.atom_stack import AtomStack
from cryoforward.cryoesp_calculator import setup_fast_esp_solver
import math
from cryoforward.ctf import CTFParams, compute_ctf_2d
from cryoforward.lattice import Lattice


def projection_shape(grid_dims: tuple[int, int, int], axis: int) -> tuple[int, int]:
    if axis == 0:
        return (grid_dims[1], grid_dims[2])
    if axis == 1:
        return (grid_dims[0], grid_dims[2])
    return (grid_dims[0], grid_dims[1])


def prepare_lattice_from_density_map(
    *,
    density_map_path: str | Path,
    sublattice_radius: float,
    projection_axis: int,
    collapse_projection_axis: bool,
    device: torch.device | str,
    grid_size_override: int | None = None,
    pixel_size_override: float | None = None,
) -> tuple[Lattice, dict[str, Any]]:
    density_map = gemmi.read_ccp4_map(str(density_map_path))
    D_map = density_map.grid.nu
    maxsize_map = density_map.grid.unit_cell.a
    pixel_size_map = maxsize_map / D_map
    left_bottom_map = list(np.array(list(density_map.get_extent().minimum)) * maxsize_map)
    right_upper_map = list(np.array(list(density_map.get_extent().maximum)) * maxsize_map)

    D = grid_size_override if grid_size_override is not None else D_map
    pixel_size = pixel_size_override if pixel_size_override is not None else pixel_size_map

    # Recompute extent around the original map center when FOV changes.
    center = [(lb + ru) / 2.0 for lb, ru in zip(left_bottom_map, right_upper_map)]
    half_fov = D * pixel_size / 2.0
    left_bottom = [c - half_fov for c in center]
    right_upper = [c + half_fov for c in center]

    grid_dimensions = [D, D, D]
    voxel_sizes = [pixel_size, pixel_size, pixel_size]
    projection_depth = D * pixel_size

    if collapse_projection_axis:
        center_on_axis = center[projection_axis]
        grid_dimensions[projection_axis] = 1
        voxel_sizes[projection_axis] = projection_depth
        left_bottom[projection_axis] = center_on_axis
        right_upper[projection_axis] = center_on_axis

    lattice = Lattice.from_grid_dimensions_and_voxel_sizes(
        grid_dimensions=tuple(grid_dimensions),
        voxel_sizes_in_A=tuple(voxel_sizes),
        left_bottom_point_in_A=left_bottom,
        right_upper_point_in_A=right_upper,
        sublattice_radius_in_A=sublattice_radius,
        dtype=torch.float32,
        device=device,
    )

    lattice_meta = {
        "grid_dimensions": grid_dimensions,
        "voxel_sizes": voxel_sizes,
        "left_bottom": left_bottom,
        "right_upper": right_upper,
        "D": D,
        "pixel_size": pixel_size,
        "D_map": D_map,
        "pixel_size_map": pixel_size_map,
        "sublattice_radius": sublattice_radius,
        "projection_axis": projection_axis,
        "projection_depth": projection_depth,
        "collapse_projection_axis": collapse_projection_axis,
    }
    return lattice, lattice_meta


def lattice_from_meta(
    *,
    meta: dict[str, Any],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> Lattice:
    lattice_meta = meta.get("lattice", {})
    grid_dimensions = lattice_meta.get("grid_dimensions")
    voxel_sizes = lattice_meta.get("voxel_sizes")
    left_bottom = lattice_meta.get("left_bottom")
    right_upper = lattice_meta.get("right_upper")
    sublattice_radius = lattice_meta.get("sublattice_radius", 10.0)

    if grid_dimensions is None or voxel_sizes is None:
        raise ValueError("Missing lattice metadata: grid_dimensions and voxel_sizes are required.")

    return Lattice.from_grid_dimensions_and_voxel_sizes(
        grid_dimensions=tuple(grid_dimensions),
        voxel_sizes_in_A=tuple(voxel_sizes),
        left_bottom_point_in_A=left_bottom,
        right_upper_point_in_A=right_upper,
        sublattice_radius_in_A=sublattice_radius,
        dtype=dtype,
        device=device,
    )


class CryoImageDataset(TorchDataset[dict[str, Any]]):
    """
    Master dataset for cryo-image projections.

    - Stores one or more conformations loaded from .pt/.json files.
    - Exposes per-rotation items for DataLoader batching/shuffling.
    - Retains per-conformation tensors and metadata for full-tensor access.
    """

    AXIS_TO_DIM = {"x": 0, "y": 1, "z": 2}

    def __init__(self, conformations: list[dict[str, Any]]) -> None:
        self._conformations = conformations
        self._index: list[tuple[int, int]] = []
        for conf_idx, conformation in enumerate(self._conformations):
            num_rotations = int(conformation["rotations"].shape[0])
            for rot_idx in range(num_rotations):
                self._index.append((conf_idx, rot_idx))

    @classmethod
    def _resolve_projection_axis(cls, meta: dict[str, Any]) -> int:
        lattice_meta = meta.get("lattice", {})
        projection_axis = lattice_meta.get("projection_axis")
        if projection_axis is not None:
            return int(projection_axis)

        axis_name = meta.get("projection_axis")
        if axis_name not in cls.AXIS_TO_DIM:
            raise ValueError(
                "Missing projection axis metadata. Expected `lattice.projection_axis` "
                "or `projection_axis` in {'x','y','z'}."
            )
        return cls.AXIS_TO_DIM[axis_name]

    @staticmethod
    def _load_meta(pt_path: Path, loaded_pt_data: dict[str, Any], json_path: str | Path | None) -> dict[str, Any]:
        meta = loaded_pt_data.get("meta")
        if meta is not None:
            return meta
        if json_path is None:
            json_path = pt_path.with_suffix(".json")
        return json.loads(Path(json_path).read_text())

    @classmethod
    def load_conformation(
        cls,
        pt_path: str | Path,
        *,
        json_path: str | Path | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, Any]:
        device = torch.device(device)
        pt_path = Path(pt_path)
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict payload in {pt_path}, got {type(data)!r}")
        if "projections" not in data or "rotations" not in data:
            raise ValueError(f"Missing required tensors in {pt_path}; expected 'projections' and 'rotations'.")

        meta = cls._load_meta(pt_path, data, json_path)
        lattice = lattice_from_meta(meta=meta, device=device, dtype=dtype)
        projection_axis = cls._resolve_projection_axis(meta)
        collapse_projection_axis = bool(meta.get("lattice", {}).get("collapse_projection_axis", True))

        dose = meta.get("dose")
        has_poisson_noise = dose is not None and meta.get("ctf_enabled", False)

        return {
            "projections": data["projections"].to(device),
            "rotations": data["rotations"].to(device),
            "lattice": lattice,
            "projection_axis": projection_axis,
            "collapse_projection_axis": collapse_projection_axis,
            "meta": meta,
            "has_poisson_noise": has_poisson_noise,
        }

    @classmethod
    def from_paths(
        cls,
        *,
        pt_paths: list[str | Path],
        json_paths: list[str | Path] | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "CryoImageDataset":
        if json_paths is not None and len(json_paths) != len(pt_paths):
            raise ValueError("json_paths must have the same length as pt_paths.")
        conformations: list[dict[str, Any]] = []
        for idx, pt_path in enumerate(pt_paths):
            json_path = None if json_paths is None else json_paths[idx]
            conformation = cls.load_conformation(
                pt_path,
                json_path=json_path,
                device=device,
                dtype=dtype,
            )
            conformations.append(conformation)
        return cls(conformations=conformations)

    def __len__(self) -> int:
        return len(self._index)

    @property
    def num_conformations(self) -> int:
        return len(self._conformations)

    @property
    def conformations(self) -> list[dict[str, Any]]:
        return self._conformations

    @property
    def has_poisson_noise(self) -> bool:
        """True if all conformations were generated with Poisson noise (dose was set)."""
        return all(c.get("has_poisson_noise", False) for c in self._conformations)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        conf_idx, rot_idx = self._index[idx]
        conformation = self._conformations[conf_idx]
        return {
            "projection": conformation["projections"][rot_idx],
            "rotation": conformation["rotations"][rot_idx],
            "conformation_index": conf_idx,
            "rotation_index": rot_idx,
        }


class CryoImageRenderer:
    """
    Shared fast ESP projection renderer for cryo-image loss and dataset generation.
    """

    def __init__(
        self,
        *,
        lattice: Lattice,
        projection_axis: int,
        collapse_projection_axis: bool,
        fast_solver,
        ctf_params: CTFParams | None = None,
    ) -> None:
        self.lattice = lattice
        self.projection_axis = int(projection_axis)
        self.collapse_projection_axis = bool(collapse_projection_axis)
        _, self._compute_batch_from_coords = fast_solver
        self._grid_dims = tuple(int(x) for x in lattice.grid_dimensions.tolist())
        self._projection_depth = float(lattice.voxel_sizes_in_A[self.projection_axis].item())

        self.ctf_params = ctf_params
        self._ctf_2d: torch.Tensor | None = None
        if ctf_params is not None:
            freq_grid = self.lattice.frequency_grid_in_m_2d  # (D*D, 2)
            ctf_flat = compute_ctf_2d(freq_grid[:, 0], freq_grid[:, 1], ctf_params)
            proj_shape = projection_shape(self._grid_dims, self.projection_axis)
            self._ctf_2d = torch.fft.ifftshift(
                ctf_flat.reshape(proj_shape)
            ).to(lattice.device)

    @classmethod
    def default_ctf_params(cls) -> CTFParams:
        return CTFParams(
            delta_f=17136.9570315e-10,
            A1=124.1328125e-10,
            alpha_1=4.574108603237696,
            C_s=3.00e-3,
            C_c=2.00e-3,
            delta_E=0.7,
            voltage_kV=300.0,
            alpha_i=0.1e-3,
            d_ap=100e-6,
            f=2.0e-3,
            q_cuton=None,
            A=None,
            objective_plane_z_A=140.0,
            slice_thickness_delta_z_A=5.0,
        )

    @staticmethod
    def _electron_wavelength_m(voltage_kv: float) -> float:
        """Relativistic electron wavelength in meters."""
        h = 6.626e-34
        m_e = 9.109e-31
        e = 1.602e-19
        c = 3.0e8
        V = voltage_kv * 1e3
        return h / math.sqrt(2 * m_e * e * V * (1 + e * V / (2 * m_e * c ** 2)))

    def apply_forward_model(self, projections: torch.Tensor) -> torch.Tensor:
        """Convert raw ESP projections (B, D, D) to cryo-EM images via exit wavelets + CTF."""
        if self._ctf_2d is None:
            raise RuntimeError("Forward model requires ctf_params to be set.")

        # Exit wavelets: 1e10 * exp(i * lambda_A * V_z)  where lambda_A = lambda_m * 1e10
        lam_m = self._electron_wavelength_m(self.ctf_params.voltage_kV)
        lam_A = lam_m * 1e10  # wavelength in Angstroms
        exit_wave = 1e10 * torch.exp(1j * lam_A * projections)  # (B, D, D) complex

        # CTF convolution in Fourier space
        convolved = torch.fft.ifft2(torch.fft.fft2(exit_wave) * self._ctf_2d)
        return convolved.abs() ** 2 / 1e20  # (B, D, D), real, ~1.0 + tiny modulation

    @staticmethod
    def add_poisson_noise(images: torch.Tensor, dose_per_A2: float, pixel_size_A: float) -> torch.Tensor:
        """Apply Poisson noise. dose_per_A2 is electrons/A^2 (typical: 40-80), pixel_size_A in Angstroms."""
        dose_per_pixel = dose_per_A2 * pixel_size_A ** 2
        return torch.poisson(images * dose_per_pixel)

    @classmethod
    def from_atom_stack(
        cls,
        *,
        atom_stack: AtomStack,
        lattice: Lattice,
        projection_axis: int,
        collapse_projection_axis: bool,
        use_checkpointing: bool = False,
        use_autocast: bool = False,
        ctf_params: CTFParams | None = None,
    ) -> "CryoImageRenderer":
        fast_solver = setup_fast_esp_solver(
            atom_stack,
            lattice,
            per_voxel_averaging=True,
            use_checkpointing=use_checkpointing,
            use_autocast=use_autocast,
        )
        return cls(
            lattice=lattice,
            projection_axis=projection_axis,
            collapse_projection_axis=collapse_projection_axis,
            fast_solver=fast_solver,
            ctf_params=ctf_params,
        )

    @classmethod
    def from_coords_and_atomic_numbers(
        cls,
        *,
        coords: torch.Tensor,
        atomic_numbers: torch.Tensor,
        bfactors: torch.Tensor,
        lattice: Lattice,
        projection_axis: int,
        collapse_projection_axis: bool,
        device: torch.device | str,
        use_checkpointing: bool = False,
        use_autocast: bool = False,
        ctf_params: CTFParams | None = None,
    ) -> "CryoImageRenderer":
        if coords.ndim == 2:
            coords = coords.unsqueeze(0)
        if bfactors.ndim == 1:
            bfactors = bfactors.unsqueeze(0)
        if bfactors.ndim == 2:
            bfactors = bfactors.unsqueeze(-1)

        atom_stack = AtomStack.from_coords_and_atomic_numbers(
            atom_coordinates=coords,
            atomic_numbers=atomic_numbers,
            device=device,
        )
        atom_stack.bfactors = bfactors
        if atom_stack.occupancies is None:
            atom_stack.occupancies = torch.ones(
                (coords.shape[0],),
                dtype=coords.dtype,
                device=coords.device,
            )

        return cls.from_atom_stack(
            atom_stack=atom_stack,
            lattice=lattice,
            projection_axis=projection_axis,
            collapse_projection_axis=collapse_projection_axis,
            use_checkpointing=use_checkpointing,
            use_autocast=use_autocast,
            ctf_params=ctf_params,
        )

    @staticmethod
    def apply_atom_mask(
        coords: torch.Tensor,
        atomic_numbers: torch.Tensor | None,
        bfactors: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if coords.ndim == 2:
            coords = coords[mask]
        elif coords.ndim == 3:
            coords = coords[:, mask, :]

        if atomic_numbers is not None:
            atomic_numbers = atomic_numbers[mask]

        if bfactors is not None:
            bfactors = bfactors[mask]

        return coords, atomic_numbers, bfactors

    def render_all_rotations_from_coords(
        self,
        *,
        coords: torch.Tensor,
        atomic_numbers: torch.Tensor,
        bfactors: torch.Tensor,
        rotations: torch.Tensor,
        max_rotations_per_batch: int | None = None,
        occupancies: torch.Tensor | None = None,
        apply_ctf: bool = True,
    ) -> torch.Tensor:
        batch = coords.shape[0]
        if bfactors.ndim == 1:
            bfactors = bfactors.unsqueeze(0)
        if bfactors.ndim == 2:
            bfactors = bfactors.unsqueeze(-1)

        atomic_numbers_for_solver = atomic_numbers.contiguous()
        if atomic_numbers_for_solver.ndim == 1:
            atomic_numbers_for_solver = atomic_numbers_for_solver.unsqueeze(-1)

        bfactors_for_solver = bfactors.contiguous()
        if occupancies is None:
            occupancies_for_solver = torch.ones(
                (batch,),
                dtype=coords.dtype,
                device=coords.device,
            )
        else:
            occupancies_for_solver = occupancies.reshape(-1)
            if occupancies_for_solver.shape[0] == 1 and batch > 1:
                occupancies_for_solver = occupancies_for_solver.expand(batch)
            if occupancies_for_solver.shape[0] != batch:
                raise ValueError(
                    "Occupancies must have one entry per structure in the batch."
                )

        num_rots = int(rotations.shape[0])
        max_rots = (
            num_rots
            if max_rotations_per_batch is None or max_rotations_per_batch <= 0
            else int(max_rotations_per_batch)
        )

        projections = []
        for rot_start in range(0, num_rots, max_rots):
            rot_end = min(rot_start + max_rots, num_rots)
            rot_batch = rotations[rot_start:rot_end]
            rot_count = int(rot_batch.shape[0])

            coords_for_transform = coords.repeat(rot_count, 1, 1)
            rotations_for_transform = rot_batch.repeat_interleave(batch, dim=0)

            transform_stack = AtomStack.from_coords_and_atomic_numbers(
                atom_coordinates=coords_for_transform,
                atomic_numbers=atomic_numbers_for_solver,
                device=coords.device,
            )
            coords_batch = transform_stack.apply_rigid_transform(
                rotation=rotations_for_transform,
                translation=None,
                center=None,
                copy=False,
            ).atom_coordinates.reshape(rot_count, batch, coords.shape[1], 3)

            volumes = self._compute_batch_from_coords(
                coords_batch,
                bfactors_for_solver,
                atomic_numbers_for_solver,
                occupancies_for_solver,
            )

            volumes = volumes.squeeze(1)

            projection_dim = self.projection_axis + 1
            if self.collapse_projection_axis:
                projection = volumes.squeeze(dim=projection_dim) * self._projection_depth
            else:
                projection = volumes.sum(dim=projection_dim) * self._projection_depth

            projections.append(projection)

        result = torch.cat(projections, dim=0)
        if apply_ctf and self._ctf_2d is not None:
            result = self.apply_forward_model(result)
        return result
