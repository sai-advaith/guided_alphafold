from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset

from cryoforward.atom_stack import AtomStack
from cryoforward.cryoesp_calculator import setup_fast_esp_solver
from cryoforward.lattice import Lattice


def projection_shape(grid_dims: tuple[int, int, int], axis: int) -> tuple[int, int]:
    if axis == 0:
        return (grid_dims[1], grid_dims[2])
    if axis == 1:
        return (grid_dims[0], grid_dims[2])
    return (grid_dims[0], grid_dims[1])


def prepare_lattice_from_atom_stack(
    *,
    atom_stack: AtomStack,
    voxel_size: float,
    padding: float,
    sublattice_radius: float,
    projection_axis: int,
    collapse_projection_axis: bool,
    device: torch.device | str,
) -> tuple[Lattice, dict[str, Any], torch.Tensor]:
    coords = atom_stack.atom_coordinates[0]
    center = coords.mean(dim=0)
    radius = torch.linalg.norm(coords - center, dim=1).max().item()

    half_span = radius + padding
    span = 2.0 * half_span
    grid_dim = int(math.ceil(span / voxel_size)) + 1
    half_span_adjusted = 0.5 * (grid_dim - 1) * voxel_size

    center_cpu = center.detach().cpu().numpy()
    left_bottom = (center_cpu - half_span_adjusted).tolist()
    right_upper = (center_cpu + half_span_adjusted).tolist()

    grid_dimensions = [grid_dim, grid_dim, grid_dim]
    voxel_sizes = [voxel_size, voxel_size, voxel_size]
    projection_depth = grid_dim * voxel_size

    if collapse_projection_axis:
        grid_dimensions[projection_axis] = 1
        voxel_sizes[projection_axis] = projection_depth
        left_bottom[projection_axis] = float(center_cpu[projection_axis])
        right_upper[projection_axis] = float(center_cpu[projection_axis])

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
        "center": center_cpu.tolist(),
        "radius": radius,
        "padding": padding,
        "sublattice_radius": sublattice_radius,
        "projection_axis": projection_axis,
        "projection_depth": projection_depth,
        "collapse_projection_axis": collapse_projection_axis,
    }
    return lattice, lattice_meta, center.unsqueeze(0)


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
        data = torch.load(pt_path, map_location="cpu")
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict payload in {pt_path}, got {type(data)!r}")
        if "projections" not in data or "rotations" not in data:
            raise ValueError(f"Missing required tensors in {pt_path}; expected 'projections' and 'rotations'.")

        meta = cls._load_meta(pt_path, data, json_path)
        lattice = lattice_from_meta(meta=meta, device=device, dtype=dtype)
        projection_axis = cls._resolve_projection_axis(meta)
        collapse_projection_axis = bool(meta.get("lattice", {}).get("collapse_projection_axis", True))

        return {
            "projections": data["projections"].to(device),
            "rotations": data["rotations"].to(device),
            "lattice": lattice,
            "projection_axis": projection_axis,
            "collapse_projection_axis": collapse_projection_axis,
            "meta": meta,
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

    def get_conformation(self, conformation_index: int) -> dict[str, Any]:
        return self._conformations[conformation_index]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        conf_idx, rot_idx = self._index[idx]
        conformation = self._conformations[conf_idx]
        return {
            "projection": conformation["projections"][rot_idx],
            "rotation": conformation["rotations"][rot_idx],
            "conformation_index": conf_idx,
            "rotation_index": rot_idx,
            "projection_axis": int(conformation["projection_axis"]),
            "collapse_projection_axis": bool(conformation["collapse_projection_axis"]),
        }


class CryoImageDataLoader(TorchDataLoader):
    """
    Torch DataLoader specialized for CryoImageDataset construction from .pt/.json files.
    """

    def __init__(
        self,
        *,
        pt_paths: list[str | Path],
        json_paths: list[str | Path] | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool | None = None,
        persistent_workers: bool | None = None,
        prefetch_factor: int | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.cryo_dataset = CryoImageDataset.from_paths(
            pt_paths=pt_paths,
            json_paths=json_paths,
            device=self.device,
            dtype=self.dtype,
        )

        if num_workers > 0 and self.device.type != "cpu":
            raise ValueError(
                "num_workers > 0 requires CPU-backed dataset tensors. "
                "Initialize CryoImageDataLoader with device='cpu' for multi-worker loading."
            )

        if pin_memory is None:
            pin_memory = self.device.type == "cuda"
        if persistent_workers is None:
            persistent_workers = num_workers > 0

        kwargs: dict[str, Any] = {
            "dataset": self.cryo_dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "drop_last": drop_last,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers if num_workers > 0 else False,
            "generator": generator,
        }
        if num_workers > 0 and prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)

        super().__init__(**kwargs)

    @classmethod
    def as_dataset(
        cls,
        *,
        pt_paths: list[str | Path],
        json_paths: list[str | Path] | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> CryoImageDataset:
        return CryoImageDataset.from_paths(
            pt_paths=pt_paths,
            json_paths=json_paths,
            device=device,
            dtype=dtype,
        )


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
    ) -> None:
        self.lattice = lattice
        self.projection_axis = int(projection_axis)
        self.collapse_projection_axis = bool(collapse_projection_axis)
        self.fast_solver = fast_solver
        _, self._compute_batch_from_coords = fast_solver
        self._grid_dims = tuple(int(x) for x in lattice.grid_dimensions.tolist())
        self._projection_depth = float(lattice.voxel_sizes_in_A[self.projection_axis].item())

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
    ) -> "CryoImageRenderer":
        if coords.ndim == 2:
            coords = coords.unsqueeze(0)
        if bfactors.ndim == 1:
            bfactors = bfactors.unsqueeze(0)
        if bfactors.ndim == 2:
            bfactors = bfactors.unsqueeze(-1)
        if bfactors.shape[0] == 1 and coords.shape[0] > 1:
            bfactors = bfactors.expand(coords.shape[0], -1, -1)

        atom_stack = AtomStack.from_coords_and_atomic_numbers(
            atom_coordinates=coords,
            atomic_numbers=atomic_numbers,
            device=device,
        )
        atom_stack.bfactors = bfactors
        if getattr(atom_stack, "occupancies", None) is None:
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
            if atomic_numbers.ndim == 1:
                atomic_numbers = atomic_numbers[mask]
            elif atomic_numbers.ndim == 2:
                if atomic_numbers.shape[0] == mask.shape[0]:
                    atomic_numbers = atomic_numbers[mask]
                elif atomic_numbers.shape[1] == mask.shape[0]:
                    atomic_numbers = atomic_numbers[:, mask]

        if bfactors is not None:
            if bfactors.ndim == 1:
                bfactors = bfactors[mask]
            elif bfactors.ndim == 2:
                bfactors = bfactors[:, mask]
            elif bfactors.ndim == 3:
                bfactors = bfactors[:, mask, :]

        return coords, atomic_numbers, bfactors

    def render_all_rotations_from_coords(
        self,
        *,
        coords: torch.Tensor,
        atomic_numbers: torch.Tensor,
        bfactors: torch.Tensor,
        rotations: torch.Tensor,
        max_rotations_per_batch: int | None = None,
        center: torch.Tensor | None = None,
        occupancies: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if coords.ndim == 2:
            coords = coords.unsqueeze(0)
        if rotations.ndim == 2:
            rotations = rotations.unsqueeze(0)

        batch = coords.shape[0]
        if bfactors.ndim == 1:
            bfactors = bfactors.unsqueeze(0)
        if bfactors.ndim == 2:
            bfactors = bfactors.unsqueeze(-1)
        if bfactors.shape[0] == 1 and batch > 1:
            bfactors = bfactors.expand(batch, -1, -1)

        if center is None:
            center = coords.mean(dim=1, keepdim=True)
        else:
            if center.ndim == 1:
                center = center.unsqueeze(0).unsqueeze(1)
            elif center.ndim == 2:
                center = center.unsqueeze(1)
        coords_centered = coords - center

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
        max_rots = num_rots if max_rotations_per_batch is None or max_rotations_per_batch <= 0 else int(max_rotations_per_batch)

        projections = []
        for rot_start in range(0, num_rots, max_rots):
            rot_end = min(rot_start + max_rots, num_rots)
            rot_batch = rotations[rot_start:rot_end]
            # Match cryoforward AtomStack.rotate convention: (x - c) @ R.T + c
            coords_batch = torch.einsum("rji,bnj->rbni", rot_batch, coords_centered) + center.unsqueeze(0)

            volumes = self._compute_batch_from_coords(
                coords_batch,
                bfactors_for_solver,
                atomic_numbers_for_solver,
                occupancies_for_solver,
            )

            if volumes.ndim == 2:
                volumes = volumes.reshape(-1, *self._grid_dims)
            elif volumes.ndim == 5 and volumes.shape[1] == 1:
                volumes = volumes.squeeze(1)

            projection_dim = self.projection_axis + 1
            if self.collapse_projection_axis:
                projection = volumes.squeeze(dim=projection_dim) * self._projection_depth
            else:
                projection = volumes.sum(dim=projection_dim) * self._projection_depth

            projections.append(projection)

        return torch.cat(projections, dim=0)
