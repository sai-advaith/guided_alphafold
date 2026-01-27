#!/usr/bin/env python3

import argparse
import json
import math
import shutil
import warnings
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from cryoforward.atom_stack import AtomStack
    from cryoforward.cryoesp_calculator import compute_volume_over_insertable_matrices
    from cryoforward.lattice import Lattice
    from cryoforward.utils.rigid_transform import Rotation
except ImportError as exc:
    raise ImportError(
        "cryoforward is not importable. Install it via pip or activate the correct environment."
    ) from exc


PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

AXIS_TO_DIM = {
    "x": 0,
    "y": 1,
    "z": 2,
}


def _progress(iterable, enabled: bool, desc: str):
    if enabled and tqdm is not None:
        return tqdm(iterable, desc=desc)
    return iterable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ESP projections from random rotations of PDBs using cryoforward."
        )
    )
    parser.add_argument(
        "--pdb-file",
        action="append",
        dest="pdb_files",
        default=[],
        help="Path to a PDB file (repeatable).",
    )
    parser.add_argument(
        "--pdb-id",
        action="append",
        dest="pdb_ids",
        default=[],
        help="PDB ID to download (repeatable, e.g. 7T54).",
    )
    parser.add_argument(
        "--num-rotations",
        type=int,
        default=1000,
        help="Number of random rotations per structure.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=2.0,
        help="Target resolution in Angstroms (used to derive voxel size if not provided).",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Voxel size in Angstroms. If omitted, uses resolution/2 (Nyquist).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=10.0,
        help="Padding in Angstroms added to the bounding sphere.",
    )
    parser.add_argument(
        "--sublattice-radius",
        type=float,
        default=14.0,
        help="Sublattice radius in Angstroms for the ESP computation.",
    )
    parser.add_argument(
        "--projection-axis",
        choices=sorted(AXIS_TO_DIM.keys()),
        default="z",
        help="Axis along which to sum the volume for projection.",
    )
    parser.add_argument(
        "--collapse-projection-axis",
        action="store_true",
        default=True,
        help=(
            "Collapse the projection axis to a single voxel and scale by the depth "
            "(default)."
        ),
    )
    parser.add_argument(
        "--full-lattice",
        action="store_false",
        dest="collapse_projection_axis",
        help="Use a full 3D lattice and sum along the projection axis.",
    )
    parser.add_argument(
        "--atom-batch-size",
        type=int,
        default=1024,
        help="Atom batch size for compute_volume_over_insertable_matrices.",
    )
    parser.add_argument(
        "--bfactor",
        type=float,
        default=None,
        help="Override all B-factors with a constant value (Angstrom^2).",
    )
    parser.add_argument(
        "--atomic-radius",
        type=float,
        default=0.5,
        help="Atomic radius used to compute fallback B-factors when missing.",
    )
    parser.add_argument(
        "--output-format",
        choices=["pt", "npz"],
        default="pt",
        help="Output format for projections and rotations.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT / "guided-alphafold-utils" / "datasets"),
        help="Output directory for generated datasets.",
    )
    parser.add_argument(
        "--pdb-cache-dir",
        type=str,
        default=None,
        help="Directory for downloaded PDBs (defaults to <out-dir>/pdbs).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (e.g., cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible rotations.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        default=True,
        help="Show a progress bar if tqdm is available (default).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        help="Disable the progress bar.",
    )
    parser.add_argument(
        "--clear-cache-every",
        type=int,
        default=0,
        help="Clear CUDA cache every N iterations (0 disables).",
    )
    args = parser.parse_args()
    if not args.pdb_files and not args.pdb_ids:
        parser.error("At least one --pdb-file or --pdb-id must be provided.")
    return args


def _download_pdb_if_not_cached(pdb_id: str, cache_dir: Path) -> Path:
    cleaned_id = pdb_id.strip()
    if not cleaned_id:
        raise ValueError("Empty PDB ID provided.")
    pdb_id_upper = cleaned_id.upper()

    cache_dir.mkdir(parents=True, exist_ok=True)
    dest_path = cache_dir / f"{pdb_id_upper}.pdb"
    if dest_path.exists():
        print(f"Using cached PDB file of {pdb_id_upper} at {dest_path}")
        return dest_path


    print(f"Downloading PDB file {pdb_id_upper}")
    url = PDB_DOWNLOAD_URL.format(pdb_id=pdb_id_upper)
    try:
        with urlopen(url) as response, dest_path.open("wb") as f:
            shutil.copyfileobj(response, f)
    except URLError as exc:
        raise RuntimeError(f"Failed to download PDB {pdb_id_upper} from {url}") from exc

    return dest_path


def _projection_shape(grid_dims: tuple[int, int, int], axis: int) -> tuple[int, int]:
    if axis == 0:
        return (grid_dims[1], grid_dims[2])
    if axis == 1:
        return (grid_dims[0], grid_dims[2])
    return (grid_dims[0], grid_dims[1])


def _prepare_lattice(
    atom_stack: AtomStack,
    voxel_size: float,
    padding: float,
    sublattice_radius: float,
    projection_axis: int,
    collapse_projection_axis: bool,
    device: torch.device,
) -> tuple[Lattice, dict, torch.Tensor]:
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


def _ensure_bfactors(atom_stack: AtomStack, bfactor_override: float | None, atomic_radius: float) -> None:
    if bfactor_override is not None:
        atom_stack.fill_constant_bfactor(bfactor_override)
        return
    if atom_stack.bfactors is not None:
        return
    bfactor = 8.0 * torch.pi**2 * atomic_radius**2
    atom_stack.fill_constant_bfactor(float(bfactor))


def _save_outputs(
    projections: torch.Tensor,
    rotations: torch.Tensor,
    meta: dict,
    output_path: Path,
    output_format: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "pt":
        torch.save(
            {
                "projections": projections,
                "rotations": rotations,
                "meta": meta,
            },
            output_path,
        )
        meta_path = output_path.with_suffix(".json")
    else:
        np.savez_compressed(
            output_path,
            projections=projections.numpy(),
            rotations=rotations.numpy(),
        )
        meta_path = output_path.with_suffix(".json")

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _render_projection(
    atom_stack: AtomStack,
    lattice: Lattice,
    projection_axis: int,
    collapse_projection_axis: bool,
    atom_batch_size: int,
    center: torch.Tensor | None = None,
    rotation: Rotation | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rotation is None:
        rotation = Rotation.random(batch=1, device=atom_stack.device)

    rotated_stack = atom_stack.rotate(rotation, center=center, copy=True)

    with torch.no_grad():
        volume = compute_volume_over_insertable_matrices(
            atom_stack=rotated_stack,
            lattice=lattice,
            B=atom_batch_size,
            per_voxel_averaging=True,
            subvolume_mask_in_indices=None,
            use_checkpointing=False,
            verbose=False,
        )
        if collapse_projection_axis:
            depth = float(lattice.voxel_sizes_in_A[projection_axis].item())
            projection = volume.squeeze(dim=projection_axis) * depth
        else:
            depth = float(lattice.voxel_sizes_in_A[projection_axis].item())
            projection = volume.sum(dim=projection_axis) * depth

    rotation_matrix = rotation.to_matrix().detach().cpu()[0]
    return projection, rotation_matrix


def _generate_dataset_for_pdb(
    pdb_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    axis = AXIS_TO_DIM[args.projection_axis]

    atom_stack = AtomStack.from_pdb_file(str(pdb_path), device=device)
    _ensure_bfactors(atom_stack, args.bfactor, args.atomic_radius)

    voxel_size = args.voxel_size
    if voxel_size is None:
        voxel_size = args.resolution / 2.0

    lattice, lattice_meta, center = _prepare_lattice(
        atom_stack=atom_stack,
        voxel_size=voxel_size,
        padding=args.padding,
        sublattice_radius=args.sublattice_radius,
        projection_axis=axis,
        collapse_projection_axis=args.collapse_projection_axis,
        device=device,
    )
    if not args.collapse_projection_axis:
        depth = float(lattice.grid_dimensions[axis].item() * lattice.voxel_sizes_in_A[axis].item())
        if 2.0 * args.sublattice_radius < depth:
            warnings.warn(
                "Sublattice radius is smaller than the projection depth; "
                "full-sum projections may be underestimated. Consider increasing "
                "--sublattice-radius or using the collapsed projection mode.",
                RuntimeWarning,
            )

    grid_dims = tuple(lattice.grid_dimensions.tolist())
    proj_shape = _projection_shape(grid_dims, axis)

    projections = torch.empty(
        (args.num_rotations, proj_shape[0], proj_shape[1]),
        dtype=torch.float32,
        device="cpu",
    )
    rotations = torch.empty(
        (args.num_rotations, 3, 3),
        dtype=torch.float32,
        device="cpu",
    )

    rotation_iter = _progress(
        range(args.num_rotations),
        args.progress,
        desc=f"{pdb_path.stem} rotations",
    )

    for idx in rotation_iter:
        projection, rotation_matrix = _render_projection(
            atom_stack=atom_stack,
            lattice=lattice,
            projection_axis=axis,
            collapse_projection_axis=args.collapse_projection_axis,
            atom_batch_size=args.atom_batch_size,
            center=center,
        )
        projections[idx] = projection.detach().cpu()
        rotations[idx] = rotation_matrix

        if args.clear_cache_every > 0 and device.type == "cuda":
            if (idx + 1) % args.clear_cache_every == 0:
                torch.cuda.empty_cache()

    pdb_id = pdb_path.stem
    out_dir = Path(args.out_dir)
    out_name = f"{pdb_id}_esp_projections_{args.num_rotations}.{args.output_format}"
    output_path = out_dir / out_name

    meta = {
        "pdb_id": pdb_id,
        "pdb_path": str(pdb_path),
        "num_rotations": args.num_rotations,
        "projection_axis": args.projection_axis,
        "voxel_size": voxel_size,
        "resolution": args.resolution,
        "atom_batch_size": args.atom_batch_size,
        "bfactor_override": args.bfactor,
        "atomic_radius": args.atomic_radius,
        "device": str(device),
        "lattice": lattice_meta,
    }

    _save_outputs(projections, rotations, meta, output_path, args.output_format)


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pdb_paths = [Path(p) for p in args.pdb_files]
    pdb_cache_dir = Path(args.pdb_cache_dir) if args.pdb_cache_dir else Path(args.out_dir) / "pdbs"
    for pdb_id in args.pdb_ids:
        pdb_paths.append(_download_pdb_if_not_cached(pdb_id, pdb_cache_dir))

    for pdb_path in pdb_paths:
        _generate_dataset_for_pdb(pdb_path, args, device)


if __name__ == "__main__":
    main()
