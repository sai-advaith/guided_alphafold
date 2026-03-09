#!/usr/bin/env python3

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import gemmi
import numpy as np
import torch
import yaml

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_pdb_atom_locations_full
from src.protenix.metrics.rmsd import self_aligned_rmsd

try:
    from src.utils.cryoimage_renderer import (
        CryoImageRenderer,
        prepare_lattice_from_density_map,
        projection_shape,
    )
    from cryoforward.atom_stack import AtomStack
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


def _canonical_output_pdb_id(pdb_path: Path) -> str:
    # Normalize dataset IDs so outputs are stable regardless input path casing.
    return pdb_path.stem.upper()


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
        "--config",
        type=str,
        default=None,
        help=(
            "Optional pipeline YAML to use AF3-style sequences. "
            "If set, unresolved atoms are masked out to match guidance."
        ),
    )
    parser.add_argument(
        "--density-map",
        type=str,
        required=True,
        help="Path to a CCP4 density map file. Grid dimensions and voxel size are read from this map.",
    )
    parser.add_argument(
        "--num-rotations",
        type=int,
        default=1000,
        help="Number of random rotations per structure.",
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
        default=str(REPO_ROOT / "pipeline_inputs" / "projections"),
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
    parser.add_argument(
        "--align-to-reference-pdb",
        type=str,
        default=None,
        help=(
            "Optional reference PDB used to rigidly align the config-restricted atom set "
            "before rendering. Requires --config so the alignment is done on the configured "
            "sequence/chains rather than the full PDB."
        ),
    )
    parser.add_argument(
        "--aligned-pdb-out",
        type=str,
        default=None,
        help=(
            "Optional output path for saving the input PDB after applying the rigid transform "
            "fit from --align-to-reference-pdb."
        ),
    )
    args = parser.parse_args()
    if not args.pdb_files and not args.pdb_ids:
        parser.error("At least one --pdb-file or --pdb-id must be provided.")
    if args.align_to_reference_pdb is not None and args.config is None:
        parser.error("--align-to-reference-pdb requires --config.")
    if args.aligned_pdb_out is not None and args.align_to_reference_pdb is None:
        parser.error("--aligned-pdb-out requires --align-to-reference-pdb.")
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
    meta_path = output_path.with_suffix(".json")

    if output_format == "pt":
        torch.save(
            {
                "projections": projections,
                "rotations": rotations,
                "meta": meta,
            },
            output_path,
        )
    else:
        np.savez_compressed(
            output_path,
            projections=projections.numpy(),
            rotations=rotations.numpy(),
        )

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _render_projection(
    atom_stack: AtomStack,
    fast_renderer: CryoImageRenderer,
    rotation: Rotation | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rotation is None:
        rotation = Rotation.random(batch=1, device=atom_stack.device)

    rot_matrix = rotation.to_matrix()
    with torch.no_grad():
        projection = fast_renderer.render_all_rotations_from_coords(
            coords=atom_stack.atom_coordinates,
            atomic_numbers=atom_stack.atomic_numbers,
            bfactors=atom_stack.bfactors,
            rotations=rot_matrix,
            max_rotations_per_batch=1,
            occupancies=atom_stack.occupancies,
        )[0]

    return projection, rot_matrix.detach().cpu()[0]


def _load_sequences_from_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sequences = cfg["protein"]["sequences"]
    for entry in sequences:
        entry.setdefault("sequence_type", "proteinChain")
    chains_to_read = cfg["protein"].get("chains_to_use")
    bfactor_override = cfg.get("loss_function", {}).get("cryoimage_loss_function", {}).get("bfactor_override")
    return sequences, chains_to_read, bfactor_override


def _load_config_atom_payload(
    pdb_path: Path,
    sequences,
    chains_to_read,
    device: torch.device,
):
    load_result = load_pdb_atom_locations_full(
        pdb_file=str(pdb_path),
        full_sequences_dict=sequences,
        chains_to_read=chains_to_read,
        return_elements=True,
        return_bfacs=True,
        return_mask=True,
        return_starting_indices=True,
        device=device,
    )
    coords, mask, bfactors, elements, starting_residue_indices = load_result
    mask = mask.to(dtype=torch.bool)
    if int(mask.sum().item()) == 0:
        raise ValueError("No resolved atoms found after masking; check sequences/chains.")
    return {
        "coords_full": coords,
        "mask": mask,
        "bfactors_full": bfactors,
        "elements_full": elements,
        "starting_residue_indices": starting_residue_indices,
    }


def _atom_stack_from_config(
    pdb_path: Path,
    sequences,
    chains_to_read,
    device: torch.device,
):
    payload = _load_config_atom_payload(pdb_path, sequences, chains_to_read, device)
    mask = payload["mask"]
    coords = payload["coords_full"][mask].unsqueeze(0)
    elements = payload["elements_full"][mask]
    bfactors = payload["bfactors_full"][mask]
    bfactors = bfactors.reshape(1, -1, 1).to(dtype=torch.float32, device=device)

    atom_stack = AtomStack.from_coords_and_atomic_numbers(
        atom_coordinates=coords,
        atomic_numbers=elements,
        device=device,
    )
    atom_stack.bfactors = bfactors
    return atom_stack, int(mask.sum().item()), int(mask.numel())


def _align_atom_stack_from_config_to_reference(
    *,
    pdb_path: Path,
    reference_pdb_path: Path,
    sequences,
    chains_to_read,
    device: torch.device,
) -> tuple[AtomStack, dict[str, object], torch.Tensor, torch.Tensor]:
    moving = _load_config_atom_payload(pdb_path, sequences, chains_to_read, device)
    reference = _load_config_atom_payload(reference_pdb_path, sequences, chains_to_read, device)

    common_mask = moving["mask"] & reference["mask"]
    common_atoms = int(common_mask.sum().item())
    if common_atoms <= 0:
        raise ValueError(
            "No common resolved config atoms were found between the moving and reference PDBs."
        )

    alignment_rmsd, aligned_coords_full, rotation, translation = self_aligned_rmsd(
        moving["coords_full"].unsqueeze(0),
        reference["coords_full"].unsqueeze(0),
        common_mask,
    )

    aligned_coords = aligned_coords_full[0, moving["mask"], :]
    elements = moving["elements_full"][moving["mask"]]
    bfactors = moving["bfactors_full"][moving["mask"]].reshape(1, -1, 1).to(dtype=torch.float32, device=device)

    atom_stack = AtomStack.from_coords_and_atomic_numbers(
        atom_coordinates=aligned_coords.unsqueeze(0),
        atomic_numbers=elements,
        device=device,
    )
    atom_stack.bfactors = bfactors

    alignment_info = {
        "reference_pdb_path": str(reference_pdb_path),
        "moving_resolved_atoms": int(moving["mask"].sum().item()),
        "moving_total_atoms": int(moving["mask"].numel()),
        "common_resolved_atoms": common_atoms,
        "rmsd": float(alignment_rmsd.detach().item()),
        "rotation": rotation[0].detach().cpu().tolist(),
        "translation": translation.reshape(-1, 3)[0].detach().cpu().tolist(),
    }
    return (
        atom_stack,
        alignment_info,
        rotation[0].detach().cpu(),
        translation.reshape(-1, 3)[0].detach().cpu(),
    )


def _write_transformed_pdb(
    *,
    input_pdb_path: Path,
    output_pdb_path: Path,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> Path:
    structure = gemmi.read_structure(str(input_pdb_path))
    rot = rotation.detach().cpu().numpy()
    trans = translation.detach().cpu().numpy()

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coord = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)
                    new_coord = coord @ rot.T + trans
                    atom.pos = gemmi.Position(float(new_coord[0]), float(new_coord[1]), float(new_coord[2]))

    output_pdb_path.parent.mkdir(parents=True, exist_ok=True)
    structure.write_pdb(str(output_pdb_path))
    return output_pdb_path


def _generate_dataset_for_pdb(
    pdb_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    sequences=None,
    chains_to_read=None,
    config_bfactor_override: float | None = None,
) -> None:
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    axis = AXIS_TO_DIM[args.projection_axis]

    resolved_info = None
    alignment_info = None
    alignment_rotation = None
    alignment_translation = None
    if sequences is not None:
        if args.align_to_reference_pdb is not None:
            atom_stack, alignment_info, alignment_rotation, alignment_translation = _align_atom_stack_from_config_to_reference(
                pdb_path=pdb_path,
                reference_pdb_path=Path(args.align_to_reference_pdb),
                sequences=sequences,
                chains_to_read=chains_to_read,
                device=device,
            )
            n_resolved = int(alignment_info["moving_resolved_atoms"])
            n_total = int(alignment_info["moving_total_atoms"])
        else:
            atom_stack, n_resolved, n_total = _atom_stack_from_config(
                pdb_path, sequences, chains_to_read, device
            )
        resolved_info = {"resolved_atoms": n_resolved, "total_atoms": n_total}
    else:
        atom_stack = AtomStack.from_pdb_file(str(pdb_path), device=device)

    effective_bfactor_override = args.bfactor
    if effective_bfactor_override is None:
        effective_bfactor_override = config_bfactor_override
    _ensure_bfactors(atom_stack, effective_bfactor_override, args.atomic_radius)

    lattice, lattice_meta = prepare_lattice_from_density_map(
        density_map_path=args.density_map,
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
    proj_shape = projection_shape(grid_dims, axis)

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

    fast_renderer = CryoImageRenderer.from_atom_stack(
        atom_stack=atom_stack,
        lattice=lattice,
        projection_axis=axis,
        collapse_projection_axis=args.collapse_projection_axis,
        use_checkpointing=False,
        use_autocast=False,
    )

    rotation_iter = _progress(
        range(args.num_rotations),
        args.progress,
        desc=f"{pdb_path.stem} rotations",
    )

    for idx in rotation_iter:
        projection, rotation_matrix = _render_projection(
            atom_stack,
            fast_renderer,
        )
        projections[idx] = projection.detach().cpu()
        rotations[idx] = rotation_matrix

        if args.clear_cache_every > 0 and device.type == "cuda":
            if (idx + 1) % args.clear_cache_every == 0:
                torch.cuda.empty_cache()

    pdb_id = _canonical_output_pdb_id(pdb_path)
    out_dir = Path(args.out_dir)
    out_name = f"{pdb_id}_esp_projections_{args.num_rotations}.{args.output_format}"
    output_path = out_dir / out_name

    meta = {
        "pdb_id": pdb_id,
        "pdb_path": str(pdb_path),
        "density_map": str(args.density_map),
        "num_rotations": args.num_rotations,
        "projection_axis": args.projection_axis,
        "bfactor_override": effective_bfactor_override,
        "atomic_radius": args.atomic_radius,
        "device": str(device),
        "lattice": lattice_meta,
    }

    if resolved_info is not None:
        meta["resolved_atoms_only"] = True
        meta.update(resolved_info)
    else:
        meta["resolved_atoms_only"] = False
    if alignment_info is not None:
        if args.aligned_pdb_out is not None:
            written_aligned_pdb = _write_transformed_pdb(
                input_pdb_path=pdb_path,
                output_pdb_path=Path(args.aligned_pdb_out),
                rotation=alignment_rotation,
                translation=alignment_translation,
            )
            alignment_info["aligned_pdb_path"] = str(written_aligned_pdb)
        meta["alignment"] = alignment_info

    _save_outputs(projections, rotations, meta, output_path, args.output_format)


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)

    pdb_paths = [Path(p) for p in args.pdb_files]
    pdb_cache_dir = Path(args.pdb_cache_dir) if args.pdb_cache_dir else Path(args.out_dir) / "pdbs"
    for pdb_id in args.pdb_ids:
        pdb_paths.append(_download_pdb_if_not_cached(pdb_id, pdb_cache_dir))

    sequences = None
    chains_to_read = None
    config_bfactor_override = None
    if args.config:
        sequences, chains_to_read, config_bfactor_override = _load_sequences_from_config(args.config)

    for pdb_path in pdb_paths:
        _generate_dataset_for_pdb(
            pdb_path,
            args,
            device,
            sequences=sequences,
            chains_to_read=chains_to_read,
            config_bfactor_override=config_bfactor_override,
        )


if __name__ == "__main__":
    main()
