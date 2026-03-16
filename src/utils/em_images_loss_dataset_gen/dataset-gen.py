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

from src.utils.io import (
    load_pdb_atom_locations_full,
    ATOM_NAME_TO_ELEMENT,
    SEQUENCE_TYPE_TO_ATOM_DICTIONARY,
    SEQUENCE_TYPE_TO_RESIDUE_KIND,
)
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
        "--grid-size",
        type=int,
        default=None,
        help=(
            "Override the number of voxels per axis (isotropic). "
            "If not set, the value is read from the density map. "
            "When grid-size * pixel-size differs from the map, the FOV adapts "
            "while keeping the map center."
        ),
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help=(
            "Override the voxel size in Angstroms. "
            "If not set, the value is read from the density map. "
            "When grid-size * pixel-size differs from the map, the FOV adapts "
            "while keeping the map center."
        ),
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
        "--pdb-out",
        type=str,
        default=None,
        help=(
            "Optional output path for saving the input PDB restricted to the config-defined "
            "chains. When --align-to-reference-pdb is also set, the rigid alignment transform "
            "is applied before saving. Requires --config."
        ),
    )
    args = parser.parse_args()
    if not args.pdb_files and not args.pdb_ids:
        parser.error("At least one --pdb-file or --pdb-id must be provided.")
    if args.align_to_reference_pdb is not None and args.config is None:
        parser.error("--align-to-reference-pdb requires --config.")
    if args.pdb_out is not None and args.config is None:
        parser.error("--pdb-out requires --config.")
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
    return atom_stack, int(mask.sum().item()), int(mask.numel()), payload


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
        aligned_coords_full[0].detach().cpu(),
        moving,
    )






def _save_pdb(structure, full_sequences, sequence_types, write_file_name, bfactors=None, atom_mask=None, chain_names=None, starting_residue_indices=None):
    """
    Save structure with proper chain names matching original PDB chain names.
    Copied from non_diffusion_model_manager.save_structure_full to avoid heavy import chain.

    Args:
        structure: coords tensor of shape (N_total, 3)
        full_sequences: list of one-letter sequences per chain
        sequence_types: list of sequence types per chain (proteinChain, rnaSequence, dnaSequence)
        write_file_name: output PDB file path
        bfactors: optional tensor/array of B-factors (N_total,) or (ensemble, N_total)
        atom_mask: boolean mask (N_total,) indicating resolved atoms
        chain_names: optional list of chain names
        starting_residue_indices: optional list of starting residue indices (1-indexed) per chain
    """
    gemmi_structure = gemmi.Structure()
    model = gemmi.Model("1")

    chains = []
    if atom_mask is None:
        atom_mask = torch.ones(structure.shape[0], dtype=torch.bool)

    for i in range(len(full_sequences)):
        if chain_names is not None and i < len(chain_names) and chain_names[i] is not None:
            chain_name = chain_names[i]
        else:
            chain_name = chr(ord("A") + i)
        chains.append(gemmi.Chain(chain_name))

    iter_index = 0

    structure = structure.cpu().detach().numpy()
    if bfactors is not None:
        if isinstance(bfactors, torch.Tensor):
            bfactors = bfactors.cpu().detach().numpy()
        if bfactors.ndim > 1:
            bfactors = bfactors[0]

    for chain_i, (chain, sequence_type) in enumerate(zip(chains, sequence_types)):
        sequence = full_sequences[chain_i]
        if starting_residue_indices is not None and chain_i < len(starting_residue_indices):
            start_residue_num = starting_residue_indices[chain_i]
        else:
            start_residue_num = 1

        for i, res_name_one_letter in enumerate(sequence):
            res = gemmi.Residue()
            res.name = gemmi.expand_one_letter(res_name_one_letter, SEQUENCE_TYPE_TO_RESIDUE_KIND[sequence_type])
            res.seqid = gemmi.SeqId(start_residue_num + i, " ")
            residue_has_atoms = False

            # Every first residue of a dna or rna chain should have the OP3 atom
            if sequence_type in ["rnaSequence", "dnaSequence"] and i == 0:
                if atom_mask[iter_index] == True:
                    atom = gemmi.Atom()
                    atom.name = "OP3"
                    atom.element = gemmi.Element("O")
                    atom.pos = gemmi.Position(*structure[iter_index])
                    if bfactors is not None:
                        atom.b_iso = bfactors[iter_index]
                    res.add_atom(atom)
                    residue_has_atoms = True
                iter_index += 1

            for atom_name in SEQUENCE_TYPE_TO_ATOM_DICTIONARY[sequence_type][res.name]:
                if atom_mask[iter_index] == True:
                    atom = gemmi.Atom()
                    atom.name = atom_name
                    atom.element = gemmi.Element(ATOM_NAME_TO_ELEMENT[atom_name])
                    atom.pos = gemmi.Position(*structure[iter_index])
                    if bfactors is not None:
                        atom.b_iso = bfactors[iter_index]
                    res.add_atom(atom)
                    residue_has_atoms = True
                iter_index += 1

            # Handle OXT atom for the last residue of each chain
            if i == len(sequence) - 1 and sequence_type == "proteinChain":
                if iter_index < len(atom_mask):
                    if atom_mask[iter_index] == True:
                        atom = gemmi.Atom()
                        atom.name = "OXT"
                        atom.element = gemmi.Element("O")
                        atom.pos = gemmi.Position(*structure[iter_index])
                        if bfactors is not None:
                            atom.b_iso = bfactors[iter_index]
                        res.add_atom(atom)
                        residue_has_atoms = True
                    iter_index += 1

            if residue_has_atoms:
                chain.add_residue(res)

        model.add_chain(chain)

    gemmi_structure.add_model(model)

    Path(write_file_name).parent.mkdir(parents=True, exist_ok=True)
    gemmi_structure.write_pdb(str(write_file_name))


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
    pdb_out_coords = None
    pdb_out_payload = None
    if sequences is not None:
        if args.align_to_reference_pdb is not None:
            atom_stack, alignment_info, alignment_rotation, alignment_translation, pdb_out_coords, pdb_out_payload = _align_atom_stack_from_config_to_reference(
                pdb_path=pdb_path,
                reference_pdb_path=Path(args.align_to_reference_pdb),
                sequences=sequences,
                chains_to_read=chains_to_read,
                device=device,
            )
            n_resolved = int(alignment_info["moving_resolved_atoms"])
            n_total = int(alignment_info["moving_total_atoms"])
        else:
            atom_stack, n_resolved, n_total, pdb_out_payload = _atom_stack_from_config(
                pdb_path, sequences, chains_to_read, device
            )
            pdb_out_coords = pdb_out_payload["coords_full"]
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
        grid_size_override=args.grid_size,
        pixel_size_override=args.pixel_size,
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
    suffix_parts = [f"{pdb_id}_esp_projections_{args.num_rotations}"]
    if args.grid_size is not None:
        suffix_parts.append(f"D{args.grid_size}")
    if args.pixel_size is not None:
        suffix_parts.append(f"ps{args.pixel_size}")
    out_name = f"{'_'.join(suffix_parts)}.{args.output_format}"
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
    if args.pdb_out is not None and pdb_out_payload is not None:
        full_sequences_strs = [s for d in sequences for s in [d["sequence"]] * d["count"]]
        sequence_types = [t for d in sequences for t in [d.get("sequence_type", "proteinChain")] * d["count"]]
        _save_pdb(
            structure=pdb_out_coords,
            full_sequences=full_sequences_strs,
            sequence_types=sequence_types,
            write_file_name=args.pdb_out,
            bfactors=pdb_out_payload["bfactors_full"],
            atom_mask=pdb_out_payload["mask"],
            chain_names=chains_to_read,
            starting_residue_indices=pdb_out_payload["starting_residue_indices"],
        )
        meta["pdb_out"] = args.pdb_out
    if alignment_info is not None:
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
