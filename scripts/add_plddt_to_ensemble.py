#!/usr/bin/env python3
"""
Write per-atom pLDDT into the B-factor column of a finished ensemble.

    python3 scripts/add_plddt_to_ensemble.py \
        --config generated_configurations/<id>_nmr_guided.yaml \
        --ensemble <output_dir>/<id>_nmr_guided/diffusion_process/<id>_ensemble.pdb

Every model in the ensemble is scored independently. Heavy atoms receive their predicted
pLDDT (0-100); hydrogens are set to 0, since the confidence head does not predict them.

Note on interpretation: the confidence head scores coordinates in the model's own atom
space, which is heavy-atom only. Hydrogens added during metrics and any movement from
AMBER relaxation are therefore not seen by it. The value written is pLDDT for the
relaxed heavy-atom coordinates as supplied here, evaluated against the same sequence,
MSA and trunk embeddings the run used.
"""

from __future__ import annotations

import argparse
import os
import sys

import gemmi
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.io import load_config  # noqa: E402
from src.utils.non_diffusion_model_manager import ProtenixModelManager  # noqa: E402


def is_hydrogen(atom) -> bool:
    try:
        if atom.element == gemmi.Element("H"):
            return True
    except Exception:
        pass
    return str(atom.name).lstrip("0123456789").startswith("H")


def build_model_manager(config, device):
    """Reconstruct the model exactly as the run did, reusing its cached MSA."""
    protein = config.protein
    manager = config.model_manager
    return ProtenixModelManager(
        sequences_dictionary=protein.sequences,
        pdb_id=protein.pdb_id,
        assembly_identifier=protein.assembly_identifier,
        chains_to_read=protein.chains_to_use,
        ROI_residues=protein.residue_range if protein.residue_range is not None else None,
        should_align_to_chains=protein.should_align_to_chains,
        reference_pdb=protein.reference_pdb,
        pdb_contains_missing_atoms=protein.contains_missing_atoms,
        N_cycle=manager.N_cycle,
        chunk_size=manager.chunk_size,
        diffusion_N=manager.diffusion_N,
        gamma0=manager.gamma0,
        gamma_min=manager.gamma_min,
        noise_scale_lambda=manager.noise_scale_lambda,
        step_scale_eta=manager.step_scale_eta,
        dtype="fp32",
        use_deepspeed_evo_attention=False,
        msa_save_dir=os.path.join(manager.msa_save_dir, protein.pdb_id),
        msa_embedding_cache_dir=manager.msa_embedding_cache_dir,
        model_checkpoint_path=manager.model_checkpoint_path,
        dump_dir=manager.dump_dir,
        use_msa=manager.use_msa,
        batch_size=1,
        device=device,
        pairformer_mixed_precision=manager.pairformer_mixed_precision,
    )


def model_atom_keys(atom_array, residue_offset):
    """(chain, author_res_id, atom_name) for every atom, in the model's own order."""
    chain_ids = np.asarray(atom_array.chain_id)
    res_ids = np.asarray(atom_array.res_id)
    names = np.asarray(atom_array.atom_name)
    keys = []
    for chain_id, res_id, name in zip(chain_ids, res_ids, names):
        label = str(chain_id)
        keys.append((label[0] if label else label, int(res_id) + residue_offset, str(name)))
    return keys


def coords_in_model_order(structure_model, keys):
    """
    Pull coordinates out of a PDB model, ordered to match the model's atom array.

    Returns (coords, missing) where missing lists keys absent from the PDB.
    """
    lookup = {}
    for chain in structure_model:
        label = chain.name[0] if chain.name else chain.name
        for residue in chain:
            for atom in residue:
                lookup[(label, int(residue.seqid.num), str(atom.name))] = atom.pos

    coords, missing = np.zeros((len(keys), 3), dtype=np.float32), []
    for index, key in enumerate(keys):
        position = lookup.get(key)
        if position is None:
            missing.append(key)
            continue
        coords[index] = (position.x, position.y, position.z)
    return coords, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Generated config YAML from the run")
    parser.add_argument("--ensemble", required=True, help="Multi-model ensemble PDB to annotate")
    parser.add_argument("--output", default=None, help="Output PDB (default: <ensemble>_plddt.pdb)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Models scored per forward pass. Lower this if the confidence head runs out of memory.",
    )
    parser.add_argument("--csv", default=None, help="Optional per-model, per-residue pLDDT table")
    args = parser.parse_args()

    config = load_config(args.config)
    residue_offset = int(getattr(config.protein, "start_residue_from", 1) or 1) - 1

    structure = gemmi.read_pdb(args.ensemble)
    n_models = len(structure)
    if n_models == 0:
        raise ValueError(f"{args.ensemble} contains no models.")
    print(f"{args.ensemble}: {n_models} model(s)")

    print("Rebuilding the model (reuses the run's cached MSA)...")
    model_manager = build_model_manager(config, args.device)
    atom_array = model_manager.atom_array
    keys = model_atom_keys(atom_array, residue_offset)
    print(f"model atom space: {len(keys)} heavy atoms; author numbering offset +{residue_offset}")

    # Gather coordinates for every model, in model-atom order.
    all_coords, reported_missing = [], False
    for model_index in range(n_models):
        coords, missing = coords_in_model_order(structure[model_index], keys)
        if missing:
            if not reported_missing:
                reported_missing = True
                print(
                    f"WARNING: {len(missing)}/{len(keys)} model atoms not found in ensemble model "
                    f"{model_index + 1} (e.g. {missing[:3]}). Their pLDDT will be unreliable. "
                    f"A large count usually means a residue-numbering mismatch: this ensemble "
                    f"should use the same --start_residue_from as the run that produced it."
                )
            if len(missing) == len(keys):
                raise ValueError(
                    "No model atoms could be matched to the ensemble at all. Check that "
                    "--config and --ensemble come from the same run."
                )
        all_coords.append(coords)

    coordinates = torch.tensor(np.stack(all_coords), dtype=torch.float32, device=args.device)

    print(f"Scoring {n_models} model(s) in batches of {args.batch_size}...")
    scores = []
    with torch.no_grad():
        for start in range(0, n_models, args.batch_size):
            batch = coordinates[start : start + args.batch_size]
            plddt = model_manager.get_confidance_scores(batch)
            scores.append(plddt.detach().float().cpu())
    plddt_per_model = torch.cat(scores, dim=0).numpy()
    # Confidence head may return a trailing singleton or extra leading dims; flatten to (M, N).
    plddt_per_model = plddt_per_model.reshape(n_models, -1)
    if plddt_per_model.shape[1] != len(keys):
        raise ValueError(
            f"pLDDT has {plddt_per_model.shape[1]} values per model but the model atom space "
            f"has {len(keys)} atoms; cannot map them onto the structure."
        )

    # Write pLDDT into B-factors: heavy atoms get their score, hydrogens get 0.
    key_to_index = {key: index for index, key in enumerate(keys)}
    for model_index in range(n_models):
        values = plddt_per_model[model_index]
        assigned = hydrogens = unmatched = 0
        for chain in structure[model_index]:
            label = chain.name[0] if chain.name else chain.name
            for residue in chain:
                for atom in residue:
                    if is_hydrogen(atom):
                        atom.b_iso = 0.0
                        hydrogens += 1
                        continue
                    index = key_to_index.get((label, int(residue.seqid.num), str(atom.name)))
                    if index is None:
                        atom.b_iso = 0.0
                        unmatched += 1
                        continue
                    atom.b_iso = float(values[index])
                    assigned += 1
        mean_plddt = float(np.mean([values[key_to_index[k]] for k in keys])) if keys else float("nan")
        print(
            f"  model {model_index + 1:>3}: mean pLDDT {mean_plddt:6.2f}  "
            f"heavy={assigned} hydrogens_zeroed={hydrogens} unmatched={unmatched}"
        )

    output_path = args.output or args.ensemble.replace(".pdb", "_plddt.pdb")
    structure.write_pdb(output_path)
    print(f"Wrote {output_path}")

    if args.csv:
        import csv as csv_module

        with open(args.csv, "w", newline="") as handle:
            writer = csv_module.writer(handle)
            writer.writerow(["model", "chain", "residue", "mean_plddt"])
            for model_index in range(n_models):
                values = plddt_per_model[model_index]
                per_residue = {}
                for key, index in key_to_index.items():
                    per_residue.setdefault((key[0], key[1]), []).append(values[index])
                for (chain_label, res_id), residue_values in sorted(per_residue.items()):
                    writer.writerow([model_index + 1, chain_label, res_id, f"{np.mean(residue_values):.2f}"])
        print(f"Wrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
