from __future__ import annotations

from typing import Any

import torch

from .io import create_atom_mask


def is_range_pair(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    return all(isinstance(v, (int, float)) for v in value)


def normalize_residue_ranges_per_chain(
    residue_ranges_pdb: list | tuple | None,
) -> list | None:
    if residue_ranges_pdb is None:
        return None
    if not isinstance(residue_ranges_pdb, (list, tuple)):
        raise ValueError("residue_ranges_pdb must be a list.")
    if len(residue_ranges_pdb) == 0:
        return []
    if is_range_pair(residue_ranges_pdb[0]):
        # Single-chain shorthand: [[start, end], [start, end], ...]
        return [list(residue_ranges_pdb)]
    return list(residue_ranges_pdb)


def build_residue_subset_mask(
    *,
    full_sequences: list[str],
    sequence_types: list[str],
    starting_residue_indices: list[int] | None,
    residue_ranges_pdb: list | tuple | None,
) -> torch.Tensor | None:
    ranges_per_chain = normalize_residue_ranges_per_chain(residue_ranges_pdb)
    if ranges_per_chain is None:
        return None

    regions_per_sequence: list[list[int]] = []
    for chain_idx, sequence in enumerate(full_sequences):
        chain_ranges = ranges_per_chain[chain_idx] if chain_idx < len(ranges_per_chain) else None
        if chain_ranges is None:
            regions_per_sequence.append([])
            continue
        if is_range_pair(chain_ranges):
            chain_ranges = [chain_ranges]

        start_idx = 1
        if starting_residue_indices is not None and chain_idx < len(starting_residue_indices):
            start_idx = int(starting_residue_indices[chain_idx])

        seq_len = len(sequence)
        selected_residues = set()
        for range_pair in chain_ranges:
            if range_pair is None:
                continue
            if not is_range_pair(range_pair):
                raise ValueError(
                    "Each residue range must be [start_pdb_id, end_pdb_id]. "
                    f"Got: {range_pair!r}"
                )
            start_pdb, end_pdb = int(range_pair[0]), int(range_pair[1])
            start_seq = max(1, start_pdb - start_idx + 1)
            end_seq = min(seq_len, end_pdb - start_idx + 1)
            if start_seq <= end_seq:
                selected_residues.update(range(start_seq, end_seq + 1))
        regions_per_sequence.append(sorted(selected_residues))

    if not any(len(r) > 0 for r in regions_per_sequence):
        raise ValueError(
            "residue_ranges_pdb produced an empty residue selection. "
            "Check PDB numbering and ranges."
        )

    return create_atom_mask(
        full_sequences,
        regions_per_sequence,
        sequence_types=sequence_types,
    )
