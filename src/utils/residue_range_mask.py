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
