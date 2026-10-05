#!/usr/bin/env python3
"""Lightweight unit tests for the NOE OR-group reduction (CPU-only, no model weights)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.losses.nmr_loss_function import NMRLossFunction

# Rows 0 and 1 form one ambiguous OR group (H-HB2 | H-HB3); row 2 is a group on its own.
UB = 2.507
LB = 1.907


def _loss() -> NMRLossFunction:
    # Bypass __init__: only the state used by the reduction is needed.
    loss = NMRLossFunction.__new__(NMRLossFunction)
    or_ids = torch.tensor([4.0, 4.0, 7.0])
    loss.single_chain_mask_np = np.ones(3, dtype=bool)
    loss.multi_chain_mask_np = np.zeros(3, dtype=bool)
    loss.within_chain_unique_or, loss.within_chain_inverse_or_indices = torch.unique(
        or_ids, return_inverse=True
    )
    loss.upper_bound = torch.tensor([UB, UB, 3.0])
    loss.lower_bound = torch.tensor([LB, LB, 1.8])
    return loss


def _reduce(dist, mask=None):
    mask = torch.ones(3, dtype=torch.bool) if mask is None else mask
    return _loss()._compute_bound_losses(dist, mask, within_chain=True)


def test_members_cannot_split_bounds() -> None:
    # Member 0 is too close and member 1 is too far: no member lies inside [lb, ub].
    dist = torch.tensor([[1.2, 3.5, 2.5]], requires_grad=True)
    ub_val, lb_val, ub_or, lb_or = _reduce(dist)
    expected = min(LB - 1.2, 3.5 - UB)
    assert abs((ub_or + lb_or)[0, 0].item() - expected) < 1e-5
    assert ub_or[0, 0].item() == 0.0
    assert abs(lb_or[0, 0].item() - expected) < 1e-5
    assert (ub_or + lb_or)[0, 1].item() == 0.0

    (ub_val + lb_val).backward()
    # Only the selected member (the one too close) is pushed, and it is pushed outward.
    assert dist.grad[0, 0] < 0
    assert dist.grad[0, 1] == 0 and dist.grad[0, 2] == 0


def test_satisfied_member_gives_zero_loss() -> None:
    _, _, ub_or, lb_or = _reduce(torch.tensor([[2.2, 3.5, 2.5]]))
    assert (ub_or + lb_or).abs().sum().item() == 0.0


def test_tie_keeps_total() -> None:
    _, _, ub_or, lb_or = _reduce(torch.tensor([[LB - 0.5, UB + 0.5, 2.5]]))
    assert abs((ub_or + lb_or)[0, 0].item() - 0.5) < 1e-5
    assert ub_or[0, 0].item() == 0.0


def test_unresolved_member_is_skipped() -> None:
    mask = torch.tensor([True, False, True])
    _, _, ub_or, lb_or = _reduce(torch.tensor([[1.2, 2.5]]), mask)
    assert abs(lb_or[0, 0].item() - (LB - 1.2)) < 1e-5


def test_batched_rows_reduce_independently() -> None:
    _, _, ub_or, lb_or = _reduce(torch.tensor([[1.2, 3.5, 2.5], [2.2, 3.5, 2.5]]))
    total = ub_or + lb_or
    assert total[0, 0].item() > 0 and total[1, 0].item() == 0.0


if __name__ == "__main__":
    test_members_cannot_split_bounds()
    test_satisfied_member_gives_zero_loss()
    test_tie_keeps_total()
    test_unresolved_member_is_skipped()
    test_batched_rows_reduce_independently()
    print("TEST_NMR_OR_REDUCTION_OK")
