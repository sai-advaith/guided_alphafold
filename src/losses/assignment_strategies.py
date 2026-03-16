"""
Assignment strategies for mapping projections to structures in multi-conformation cryo-EM guidance.

Each strategy takes a loss matrix of shape (num_samples, num_structures) and returns
per-sample losses (with gradients) and hard assignments (for diagnostics).

Available strategies:
    - hard:     Pure argmin (winner-take-all). Fast but can oscillate.
    - softmax:  Soft assignment via softmax with temperature annealing. Smooth gradients.
    - sinkhorn: Optimal-transport balanced assignment via Sinkhorn iterations.
    - sticky:   Hard assignment with a switching cost that penalises flipping.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F


def build_assignment_strategy(name: str, **kwargs: Any) -> AssignmentStrategy:
    """Factory: create a strategy by name with keyword arguments."""
    registry: dict[str, type[AssignmentStrategy]] = {
        "hard": HardAssignment,
        "softmax": SoftmaxAssignment,
        "sinkhorn": SinkhornAssignment,
        "sticky": StickyAssignment,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown assignment strategy '{name}'. Available: {sorted(registry)}."
        )
    return registry[name](**kwargs)


class AssignmentStrategy(ABC):
    """Base class for assignment strategies."""

    @abstractmethod
    def assign(
        self,
        loss_matrix: torch.Tensor,
        step: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            loss_matrix: (num_samples, num_structures) – per-sample, per-structure losses.
            step: optional global step for annealing schedules.

        Returns:
            selected_losses: (num_samples,) – loss values to backpropagate.
            hard_assignments: (num_samples,) – integer structure indices for diagnostics.
        """

    def extra_wandb_log(self) -> dict[str, float]:
        """Return strategy-specific scalars to log each step."""
        return {}


# ---------------------------------------------------------------------------
# Hard (current default)
# ---------------------------------------------------------------------------

class HardAssignment(AssignmentStrategy):
    """Pure argmin assignment – each projection goes to its lowest-loss structure."""

    def assign(
        self,
        loss_matrix: torch.Tensor,
        step: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assignments = loss_matrix.argmin(dim=1)
        selected_losses = loss_matrix.gather(1, assignments.unsqueeze(1)).squeeze(1)
        return selected_losses, assignments


# ---------------------------------------------------------------------------
# Softmax with temperature annealing
# ---------------------------------------------------------------------------

class SoftmaxAssignment(AssignmentStrategy):
    """
    Soft assignment via softmax over negative losses.

    The per-sample loss is the weighted combination:  sum_j w_j * L_j  where
    w = softmax(-L / temperature).

    Temperature is annealed:  T(step) = max(T_min, T_init * decay^step).
    At high T every structure contributes equally; as T → 0 it converges to hard argmin.
    """

    def __init__(
        self,
        temperature_init: float = 1.0,
        temperature_min: float = 0.01,
        temperature_decay: float = 1.0,
    ):
        if temperature_init <= 0:
            raise ValueError("temperature_init must be positive.")
        if temperature_min <= 0:
            raise ValueError("temperature_min must be positive.")
        if not 0 < temperature_decay <= 1:
            raise ValueError("temperature_decay must be in (0, 1].")
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        self._last_temperature: float | None = None

    def _temperature(self, step: int | None) -> float:
        if step is None or self.temperature_decay >= 1.0:
            t = self.temperature_init
        else:
            t = self.temperature_init * (self.temperature_decay ** step)
        t = max(t, self.temperature_min)
        self._last_temperature = t
        return t

    def assign(
        self,
        loss_matrix: torch.Tensor,
        step: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = self._temperature(step)
        weights = F.softmax(-loss_matrix / T, dim=1)                     # (N, S)
        selected_losses = (weights * loss_matrix).sum(dim=1)             # (N,)
        hard_assignments = loss_matrix.detach().argmin(dim=1)            # (N,)
        return selected_losses, hard_assignments

    def extra_wandb_log(self) -> dict[str, float]:
        log: dict[str, float] = {}
        if self._last_temperature is not None:
            log["cryoimage/assignment_temperature"] = self._last_temperature
        return log


# ---------------------------------------------------------------------------
# Sinkhorn (balanced optimal-transport)
# ---------------------------------------------------------------------------

class SinkhornAssignment(AssignmentStrategy):
    """
    Balanced assignment via the Sinkhorn algorithm on the loss matrix.

    Solves an entropy-regularised optimal transport problem so that each
    structure receives (approximately) an equal share of projections.

    Temperature controls the entropy regularisation (lower = harder assignment).
    """

    def __init__(
        self,
        temperature_init: float = 1.0,
        temperature_min: float = 0.01,
        temperature_decay: float = 1.0,
        num_iters: int = 10,
    ):
        if temperature_init <= 0:
            raise ValueError("temperature_init must be positive.")
        if temperature_min <= 0:
            raise ValueError("temperature_min must be positive.")
        if not 0 < temperature_decay <= 1:
            raise ValueError("temperature_decay must be in (0, 1].")
        if num_iters < 1:
            raise ValueError("num_iters must be >= 1.")
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        self.num_iters = num_iters
        self._last_temperature: float | None = None

    def _temperature(self, step: int | None) -> float:
        if step is None or self.temperature_decay >= 1.0:
            t = self.temperature_init
        else:
            t = self.temperature_init * (self.temperature_decay ** step)
        t = max(t, self.temperature_min)
        self._last_temperature = t
        return t

    @staticmethod
    def _sinkhorn(
        log_K: torch.Tensor,
        target_row: torch.Tensor,
        target_col: torch.Tensor,
        num_iters: int,
    ) -> torch.Tensor:
        """
        Sinkhorn iterations in log-space for numerical stability.

        Args:
            log_K:      (N, S) - log of the Gibbs kernel, i.e. -cost / temperature.
            target_row: (N,)   - desired row marginals (sums to 1).
            target_col: (S,)   - desired column marginals (sums to 1).
            num_iters:  number of alternating projection iterations.

        Returns:
            P: (N, S) - transport plan (rows sum to target_row, cols to target_col).
        """
        u = torch.zeros_like(log_K[:, 0])   # (N,)
        v = torch.zeros_like(log_K[0, :])   # (S,)

        log_target_row = target_row.log()
        log_target_col = target_col.log()

        for _ in range(num_iters):
            # Row normalisation
            log_sum_cols = torch.logsumexp(log_K + v.unsqueeze(0), dim=1)  # (N,)
            u = log_target_row - log_sum_cols
            # Column normalisation
            log_sum_rows = torch.logsumexp(log_K + u.unsqueeze(1), dim=0)  # (S,)
            v = log_target_col - log_sum_rows

        log_P = log_K + u.unsqueeze(1) + v.unsqueeze(0)
        return log_P.exp()

    def assign(
        self,
        loss_matrix: torch.Tensor,
        step: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N, S = loss_matrix.shape
        T = self._temperature(step)

        log_K = -loss_matrix / T   # (N, S)

        # Uniform marginals: each sample contributes 1/N, each structure gets 1/S
        target_row = torch.full((N,), 1.0 / N, device=loss_matrix.device, dtype=loss_matrix.dtype)
        target_col = torch.full((S,), 1.0 / S, device=loss_matrix.device, dtype=loss_matrix.dtype)

        # Detach the Sinkhorn plan computation (assignment decision)
        # but apply weights to the live loss_matrix (gradient flows through losses)
        with torch.no_grad():
            P = self._sinkhorn(log_K.detach(), target_row, target_col, self.num_iters)

        # Normalise rows so each sample's weights sum to 1
        row_sums = P.sum(dim=1, keepdim=True).clamp_min(1e-12)
        weights = P / row_sums                                          # (N, S)

        selected_losses = (weights * loss_matrix).sum(dim=1)            # (N,)
        hard_assignments = loss_matrix.detach().argmin(dim=1)           # (N,)
        return selected_losses, hard_assignments

    def extra_wandb_log(self) -> dict[str, float]:
        log: dict[str, float] = {}
        if self._last_temperature is not None:
            log["cryoimage/assignment_temperature"] = self._last_temperature
        return log


# ---------------------------------------------------------------------------
# Sticky (hard assignment with switching cost)
# ---------------------------------------------------------------------------

class StickyAssignment(AssignmentStrategy):
    """
    Hard assignment with a switching cost that discourages rapid reassignment.

    Each sample remembers its previous assignment.
    """

    def __init__(
        self,
        stickiness: float = 0.1,
        stickiness_decay: float = 1.0,
        stickiness_min: float = 0.0,
    ):
        if stickiness < 0:
            raise ValueError("stickiness must be non-negative.")
        if not 0 < stickiness_decay <= 1:
            raise ValueError("stickiness_decay must be in (0, 1].")
        if stickiness_min < 0:
            raise ValueError("stickiness_min must be non-negative.")
        self.stickiness_init = stickiness
        self.stickiness_decay = stickiness_decay
        self.stickiness_min = stickiness_min
        self._prev_assignments: torch.Tensor | None = None
        self._last_stickiness: float | None = None

    def _current_stickiness(self, step: int | None) -> float:
        if step is None or self.stickiness_decay >= 1.0:
            s = self.stickiness_init
        else:
            s = self.stickiness_init * (self.stickiness_decay ** step)
        s = max(s, self.stickiness_min)
        self._last_stickiness = s
        return s

    def assign(
        self,
        loss_matrix: torch.Tensor,
        step: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N, S = loss_matrix.shape
        stickiness = self._current_stickiness(step)

        if self._prev_assignments is not None and self._prev_assignments.shape[0] == N:
            # Add switching cost: penalise all structures except the previous one
            penalty = torch.full_like(loss_matrix, stickiness)
            penalty.scatter_(1, self._prev_assignments.unsqueeze(1), 0.0)
            decision_matrix = loss_matrix.detach() + penalty
        else:
            decision_matrix = loss_matrix.detach()

        assignments = decision_matrix.argmin(dim=1)
        self._prev_assignments = assignments.detach().clone()

        selected_losses = loss_matrix.gather(1, assignments.unsqueeze(1)).squeeze(1)
        return selected_losses, assignments

    def extra_wandb_log(self) -> dict[str, float]:
        log: dict[str, float] = {}
        if self._last_stickiness is not None:
            log["cryoimage/assignment_stickiness"] = self._last_stickiness
        return log
