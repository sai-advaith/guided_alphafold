"""
Multi-reference RMSD loss for multi-conformation guidance.

Wraps one RMSDLossFunction per conformation so that each predicted structure
is compared to its own reference PDB.  Works seamlessly with cryoimage
(multiple reference_pdbs) and with cryoesp (single reference_pdb).
"""

from __future__ import annotations

from typing import Any

import torch

from .abstract_loss_funciton import AbstractLossFunction
from .rmsd_loss_function import RMSDLossFunction


class MultiReferenceRMSDLossFunction(AbstractLossFunction):
    """RMSD loss that supports one reference PDB per conformation.

    For each structure index *i* in ``x_0_hat[i]``, a dedicated
    :class:`RMSDLossFunction` computes the RMSD against
    ``reference_pdbs[i]``.  The final loss is the mean over all
    conformations.

    When only a single reference PDB is supplied this degenerates to
    the plain ``RMSDLossFunction`` behaviour (all batch elements
    are compared to the same reference).
    """

    def __init__(
        self,
        reference_pdbs: list[str],
        mask: torch.Tensor,
        sequences_dictionary: list[dict[str, Any]],
        chains_to_read: list[str] | None = None,
        rmsd_loss_sequence_indices: list[int] | None = None,
        device: str = "cpu",
        should_align_to_chains: list[int] | None = None,
        frozen_atoms_dict: dict | None = None,
    ):
        if not reference_pdbs:
            raise ValueError("reference_pdbs must be a non-empty list.")

        self._per_conf: list[RMSDLossFunction] = []
        for pdb_path in reference_pdbs:
            self._per_conf.append(
                RMSDLossFunction(
                    reference_pdb=pdb_path,
                    mask=mask,
                    sequences_dictionary=sequences_dictionary,
                    chains_to_read=chains_to_read,
                    rmsd_loss_sequence_indices=rmsd_loss_sequence_indices,
                    device=device,
                    should_align_to_chains=should_align_to_chains,
                    frozen_atoms_dict=frozen_atoms_dict,
                )
            )

        self.last_rmsd_per_conformation: dict[int, float] = {}
        self.last_rmsd_loss_value: float | None = None

    @property
    def num_conformations(self) -> int:
        return len(self._per_conf)

    # ------------------------------------------------------------------
    # pre / post optimisation – delegate to each sub-loss
    # ------------------------------------------------------------------

    def pre_optimization_step(self, x_0_hat: torch.Tensor, i=None, step=None):
        # For multi-conformation x_0_hat has shape [num_structures, N, 3].
        # Each sub-loss only sees its own slice, but pre_optimization_step
        # is called on the full tensor for mask extension bookkeeping.
        for loss_fn in self._per_conf:
            loss_fn.pre_optimization_step(x_0_hat, i=i, step=step)
        return x_0_hat

    def post_optimization_step(self, x_0_hat: torch.Tensor):
        for loss_fn in self._per_conf:
            loss_fn.post_optimization_step(x_0_hat)
        return x_0_hat

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(self, x_0_hat, time, structures=None, i=None, step=None):
        num_structures = x_0_hat.shape[0]

        if num_structures != self.num_conformations and self.num_conformations == 1:
            # Single reference – behave like plain RMSDLossFunction
            loss, _, _ = self._per_conf[0](x_0_hat, time, structures=structures, i=i, step=step)
            self.last_rmsd_loss_value = float(loss.detach().item())
            self.last_rmsd_per_conformation = {0: self.last_rmsd_loss_value}
            return loss, None, None

        if num_structures != self.num_conformations:
            raise ValueError(
                f"x_0_hat has {num_structures} structures but "
                f"{self.num_conformations} reference PDBs were provided."
            )

        total_loss = x_0_hat.new_tensor(0.0)
        self.last_rmsd_per_conformation = {}

        for idx, loss_fn in enumerate(self._per_conf):
            single = x_0_hat[idx : idx + 1]  # [1, N, 3]
            loss_i, _, _ = loss_fn(single, time, structures=structures, i=i, step=step)
            total_loss = total_loss + loss_i
            self.last_rmsd_per_conformation[idx] = float(loss_i.detach().item())

        mean_loss = total_loss / self.num_conformations
        self.last_rmsd_loss_value = float(mean_loss.detach().item())
        return mean_loss, None, None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def wandb_log(self, x_0_hat):
        log: dict[str, Any] = {"rmsd_loss": self.last_rmsd_loss_value}
        for idx, value in self.last_rmsd_per_conformation.items():
            log[f"rmsd_loss/conf_{idx}"] = value
        return log
