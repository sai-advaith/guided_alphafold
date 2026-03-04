from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .abstract_loss_funciton import AbstractLossFunction
from ..protenix.metrics.rmsd import self_aligned_rmsd
from ..utils.cryoimage_renderer import (
    CryoImageDataset,
    CryoImageRenderer,
)
from ..utils.io import load_pdb_atom_locations_full, alignment_mask_by_chain


class CryoEM_Images_GuidanceLossFunction(AbstractLossFunction):
    """
    Guidance loss over CryoEM projections.

      1) Load ground-truth projections + rotations from .pt/.json files.
      2) Load reference PDB(s) for alignment.
      3) At each call, align x_0_hat to each reference conformation.
      4) Render projections for all GT rotations and compare to GT images.
    """

    def __init__(
        self,
        image_pt_files: list[str],
        image_json_files: list[str] | None,
        reference_pdbs: list[str],
        mask: torch.Tensor,
        sequences_dictionary: list[dict[str, Any]],
        chains_to_read: list[str] | None = None,
        device: str = "cpu",
        should_align_to_chains: list[int] | None = None,
        log_projection_every: int = 10,
        log_projection_pairs: int = 3,
        max_rotations_per_batch: int | None = None,
        use_resolved_atoms_only: bool = True,
        bfactor_override: float | None = None,
        supervised_assignment_by_index: bool = True,
    ):
        self.device = torch.device(device)
        self._projection_log_every = log_projection_every
        self._projection_log_pairs = log_projection_pairs
        self._projection_log_calls = 0
        self._projection_log_payload = None
        self._fast_renderers: list[CryoImageRenderer] = []
        self.max_rotations_per_batch = max_rotations_per_batch
        self.use_resolved_atoms_only = use_resolved_atoms_only
        self.bfactor_override = None if bfactor_override is None else float(bfactor_override)
        self.supervised_assignment_by_index = bool(supervised_assignment_by_index)

        if not image_pt_files:
            raise ValueError("image_pt_files must be a non-empty list.")
        if not reference_pdbs:
            raise ValueError("reference_pdbs must be a non-empty list.")
        if len(reference_pdbs) != len(image_pt_files):
            raise ValueError(
                "reference_pdbs and image_pt_files must have the same length "
                "(one per conformation)."
            )
        if image_json_files is not None and len(image_json_files) != len(image_pt_files):
            raise ValueError("image_json_files must match image_pt_files length.")
        if self.bfactor_override is not None and self.bfactor_override < 0:
            raise ValueError("bfactor_override must be non-negative.")

        self.AF3_to_pdb_mask = mask.to(self.device)

        # Build full sequences list (same as ESP/RMSD loss)
        full_sequences = [[d["sequence"]] * d["count"] for d in sequences_dictionary]
        full_sequences = [item for sublist in full_sequences for item in sublist]
        self.full_sequences = full_sequences
        self.sequences_dictionary = sequences_dictionary

        self.sequence_types = [
            sequence_type
            for dictionary in sequences_dictionary
            for sequence_type in [dictionary.get("sequence_type", "proteinChain")]
            * dictionary["count"]
        ]

        # Alignment mask (defaults to all chains)
        if should_align_to_chains is None:
            should_align_to_chains = list(range(len(full_sequences)))
        self.should_align_to_chains = should_align_to_chains
        self.align_to_chain_mask = alignment_mask_by_chain(
            full_sequences,
            chains_to_align=should_align_to_chains,
            sequence_types=self.sequence_types,
        ).to(self.device)

        # Load datasets and reference conformations
        self.dataset: CryoImageDataset = CryoImageDataset.from_paths(
            pt_paths=image_pt_files,
            json_paths=image_json_files,
            device=self.device,
        )

        self.reference_structures = []
        for pdb_path in reference_pdbs:
            load_result = load_pdb_atom_locations_full(
                pdb_file=pdb_path,
                full_sequences_dict=sequences_dictionary,
                chains_to_read=chains_to_read,
                return_elements=True,
                return_bfacs=True,
                return_mask=True,
                return_starting_indices=True,
            )
            coords, mask_tensor, b_factors, elements, starting_residue_indices = load_result
            if self.bfactor_override is not None:
                b_factors = torch.full_like(b_factors, self.bfactor_override)
            ref = {
                "coords": coords.to(self.device),
                "mask": mask_tensor.to(self.device),
                "bfactors": b_factors.to(self.device),
                "atomic_numbers": elements.to(self.device),
                "starting_residue_indices": starting_residue_indices,
            }
            ref["resolved_mask"] = ref["mask"].to(torch.bool)
            ref["render_mask"] = self._build_render_mask(ref)
            self.reference_structures.append(ref)

        self.last_loss_value = None
        self.last_loss_per_conformation: dict[int, float] = {}
        self.last_rmsd_per_conformation: dict[int, float] = {}
        self.last_cosine_similarity_per_conformation: dict[int, float] = {}
        self._last_projection_log_step = None
        self._setup_fast_solver()

    def _build_render_mask(self, reference: dict) -> torch.Tensor | None:
        base_mask = reference.get("resolved_mask") if self.use_resolved_atoms_only else None
        return base_mask

    def _projection_loss(self, rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return (rendered - gt).square().mean(dim=(1, 2)).mean()

    def _setup_fast_solver(self) -> None:
        if self.dataset.num_conformations <= 0 or not self.reference_structures:
            raise RuntimeError("Datasets and reference structures must be loaded before setting up the ESP solver.")
        if self.dataset.num_conformations != len(self.reference_structures):
            raise RuntimeError(
                "The number of image conformations must match the number of reference structures. "
                f"Got {self.dataset.num_conformations} and {len(self.reference_structures)}."
            )

        self._fast_renderers = []
        for dataset_idx, reference in enumerate(self.reference_structures):
            conformation = self.dataset.get_conformation(dataset_idx)
            coords = reference["coords"]
            atomic_numbers = reference["atomic_numbers"]
            bfactors = reference["bfactors"]
            resolved_mask = reference.get("render_mask")

            if resolved_mask is not None:
                coords, atomic_numbers, bfactors = CryoImageRenderer.apply_atom_mask(
                    coords, atomic_numbers, bfactors, resolved_mask
                )
            if atomic_numbers is None or bfactors is None:
                raise RuntimeError(
                    f"Reference structure {dataset_idx} is missing atomic_numbers or bfactors for rendering."
                )

            renderer = CryoImageRenderer.from_coords_and_atomic_numbers(
                coords=coords,
                atomic_numbers=atomic_numbers,
                bfactors=bfactors,
                lattice=conformation["lattice"],
                projection_axis=int(conformation["projection_axis"]),
                collapse_projection_axis=bool(conformation["collapse_projection_axis"]),
                device=self.device,
                use_checkpointing=False,
                use_autocast=False,
            )
            self._fast_renderers.append(renderer)

    def _render_all_rotations(
        self,
        conformation_index: int,
        coords: torch.Tensor,
        atomic_numbers: torch.Tensor,
        bfactors: torch.Tensor,
        rotations: torch.Tensor,
        resolved_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._fast_renderers:
            raise RuntimeError("Fast renderers were not initialized.")
        if not (0 <= conformation_index < len(self._fast_renderers)):
            raise IndexError(
                f"Invalid conformation_index={conformation_index}; "
                f"expected [0, {len(self._fast_renderers) - 1}]"
            )
        renderer = self._fast_renderers[conformation_index]
        if resolved_mask is not None:
            coords, atomic_numbers, bfactors = CryoImageRenderer.apply_atom_mask(
                coords, atomic_numbers, bfactors, resolved_mask
            )
        if atomic_numbers is None or bfactors is None:
            raise RuntimeError("Atomic numbers and bfactors are required for projection rendering.")
        return renderer.render_all_rotations_from_coords(
            coords=coords,
            atomic_numbers=atomic_numbers,
            bfactors=bfactors,
            rotations=rotations,
            max_rotations_per_batch=self.max_rotations_per_batch,
        )

    def _align_to_reference(self, x_0_hat: torch.Tensor, reference: dict):
        alignment_mask = self.AF3_to_pdb_mask & self.align_to_chain_mask
        _, aligned_x_0_hat, _, _ = self_aligned_rmsd(
            x_0_hat,
            reference["coords"].unsqueeze(0),
            alignment_mask,
        )
        return aligned_x_0_hat

    def _store_projection_log(
        self,
        rendered: torch.Tensor,
        gt: torch.Tensor,
        step: int | None,
    ) -> None:
        num_rots = int(rendered.shape[0])
        if num_rots <= 0:
            return
        k = min(self._projection_log_pairs, num_rots)
        idx = torch.linspace(0, num_rots - 1, steps=k, device=rendered.device).long()
        self._projection_log_payload = {
            "indices": idx.detach().cpu().tolist(),
            "gt": gt[idx].detach().float().cpu(),
            "pred": rendered[idx].detach().float().cpu(),
        }
        self._last_projection_log_step = step

    def __call__(self, x_0_hat: torch.Tensor, time, structures=None, i=None, step=None):
        # For multiple conformations, compute loss per dataset and average.
        losses = []
        self.last_loss_per_conformation = {}
        self.last_rmsd_per_conformation = {}
        self.last_cosine_similarity_per_conformation = {}
        do_log = False
        if self._projection_log_every and self._projection_log_every > 0:
            self._projection_log_calls += 1
            if step is not None:
                do_log = (int(step) % self._projection_log_every) == 0
            else:
                # Fallback for callers that do not provide step: log first call, then every N calls.
                do_log = ((self._projection_log_calls - 1) % self._projection_log_every) == 0

        if not self._fast_renderers:
            raise RuntimeError("Fast renderers are not initialized.")
        if self.supervised_assignment_by_index and x_0_hat.shape[0] < self.dataset.num_conformations:
            raise ValueError(
                "supervised_assignment_by_index requires batch size >= number of conformations. "
                f"Got batch={x_0_hat.shape[0]}, conformations={self.dataset.num_conformations}."
            )

        for dataset_idx, reference in enumerate(self.reference_structures):
            conformation = self.dataset.get_conformation(dataset_idx)
            assigned_x_0_hat = (
                x_0_hat[dataset_idx:dataset_idx + 1]
                if self.supervised_assignment_by_index
                else x_0_hat
            )
            aligned = self._align_to_reference(assigned_x_0_hat, reference)
            resolved_mask = reference.get("render_mask")

            gt = conformation["projections"]
            rendered = self._render_all_rotations(
                conformation_index=dataset_idx,
                coords=aligned,
                atomic_numbers=reference["atomic_numbers"],
                bfactors=reference["bfactors"],
                rotations=conformation["rotations"],
                resolved_mask=resolved_mask,
            )

            if do_log and dataset_idx == 0:
                self._store_projection_log(
                    rendered=rendered,
                    gt=gt,
                    step=step,
                )

            loss = self._projection_loss(rendered, gt)
            losses.append(loss)
            self.last_loss_per_conformation[dataset_idx] = float(loss.detach().item())

            rmsd_mask = self.AF3_to_pdb_mask
            if int(rmsd_mask.sum().item()) > 0:
                diff = aligned[:, rmsd_mask, :] - reference["coords"][rmsd_mask, :].unsqueeze(0)
            else:
                diff = aligned - reference["coords"].unsqueeze(0)
            self.last_rmsd_per_conformation[dataset_idx] = float(
                diff.square().sum(dim=-1).mean().sqrt().detach().item()
            )

            rendered_flat = rendered.reshape(rendered.shape[0], -1)
            gt_flat = gt.reshape(gt.shape[0], -1)
            cosine_similarity = F.cosine_similarity(rendered_flat, gt_flat, dim=1).mean()
            self.last_cosine_similarity_per_conformation[dataset_idx] = float(
                cosine_similarity.detach().item()
            )

        loss = torch.stack(losses).mean()
        self.last_loss_value = loss.detach().item()
        return loss, None, None

    def wandb_log(self, x_0_hat):
        log = {
            "loss": self.last_loss_value,  # keep for MultiLossFunction common_loss
            "cryoimage/loss": self.last_loss_value,
        }
        if self.last_rmsd_per_conformation:
            log["cryoimage/rmsd_mean"] = float(
                sum(self.last_rmsd_per_conformation.values()) / len(self.last_rmsd_per_conformation)
            )
        if self.last_cosine_similarity_per_conformation:
            log["cryoimage/cosine_similarity_mean"] = float(
                sum(self.last_cosine_similarity_per_conformation.values())
                / len(self.last_cosine_similarity_per_conformation)
            )
        for idx, value in self.last_loss_per_conformation.items():
            log[f"cryoimage/loss_conf_{idx}"] = float(value)
        for idx, value in self.last_rmsd_per_conformation.items():
            log[f"cryoimage/rmsd_conf_{idx}"] = float(value)
        for idx, value in self.last_cosine_similarity_per_conformation.items():
            log[f"cryoimage/cosine_similarity_conf_{idx}"] = float(value)
        log["cryoimage/supervised_assignment_by_index"] = float(self.supervised_assignment_by_index)
        payload = self._projection_log_payload
        if payload is not None:
            try:
                import wandb  # local import to avoid hard dependency
                gt_imgs = []
                pred_imgs = []
                diff_imgs = []
                for i, rot_idx in enumerate(payload["indices"]):
                    gt_img = payload["gt"][i]
                    pred_img = payload["pred"][i]
                    # Normalize to [0,1] for display
                    for_img = []
                    for img in (gt_img, pred_img):
                        img = img.float()
                        vmin = float(img.min().item())
                        vmax = float(img.max().item())
                        if vmax > vmin:
                            img = (img - vmin) / (vmax - vmin)
                        else:
                            img = torch.zeros_like(img)
                        for_img.append(img)
                    # Signed difference visualization (pred - gt), normalized to [0,1]
                    # with 0.5 corresponding to no difference.
                    diff = (pred_img.float() - gt_img.float())
                    max_abs = float(diff.abs().max().item())
                    if max_abs > 0.0:
                        diff = diff / (2.0 * max_abs) + 0.5
                    else:
                        diff = torch.full_like(diff, 0.5)
                    diff = diff.clamp(0.0, 1.0)
                    gt_imgs.append(wandb.Image(for_img[0].numpy(), caption=f"rot={rot_idx}"))
                    pred_imgs.append(wandb.Image(for_img[1].numpy(), caption=f"rot={rot_idx}"))
                    diff_imgs.append(wandb.Image(diff.numpy(), caption=f"rot={rot_idx}, pred-gt"))
                log["cryoimage/gt_projections"] = gt_imgs
                log["cryoimage/pred_projections"] = pred_imgs
                log["cryoimage/diff_projections"] = diff_imgs
                if self._last_projection_log_step is not None:
                    log["cryoimage/projection_log_step"] = self._last_projection_log_step
            except Exception:
                # If wandb isn't available or image logging fails, skip silently.
                pass
            self._projection_log_payload = None
        return log


class CryoEM_ImageAverage_GuidanceLossFunction(AbstractLossFunction):
    def __init__(self):
        raise NotImplementedError("This loss function is not implemented yet.")

    def __call__(self, x_0_hat, time, structures=None, i=None, step=None):
        raise NotImplementedError()

    def wandb_log(self, x_0_hat):
        raise NotImplementedError()
