from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader

from .abstract_loss_funciton import AbstractLossFunction
from .assignment_strategies import AssignmentStrategy, HardAssignment
from ..protenix.metrics.rmsd import self_aligned_rmsd
from cryoforward.ctf import CTFParams

from ..utils.cryoimage_renderer import (
    CryoImageDataset,
    CryoImageRenderer,
)
from ..utils.io import load_pdb_atom_locations_full, alignment_mask_by_chain


class CryoEM_Images_GuidanceLossFunction(AbstractLossFunction):
    """
    Guidance loss over CryoEM projections.
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
        supervised_assignment_by_index: bool = False,
        projection_batch_size: int | None = None,
        shuffle_projection_samples: bool = True,
        assignment_strategy: AssignmentStrategy | None = None,
        ctf_params: CTFParams | None = None,
    ):
        self.device = torch.device(device)
        self._projection_log_every = log_projection_every
        self._projection_log_pairs = log_projection_pairs
        self._projection_log_calls = 0
        self._projection_log_payload = None
        self._last_projection_log_step = None
        self._fast_renderers: list[list[CryoImageRenderer]] = []
        self.max_rotations_per_batch = max_rotations_per_batch
        self.use_resolved_atoms_only = use_resolved_atoms_only
        self.bfactor_override = None if bfactor_override is None else float(bfactor_override)
        self.supervised_assignment_by_index = bool(supervised_assignment_by_index)
        self.assignment_strategy: AssignmentStrategy = assignment_strategy or HardAssignment()
        self.projection_batch_size = None if projection_batch_size is None else int(projection_batch_size)
        if self.projection_batch_size is not None and self.projection_batch_size <= 0:
            self.projection_batch_size = None
        self.shuffle_projection_samples = bool(shuffle_projection_samples)
        self.ctf_params = ctf_params

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

        if should_align_to_chains is None:
            should_align_to_chains = list(range(len(full_sequences)))
        self.should_align_to_chains = should_align_to_chains
        self.align_to_chain_mask = alignment_mask_by_chain(
            full_sequences,
            chains_to_align=should_align_to_chains,
            sequence_types=self.sequence_types,
        ).to(self.device)

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
            reference = {
                "coords": coords.to(self.device),
                "mask": mask_tensor.to(self.device),
                "bfactors": b_factors.to(self.device),
                "atomic_numbers": elements.to(self.device),
                "starting_residue_indices": starting_residue_indices,
            }
            reference["resolved_mask"] = reference["mask"].to(torch.bool)
            reference["render_mask"] = self._build_render_mask(reference)
            self.reference_structures.append(reference)

        self.last_loss_value: float | None = None
        self.last_loss_per_conformation: dict[int, float] = {}
        self.last_rmsd_per_conformation: dict[int, float] = {}
        self.last_cosine_similarity_per_conformation: dict[int, float] = {}
        self.last_assignment_accuracy: float | None = None
        self.last_assignment_confusion_matrix: torch.Tensor | None = None
        self.last_assignment_precision_per_conformation: dict[int, float] = {}
        self.last_assignment_recall_per_conformation: dict[int, float] = {}
        self.last_assignment_f1_per_conformation: dict[int, float] = {}
        self.last_assignment_macro_precision: float | None = None
        self.last_assignment_macro_recall: float | None = None
        self.last_assignment_macro_f1: float | None = None
        self.last_assignment_counts: dict[int, int] = {}
        self.last_assignment_margin_mean: float | None = None
        self.last_assignment_margin_min: float | None = None
        self.last_assignment_margin_max: float | None = None
        self.last_assignment_mode: str = (
            "supervised" if self.supervised_assignment_by_index
            else type(self.assignment_strategy).__name__
        )

        self._setup_fast_solver()

    def _build_render_mask(self, reference: dict) -> torch.Tensor | None:
        return reference.get("resolved_mask") if self.use_resolved_atoms_only else None

    def _projection_loss_per_sample(self, rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if rendered.shape != gt.shape:
            raise ValueError(
                f"Rendered and GT projections must have matching shapes, got {rendered.shape} and {gt.shape}."
            )
        if self.ctf_params is not None:
            # MSE on mean-centered images: removes ~1.0 DC background,
            # exposes structural modulation for gradient signal.
            r_flat = rendered.flatten(start_dim=1)
            g_flat = gt.flatten(start_dim=1)
            r_centered = r_flat - r_flat.mean(dim=1, keepdim=True)
            g_centered = g_flat - g_flat.mean(dim=1, keepdim=True)
            return (r_centered - g_centered).square().mean(dim=1)
        return (rendered - gt).square().flatten(start_dim=1).mean(dim=1)

    def _projection_cosine_similarity_per_sample(
        self,
        rendered: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        rendered_flat = rendered.flatten(start_dim=1)
        gt_flat = gt.flatten(start_dim=1)
        if self.ctf_params is not None:
            # Use centered cosine similarity (= NCC) to ignore DC background
            rendered_flat = rendered_flat - rendered_flat.mean(dim=1, keepdim=True)
            gt_flat = gt_flat - gt_flat.mean(dim=1, keepdim=True)
        return F.cosine_similarity(rendered_flat, gt_flat, dim=1, eps=1e-8)

    def _setup_fast_solver(self) -> None:
        if self.dataset.num_conformations != len(self.reference_structures):
            raise RuntimeError(
                "The number of image conformations must match the number of reference structures. "
                f"Got {self.dataset.num_conformations} and {len(self.reference_structures)}."
            )

        self._fast_renderers = []
        for conformation in self.dataset.conformations:
            renderers_for_conformation: list[CryoImageRenderer] = []

            for reference in self.reference_structures:
                coords = reference["coords"]
                atomic_numbers = reference["atomic_numbers"]
                bfactors = reference["bfactors"]
                render_mask = reference.get("render_mask")

                if render_mask is not None:
                    coords, atomic_numbers, bfactors = CryoImageRenderer.apply_atom_mask(
                        coords, atomic_numbers, bfactors, render_mask
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
                    ctf_params=self.ctf_params,
                )
                renderers_for_conformation.append(renderer)

            self._fast_renderers.append(renderers_for_conformation)

    def _render_projection_batch(
        self,
        *,
        conformation_index: int,
        structure_index: int,
        aligned_structure: dict[str, torch.Tensor],
        rotations: torch.Tensor,
    ) -> torch.Tensor:
        return self._fast_renderers[conformation_index][structure_index].render_all_rotations_from_coords(
            coords=aligned_structure["coords"],
            atomic_numbers=aligned_structure["atomic_numbers"],
            bfactors=aligned_structure["bfactors"],
            rotations=rotations,
            max_rotations_per_batch=self.max_rotations_per_batch,
        )

    def _align_to_reference(self, x_0_hat: torch.Tensor, reference: dict) -> torch.Tensor:
        alignment_mask = self.AF3_to_pdb_mask & self.align_to_chain_mask
        _, aligned_x_0_hat, _, _ = self_aligned_rmsd(
            x_0_hat,
            reference["coords"].unsqueeze(0),
            alignment_mask,
        )
        return aligned_x_0_hat

    def _candidate_structure_count(self, x_0_hat: torch.Tensor) -> int:
        expected = len(self.reference_structures)
        actual = int(x_0_hat.shape[0])
        if actual != expected:
            raise ValueError(
                "CryoEM image guidance expects one candidate structure per reference conformation. "
                f"Got batch={actual}, references={expected}."
            )
        return expected

    def _prepare_aligned_structures(self, x_0_hat: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        aligned_structures: list[dict[str, torch.Tensor]] = []
        self.last_rmsd_per_conformation = {}

        for structure_index, reference in enumerate(self.reference_structures):
            aligned = self._align_to_reference(x_0_hat[structure_index:structure_index + 1], reference)

            rmsd_mask = self.AF3_to_pdb_mask
            if int(rmsd_mask.sum().item()) > 0:
                diff = aligned[:, rmsd_mask, :] - reference["coords"][rmsd_mask, :].unsqueeze(0)
            else:
                diff = aligned - reference["coords"].unsqueeze(0)
            self.last_rmsd_per_conformation[structure_index] = float(
                diff.square().sum(dim=-1).mean().sqrt().detach().item()
            )

            coords = aligned
            atomic_numbers = reference["atomic_numbers"]
            bfactors = reference["bfactors"]
            render_mask = reference.get("render_mask")
            if render_mask is not None:
                coords, atomic_numbers, bfactors = CryoImageRenderer.apply_atom_mask(
                    coords, atomic_numbers, bfactors, render_mask
                )
            aligned_structures.append(
                {
                    "coords": coords,
                    "atomic_numbers": atomic_numbers,
                    "bfactors": bfactors,
                }
            )

        return aligned_structures

    def _projection_loader_batch_size(self) -> int:
        return self.projection_batch_size or len(self.dataset)

    def _iter_projection_batches(self):
        loader = TorchDataLoader(
            self.dataset,
            batch_size=self._projection_loader_batch_size(),
            shuffle=self.shuffle_projection_samples,
            num_workers=0,
            drop_last=False,
        )
        for batch in loader:
            yield {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

    def _compute_loss_matrix_for_batch(
        self,
        aligned_structures: list[dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projections = batch["projection"]
        rotations = batch["rotation"]
        conformation_indices = batch["conformation_index"].to(dtype=torch.long)

        num_samples = int(projections.shape[0])
        num_structures = len(aligned_structures)
        loss_matrix = torch.empty((num_samples, num_structures), device=self.device, dtype=projections.dtype)
        cosine_matrix = torch.empty_like(loss_matrix)

        for conformation_index in conformation_indices.unique(sorted=True).tolist():
            conf_mask = conformation_indices == int(conformation_index)
            sample_indices = conf_mask.nonzero(as_tuple=False).squeeze(-1)
            conf_projections = projections.index_select(0, sample_indices)
            conf_rotations = rotations.index_select(0, sample_indices)

            for structure_index, aligned_structure in enumerate(aligned_structures):
                rendered = self._render_projection_batch(
                    conformation_index=int(conformation_index),
                    structure_index=structure_index,
                    aligned_structure=aligned_structure,
                    rotations=conf_rotations,
                )
                loss_matrix[sample_indices, structure_index] = self._projection_loss_per_sample(
                    rendered,
                    conf_projections,
                )
                cosine_matrix[sample_indices, structure_index] = self._projection_cosine_similarity_per_sample(
                    rendered,
                    conf_projections,
                )

        return loss_matrix, cosine_matrix

    def _select_assignments(
        self,
        loss_matrix: torch.Tensor,
        gt_conformation_index: torch.Tensor,
        step: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.supervised_assignment_by_index:
            assignments = gt_conformation_index.to(device=loss_matrix.device, dtype=torch.long)
            min_idx = int(assignments.min().item())
            max_idx = int(assignments.max().item())
            if min_idx < 0 or max_idx >= loss_matrix.shape[1]:
                raise ValueError(
                    "Ground-truth conformation indices must be valid structure indices in supervised mode. "
                    f"Got range [{min_idx}, {max_idx}] for {loss_matrix.shape[1]} structures."
                )
            selected_losses = loss_matrix.gather(1, assignments.unsqueeze(1)).squeeze(1)
            return assignments, selected_losses

        selected_losses, assignments = self.assignment_strategy.assign(loss_matrix, step=step)
        return assignments, selected_losses

    def _assignment_margin(self, loss_matrix: torch.Tensor) -> torch.Tensor:
        if loss_matrix.shape[1] < 2:
            return torch.zeros(loss_matrix.shape[0], dtype=loss_matrix.dtype, device=loss_matrix.device)
        best_two = torch.topk(loss_matrix.detach(), k=2, dim=1, largest=False).values
        return best_two[:, 1] - best_two[:, 0]

    def _store_projection_log(
        self,
        *,
        batch: dict[str, torch.Tensor],
        assignments: torch.Tensor,
        aligned_structures: list[dict[str, torch.Tensor]],
        step: int | None,
    ) -> None:
        projections = batch["projection"]
        num_samples = int(projections.shape[0])
        k = min(self._projection_log_pairs, num_samples)
        if k <= 0:
            return

        sample_indices = torch.linspace(0, num_samples - 1, steps=k, device=projections.device).long()
        gt_images = []
        pred_images = []
        rotation_indices = []
        true_conformation_indices = []
        assigned_structure_indices = []

        for sample_index in sample_indices.tolist():
            conformation_index = int(batch["conformation_index"][sample_index].item())
            structure_index = int(assignments[sample_index].item())
            rendered = self._render_projection_batch(
                conformation_index=conformation_index,
                structure_index=structure_index,
                aligned_structure=aligned_structures[structure_index],
                rotations=batch["rotation"][sample_index:sample_index + 1],
            )
            gt_images.append(projections[sample_index].detach().float().cpu())
            pred_images.append(rendered[0].detach().float().cpu())
            rotation_indices.append(int(batch["rotation_index"][sample_index].item()))
            true_conformation_indices.append(conformation_index)
            assigned_structure_indices.append(structure_index)

        self._projection_log_payload = {
            "rotation_indices": rotation_indices,
            "true_conformation_indices": true_conformation_indices,
            "assigned_structure_indices": assigned_structure_indices,
            "gt": gt_images,
            "pred": pred_images,
        }
        self._last_projection_log_step = step

    def _update_assignment_diagnostics(
        self,
        *,
        confusion: torch.Tensor,
        assignment_counts: torch.Tensor,
        loss_sums_by_structure: torch.Tensor,
        cosine_sums_by_structure: torch.Tensor,
        margin_values: torch.Tensor,
    ) -> None:
        counts_float = assignment_counts.to(dtype=loss_sums_by_structure.dtype).clamp_min(1.0)
        mean_losses = loss_sums_by_structure / counts_float
        mean_cosine = cosine_sums_by_structure / counts_float

        self.last_loss_per_conformation = {}
        self.last_cosine_similarity_per_conformation = {}
        self.last_assignment_counts = {}

        for structure_index in range(int(assignment_counts.shape[0])):
            count = int(assignment_counts[structure_index].item())
            self.last_assignment_counts[structure_index] = count
            if count > 0:
                self.last_loss_per_conformation[structure_index] = float(mean_losses[structure_index].item())
                self.last_cosine_similarity_per_conformation[structure_index] = float(
                    mean_cosine[structure_index].item()
                )

        self.last_assignment_confusion_matrix = confusion.detach().cpu()

        conf_float = confusion.to(torch.float32)
        tp = conf_float.diag()
        predicted = conf_float.sum(dim=0)
        actual = conf_float.sum(dim=1)

        precision = torch.where(predicted > 0, tp / predicted, torch.zeros_like(tp))
        recall = torch.where(actual > 0, tp / actual, torch.zeros_like(tp))
        f1 = torch.where(
            (precision + recall) > 0,
            2.0 * precision * recall / (precision + recall),
            torch.zeros_like(precision),
        )

        total = float(conf_float.sum().item())
        self.last_assignment_accuracy = float(tp.sum().item() / total) if total > 0 else None
        self.last_assignment_precision_per_conformation = {
            idx: float(value.item()) for idx, value in enumerate(precision)
        }
        self.last_assignment_recall_per_conformation = {
            idx: float(value.item()) for idx, value in enumerate(recall)
        }
        self.last_assignment_f1_per_conformation = {
            idx: float(value.item()) for idx, value in enumerate(f1)
        }
        self.last_assignment_macro_precision = float(precision.mean().item())
        self.last_assignment_macro_recall = float(recall.mean().item())
        self.last_assignment_macro_f1 = float(f1.mean().item())

        self.last_assignment_margin_mean = float(margin_values.mean().item())
        self.last_assignment_margin_min = float(margin_values.min().item())
        self.last_assignment_margin_max = float(margin_values.max().item())

    def __call__(self, x_0_hat: torch.Tensor, time, structures=None, i=None, step=None):
        self.last_assignment_mode = (
            "supervised" if self.supervised_assignment_by_index
            else type(self.assignment_strategy).__name__
        )
        self.last_loss_per_conformation = {}
        self.last_rmsd_per_conformation = {}
        self.last_cosine_similarity_per_conformation = {}
        self.last_assignment_accuracy = None
        self.last_assignment_confusion_matrix = None
        self.last_assignment_precision_per_conformation = {}
        self.last_assignment_recall_per_conformation = {}
        self.last_assignment_f1_per_conformation = {}
        self.last_assignment_macro_precision = None
        self.last_assignment_macro_recall = None
        self.last_assignment_macro_f1 = None
        self.last_assignment_counts = {}
        self.last_assignment_margin_mean = None
        self.last_assignment_margin_min = None
        self.last_assignment_margin_max = None

        do_log = False
        if self._projection_log_every:
            self._projection_log_calls += 1
            if step is not None:
                do_log = (int(step) % self._projection_log_every) == 0
            else:
                do_log = ((self._projection_log_calls - 1) % self._projection_log_every) == 0

        num_structures = self._candidate_structure_count(x_0_hat)
        aligned_structures = self._prepare_aligned_structures(x_0_hat)

        total_selected_loss = x_0_hat.new_tensor(0.0)
        total_projections = 0
        assignment_counts = torch.zeros(num_structures, dtype=torch.long, device=self.device)
        loss_sums_by_structure = torch.zeros(num_structures, dtype=x_0_hat.dtype, device=self.device)
        cosine_sums_by_structure = torch.zeros(num_structures, dtype=x_0_hat.dtype, device=self.device)
        confusion = torch.zeros(
            (self.dataset.num_conformations, num_structures),
            dtype=torch.long,
            device=self.device,
        )
        margin_chunks: list[torch.Tensor] = []

        for batch in self._iter_projection_batches():
            loss_matrix, cosine_matrix = self._compute_loss_matrix_for_batch(aligned_structures, batch)
            gt_conformation_index = batch["conformation_index"].to(dtype=torch.long)
            assignments, selected_losses = self._select_assignments(loss_matrix, gt_conformation_index, step=step)
            selected_cosine = cosine_matrix.gather(1, assignments.unsqueeze(1)).squeeze(1)

            total_selected_loss = total_selected_loss + selected_losses.sum()
            total_projections += int(selected_losses.shape[0])

            ones = torch.ones_like(assignments, dtype=torch.long)
            assignment_counts.scatter_add_(0, assignments, ones)
            loss_sums_by_structure.scatter_add_(0, assignments, selected_losses.detach())
            cosine_sums_by_structure.scatter_add_(0, assignments, selected_cosine.detach())

            flat_confusion_index = gt_conformation_index * num_structures + assignments
            confusion.view(-1).scatter_add_(0, flat_confusion_index, ones)
            margin_chunks.append(self._assignment_margin(loss_matrix))

            if do_log and self._projection_log_payload is None:
                self._store_projection_log(
                    batch=batch,
                    assignments=assignments.detach(),
                    aligned_structures=aligned_structures,
                    step=step,
                )

        if total_projections <= 0:
            raise RuntimeError("CryoEM image guidance received no projection samples.")

        loss = total_selected_loss / float(total_projections)
        self.last_loss_value = float(loss.detach().item())

        margin_values = torch.cat(margin_chunks, dim=0)
        self._update_assignment_diagnostics(
            confusion=confusion,
            assignment_counts=assignment_counts,
            loss_sums_by_structure=loss_sums_by_structure,
            cosine_sums_by_structure=cosine_sums_by_structure,
            margin_values=margin_values,
        )

        return loss, None, None

    def wandb_log(self, x_0_hat):
        log = {
            "loss": self.last_loss_value,
            "cryoimage/loss": self.last_loss_value,
            "cryoimage/supervised_assignment_by_index": float(self.supervised_assignment_by_index),
            "cryoimage/dynamic_assignment_enabled": float(not self.supervised_assignment_by_index),
            "cryoimage/num_projection_samples": float(len(self.dataset)),
            "cryoimage/projection_batch_size": float(self._projection_loader_batch_size()),
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
        if self.last_assignment_accuracy is not None:
            log["cryoimage/assignment_accuracy"] = self.last_assignment_accuracy
        if self.last_assignment_macro_precision is not None:
            log["cryoimage/assignment_precision_macro"] = self.last_assignment_macro_precision
        if self.last_assignment_macro_recall is not None:
            log["cryoimage/assignment_recall_macro"] = self.last_assignment_macro_recall
        if self.last_assignment_macro_f1 is not None:
            log["cryoimage/assignment_f1_macro"] = self.last_assignment_macro_f1
        if self.last_assignment_margin_mean is not None:
            log["cryoimage/assignment_margin_mean"] = self.last_assignment_margin_mean
        if self.last_assignment_margin_min is not None:
            log["cryoimage/assignment_margin_min"] = self.last_assignment_margin_min
        if self.last_assignment_margin_max is not None:
            log["cryoimage/assignment_margin_max"] = self.last_assignment_margin_max

        log["cryoimage/assignment_strategy"] = self.last_assignment_mode
        log.update(self.assignment_strategy.extra_wandb_log())

        for idx, value in self.last_loss_per_conformation.items():
            log[f"cryoimage/loss_conf_{idx}"] = float(value)
        for idx, value in self.last_rmsd_per_conformation.items():
            log[f"cryoimage/rmsd_conf_{idx}"] = float(value)
        for idx, value in self.last_cosine_similarity_per_conformation.items():
            log[f"cryoimage/cosine_similarity_conf_{idx}"] = float(value)
        for idx, value in self.last_assignment_counts.items():
            log[f"cryoimage/assigned_count_conf_{idx}"] = float(value)
        for idx, value in self.last_assignment_precision_per_conformation.items():
            log[f"cryoimage/assignment_precision_conf_{idx}"] = float(value)
        for idx, value in self.last_assignment_recall_per_conformation.items():
            log[f"cryoimage/assignment_recall_conf_{idx}"] = float(value)
        for idx, value in self.last_assignment_f1_per_conformation.items():
            log[f"cryoimage/assignment_f1_conf_{idx}"] = float(value)

        if self.last_assignment_confusion_matrix is not None:
            confusion = self.last_assignment_confusion_matrix
            for true_index in range(confusion.shape[0]):
                for pred_index in range(confusion.shape[1]):
                    log[f"cryoimage/confusion_true_{true_index}_pred_{pred_index}"] = float(
                        confusion[true_index, pred_index].item()
                    )

        payload = self._projection_log_payload
        if payload is not None:
            try:
                import wandb

                gt_imgs = []
                pred_imgs = []
                diff_imgs = []
                for idx, gt_img in enumerate(payload["gt"]):
                    pred_img = payload["pred"][idx]
                    rotation_index = payload["rotation_indices"][idx]
                    true_conf = payload["true_conformation_indices"][idx]
                    assigned_conf = payload["assigned_structure_indices"][idx]

                    display_imgs = []
                    for img in (gt_img, pred_img):
                        img = img.float()
                        vmin = float(img.min().item())
                        vmax = float(img.max().item())
                        if vmax > vmin:
                            img = (img - vmin) / (vmax - vmin)
                        else:
                            img = torch.zeros_like(img)
                        display_imgs.append(img)

                    diff = pred_img.float() - gt_img.float()
                    max_abs = float(diff.abs().max().item())
                    if max_abs > 0.0:
                        diff = diff / (2.0 * max_abs) + 0.5
                    else:
                        diff = torch.full_like(diff, 0.5)
                    diff = diff.clamp(0.0, 1.0)

                    caption = f"rot={rotation_index}, true={true_conf}, assigned={assigned_conf}"
                    gt_imgs.append(wandb.Image(display_imgs[0].numpy(), caption=caption))
                    pred_imgs.append(wandb.Image(display_imgs[1].numpy(), caption=caption))
                    diff_imgs.append(wandb.Image(diff.numpy(), caption=f"{caption}, pred-gt"))

                log["cryoimage/gt_projections"] = gt_imgs
                log["cryoimage/pred_projections"] = pred_imgs
                log["cryoimage/diff_projections"] = diff_imgs
                if self._last_projection_log_step is not None:
                    log["cryoimage/projection_log_step"] = self._last_projection_log_step
            except Exception:
                pass
            self._projection_log_payload = None

        return log


