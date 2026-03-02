from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from geomloss import SamplesLoss

from .abstract_loss_funciton import AbstractLossFunction
from ..protenix.metrics.rmsd import self_aligned_rmsd
from ..utils.cryoimage_renderer import (
    CryoImageDataset,
    CryoImageRenderer,
)
from ..utils.io import load_pdb_atom_locations_full, alignment_mask_by_chain
from ..utils.residue_range_mask import build_residue_subset_mask


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
        atom_batch_size: int = 1024,
        loss_reduction: str = "mean",
        loss_type: str = "mse",
        use_checkpointing: bool = False,
        log_projection_every: int = 10,
        log_projection_pairs: int = 3,
        max_rotations_per_batch: int | None = None,
        use_resolved_atoms_only: bool = True,
        normalize_projections: bool = False,
        projection_normalization_eps: float = 1e-6,
        fft_log_eps: float = 1e-6,
        fft_bandpass_low: float | None = None,
        fft_bandpass_high: float | None = None,
        ncc_eps: float = 1e-6,
        ot_p: int = 1,
        ot_blur: float = 0.5,
        ot_scaling: float = 0.9,
        ot_reach: float | None = None,
        ot_backend: str = "online",
        ot_debias: bool = True,
        ot_eps: float = 1e-6,
        ot_downsample: int | None = None,
        loss_topk: int | None = None,
        image_blur_sigma: float | None = None,
        image_blur_kernel_size: int | None = None,
        image_blur_sigma_schedule: dict | list | None = None,
        residue_ranges_pdb: list | None = None,
        bfactor_override: float | None = None,
        supervised_assignment_by_index: bool = True,
    ):
        self.device = torch.device(device)
        self.atom_batch_size = atom_batch_size
        self.loss_reduction = loss_reduction
        self.loss_type = loss_type.lower()
        self.use_checkpointing = use_checkpointing
        self._projection_log_every = log_projection_every
        self._projection_log_pairs = log_projection_pairs
        self._projection_log_calls = 0
        self._projection_log_payload = None
        self._fast_renderers: list[CryoImageRenderer] = []
        self.max_rotations_per_batch = max_rotations_per_batch
        self.use_resolved_atoms_only = use_resolved_atoms_only
        self.normalize_projections = normalize_projections
        self.projection_normalization_eps = float(projection_normalization_eps)
        self.fft_log_eps = float(fft_log_eps)
        self.fft_bandpass_low = fft_bandpass_low
        self.fft_bandpass_high = fft_bandpass_high
        self.ncc_eps = float(ncc_eps)
        self._fft_bandpass_cache: dict[tuple, torch.Tensor] = {}
        self._ot_coords_cache: dict[tuple, torch.Tensor] = {}
        self.ot_p = int(ot_p)
        self.ot_blur = float(ot_blur)
        self.ot_scaling = float(ot_scaling)
        self.ot_reach = ot_reach
        self.ot_backend = ot_backend
        self.ot_debias = bool(ot_debias)
        self.ot_eps = float(ot_eps)
        self.ot_downsample = ot_downsample if ot_downsample is None else int(ot_downsample)
        self.loss_topk = loss_topk if loss_topk is None else int(loss_topk)
        self.image_blur_sigma = image_blur_sigma
        self.image_blur_kernel_size = image_blur_kernel_size
        self.image_blur_sigma_schedule = image_blur_sigma_schedule
        self.residue_ranges_pdb = residue_ranges_pdb
        self.bfactor_override = None if bfactor_override is None else float(bfactor_override)
        self.supervised_assignment_by_index = bool(supervised_assignment_by_index)
        self._current_image_blur_sigma: float | None = None
        self._blur_kernel_cache: dict[tuple, torch.Tensor] = {}
        self._ot_loss_function = SamplesLoss(
            loss="sinkhorn",
            p=self.ot_p,
            blur=self.ot_blur,
            backend=self.ot_backend,
            debias=self.ot_debias,
            reach=self.ot_reach,
            scaling=self.ot_scaling,
        )

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
            residue_subset_mask = self._build_residue_subset_mask(starting_residue_indices)
            ref = {
                "coords": coords.to(self.device),
                "mask": mask_tensor.to(self.device),
                "bfactors": b_factors.to(self.device),
                "atomic_numbers": elements.to(self.device),
                "starting_residue_indices": starting_residue_indices,
                "residue_subset_mask": residue_subset_mask.to(self.device) if residue_subset_mask is not None else None,
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

    def _build_residue_subset_mask(self, starting_residue_indices: list[int] | None) -> torch.Tensor | None:
        return build_residue_subset_mask(
            full_sequences=self.full_sequences,
            sequence_types=self.sequence_types,
            starting_residue_indices=starting_residue_indices,
            residue_ranges_pdb=self.residue_ranges_pdb,
        )

    def _build_render_mask(self, reference: dict) -> torch.Tensor | None:
        base_mask = reference.get("resolved_mask") if self.use_resolved_atoms_only else None
        subset_mask = reference.get("residue_subset_mask")

        if subset_mask is None:
            return base_mask
        if base_mask is None:
            return subset_mask.to(torch.bool)
        return (base_mask & subset_mask.to(torch.bool))

    def _normalize_pair(self, rendered: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize both rendered and GT projections per image (per rotation) to improve conditioning.

        Shapes:
          rendered: [R, H, W]
          gt:       [R, H, W]
        """
        sigma = self._current_image_blur_sigma
        if sigma is None:
            sigma = self.image_blur_sigma
        if sigma is not None and sigma > 0:
            rendered = self._gaussian_blur(rendered, sigma=sigma)
            gt = self._gaussian_blur(gt, sigma=sigma)
        if not self.normalize_projections:
            return rendered, gt

        eps = self.projection_normalization_eps
        # Per-projection z-score normalization.
        dims = tuple(range(1, rendered.ndim))
        r_mean = rendered.mean(dim=dims, keepdim=True)
        r_std = rendered.std(dim=dims, keepdim=True).clamp_min(eps)
        g_mean = gt.mean(dim=dims, keepdim=True)
        g_std = gt.std(dim=dims, keepdim=True).clamp_min(eps)
        return (rendered - r_mean) / r_std, (gt - g_mean) / g_std

    def _gaussian_blur(self, images: torch.Tensor, sigma: float) -> torch.Tensor:
        if images.ndim != 3:
            raise ValueError(f"Expected images with shape [R, H, W], got {images.shape}.")
        sigma = float(sigma)
        if sigma <= 0:
            return images
        kernel = self._get_gaussian_kernel(
            sigma=sigma,
            kernel_size=self.image_blur_kernel_size,
            device=images.device,
            dtype=images.dtype,
        )
        pad = kernel.shape[-1] // 2
        x = images.unsqueeze(1)  # [R, 1, H, W]
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        x = F.conv2d(x, kernel)
        return x.squeeze(1)

    def _get_gaussian_kernel(
        self,
        sigma: float,
        kernel_size: int | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if kernel_size is None:
            kernel_size = int(2 * torch.ceil(torch.tensor(3.0 * sigma)).item() + 1)
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        key = (sigma, kernel_size, str(device), str(dtype))
        cached = self._blur_kernel_cache.get(key)
        if cached is not None:
            return cached

        coords = torch.arange(kernel_size, device=device, dtype=dtype)
        center = (kernel_size - 1) / 2.0
        grid = coords - center
        g1 = torch.exp(-(grid ** 2) / (2 * sigma ** 2))
        g1 = g1 / g1.sum()
        kernel_2d = torch.outer(g1, g1)
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel = kernel_2d.unsqueeze(0).unsqueeze(0)
        self._blur_kernel_cache[key] = kernel
        return kernel

    def _scheduled_blur_sigma(self, time, step: int | None, i: int | None) -> float | None:
        schedule = self.image_blur_sigma_schedule
        if schedule is None:
            return self.image_blur_sigma

        # Piecewise constant list: [{"step": 0, "sigma": 5.0}, {"step": 100, "sigma": 0.0}]
        if isinstance(schedule, (list, tuple)):
            step_val = step if step is not None else i if i is not None else 0
            best_sigma = None
            best_step = None
            for item in schedule:
                if isinstance(item, dict):
                    s = int(item.get("step", 0))
                    sigma = float(item.get("sigma", 0.0))
                else:
                    s, sigma = item
                    s = int(s)
                    sigma = float(sigma)
                if s <= step_val and (best_step is None or s > best_step):
                    best_step = s
                    best_sigma = sigma
            return best_sigma if best_sigma is not None else self.image_blur_sigma

        if isinstance(schedule, dict):
            use_time = bool(schedule.get("use_time", False))
            start = float(schedule.get("start", self.image_blur_sigma or 0.0))
            end = float(schedule.get("end", 0.0))

            if use_time and time is not None:
                t_val = float(time) if not torch.is_tensor(time) else float(time.detach().item())
                t0 = float(schedule.get("time_start", 1.0))
                t1 = float(schedule.get("time_end", 0.0))
                if t0 == t1:
                    return end
                # Map t_val to [0,1] based on [t0, t1]
                t = (t_val - t0) / (t1 - t0)
            else:
                step_val = step if step is not None else i if i is not None else 0
                start_step = int(schedule.get("start_step", 0))
                end_step = int(schedule.get("end_step", start_step))
                if end_step <= start_step:
                    return end
                t = (step_val - start_step) / float(end_step - start_step)

            t = max(0.0, min(1.0, t))
            return start + t * (end - start)

        return self.image_blur_sigma

    def _projection_loss(self, rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        rendered, gt = self._normalize_pair(rendered, gt)

        use_topk = self.loss_topk is not None and self.loss_topk > 0

        if use_topk:
            per = self._projection_loss_per_rotation(rendered, gt, already_normalized=True)
            return self._reduce_topk(per)

        if self.loss_type in ("mse", "l2"):
            return F.mse_loss(rendered, gt, reduction=self.loss_reduction)
        if self.loss_type in ("l1", "mae"):
            return F.l1_loss(rendered, gt, reduction=self.loss_reduction)
        if self.loss_type in ("smooth_l1", "huber"):
            return F.smooth_l1_loss(rendered, gt, reduction=self.loss_reduction)
        if self.loss_type in (
            "fft_mag_mse",
            "fft_magnitude_mse",
            "fft_mag",
            "fft_log_mag_mse",
            "fft_log_mag",
            "fft_log_magnitude_mse",
        ):
            return self._fft_magnitude_loss(rendered, gt, per_rotation=False)
        if self.loss_type in ("ncc", "normalized_cross_correlation"):
            return self._ncc_loss(rendered, gt, per_rotation=False)
        if self.loss_type in ("ot", "optimal_transport", "sinkhorn"):
            return self._ot_loss(rendered, gt, per_rotation=False)
        raise ValueError(
            "Unknown loss_type="
            f"{self.loss_type!r}. Expected mse|l1|smooth_l1|fft_mag_mse|fft_log_mag_mse|ncc|ot."
        )

    def _projection_loss_per_rotation(
        self,
        rendered: torch.Tensor,
        gt: torch.Tensor,
        already_normalized: bool = False,
    ) -> torch.Tensor:
        if not already_normalized:
            rendered, gt = self._normalize_pair(rendered, gt)
        if self.loss_type in ("mse", "l2"):
            return (rendered - gt).pow(2).mean(dim=(-2, -1))
        if self.loss_type in ("l1", "mae"):
            return (rendered - gt).abs().mean(dim=(-2, -1))
        if self.loss_type in ("smooth_l1", "huber"):
            return F.smooth_l1_loss(rendered, gt, reduction="none").mean(dim=(-2, -1))
        if self.loss_type in (
            "fft_mag_mse",
            "fft_magnitude_mse",
            "fft_mag",
            "fft_log_mag_mse",
            "fft_log_mag",
            "fft_log_magnitude_mse",
        ):
            return self._fft_magnitude_loss(rendered, gt, per_rotation=True)
        if self.loss_type in ("ncc", "normalized_cross_correlation"):
            return self._ncc_loss(rendered, gt, per_rotation=True)
        if self.loss_type in ("ot", "optimal_transport", "sinkhorn"):
            return self._ot_loss(rendered, gt, per_rotation=True)
        raise ValueError(f"Unknown loss_type={self.loss_type!r}.")

    def _reduce_topk(self, per_rotation: torch.Tensor) -> torch.Tensor:
        if per_rotation.ndim != 1:
            per_rotation = per_rotation.flatten()
        k = min(int(self.loss_topk), int(per_rotation.numel()))
        if k <= 0:
            return per_rotation.mean()
        topk_vals = torch.topk(per_rotation, k=k, largest=True).values
        if self.loss_reduction == "sum":
            return topk_vals.sum()
        if self.loss_reduction == "none":
            return topk_vals
        return topk_vals.mean()

    def _fft_bandpass_mask(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.fft_bandpass_low is None and self.fft_bandpass_high is None:
            return None
        low = 0.0 if self.fft_bandpass_low is None else float(self.fft_bandpass_low)
        high = 0.5 if self.fft_bandpass_high is None else float(self.fft_bandpass_high)
        key = (height, width, str(device), str(dtype), low, high)
        cached = self._fft_bandpass_cache.get(key)
        if cached is not None:
            return cached

        fy = torch.fft.fftfreq(height, d=1.0, device=device)
        fx = torch.fft.fftfreq(width, d=1.0, device=device)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(xx ** 2 + yy ** 2)
        mask = (radius >= low) & (radius <= high)
        mask = mask.to(dtype=dtype)
        self._fft_bandpass_cache[key] = mask
        return mask

    def _fft_magnitude_loss(self, rendered: torch.Tensor, gt: torch.Tensor, per_rotation: bool = False) -> torch.Tensor:
        fft_rendered = torch.fft.fftn(rendered, dim=(-2, -1))
        fft_gt = torch.fft.fftn(gt, dim=(-2, -1))
        mag_rendered = torch.abs(fft_rendered)
        mag_gt = torch.abs(fft_gt)

        if self.loss_type in ("fft_log_mag_mse", "fft_log_mag", "fft_log_magnitude_mse"):
            eps = self.fft_log_eps
            mag_rendered = torch.log(mag_rendered + eps)
            mag_gt = torch.log(mag_gt + eps)

        mask = self._fft_bandpass_mask(
            height=mag_rendered.shape[-2],
            width=mag_rendered.shape[-1],
            device=mag_rendered.device,
            dtype=mag_rendered.dtype,
        )

        diff = mag_rendered - mag_gt
        if mask is not None:
            diff = diff * mask
            denom = (mask.sum() * diff.shape[0]).clamp_min(1.0)
        else:
            denom = diff.numel()

        per = (diff ** 2).mean(dim=(-2, -1))
        if per_rotation:
            return per
        if self.loss_reduction == "sum":
            return (diff ** 2).sum()
        if self.loss_reduction == "none":
            return diff ** 2
        return (diff ** 2).sum() / denom

    def _ncc_loss(self, rendered: torch.Tensor, gt: torch.Tensor, per_rotation: bool = False) -> torch.Tensor:
        dims = tuple(range(1, rendered.ndim))
        r_mean = rendered.mean(dim=dims, keepdim=True)
        g_mean = gt.mean(dim=dims, keepdim=True)
        r = rendered - r_mean
        g = gt - g_mean
        numerator = (r * g).sum(dim=dims)
        r_norm = torch.sqrt((r ** 2).sum(dim=dims))
        g_norm = torch.sqrt((g ** 2).sum(dim=dims))
        denom = (r_norm * g_norm).clamp_min(self.ncc_eps)
        ncc = numerator / denom
        loss = 1.0 - ncc

        if per_rotation:
            return loss
        if self.loss_reduction == "sum":
            return loss.sum()
        if self.loss_reduction == "none":
            return loss
        return loss.mean()

    def _get_ot_coords(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (height, width, str(device), str(dtype))
        cached = self._ot_coords_cache.get(key)
        if cached is not None:
            return cached
        ys = torch.linspace(0.0, 1.0, steps=height, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, steps=width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).reshape(-1, 2)
        self._ot_coords_cache[key] = coords
        return coords

    def _prepare_ot_weights(self, images: torch.Tensor) -> torch.Tensor:
        # images: [R, H, W]
        min_val = images.amin(dim=(-2, -1), keepdim=True)
        weights = (images - min_val).clamp_min(0.0)
        weights = weights.reshape(weights.shape[0], -1)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(self.ot_eps)
        return weights / denom

    def _ot_loss(self, rendered: torch.Tensor, gt: torch.Tensor, per_rotation: bool = False) -> torch.Tensor:
        if self.ot_downsample is not None and self.ot_downsample > 1:
            rendered = F.avg_pool2d(rendered.unsqueeze(1), kernel_size=self.ot_downsample).squeeze(1)
            gt = F.avg_pool2d(gt.unsqueeze(1), kernel_size=self.ot_downsample).squeeze(1)

        coords = self._get_ot_coords(
            height=rendered.shape[-2],
            width=rendered.shape[-1],
            device=rendered.device,
            dtype=rendered.dtype,
        )
        coords = coords.unsqueeze(0).expand(rendered.shape[0], -1, -1).contiguous()

        weights_rendered = self._prepare_ot_weights(rendered).contiguous()
        weights_gt = self._prepare_ot_weights(gt).contiguous()

        loss = self._ot_loss_function(weights_rendered, coords, weights_gt, coords)
        if per_rotation:
            if loss.ndim == 0:
                loss = loss.unsqueeze(0)
            return loss
        if loss.ndim == 0:
            return loss
        if self.loss_reduction == "sum":
            return loss.sum()
        if self.loss_reduction == "none":
            return loss
        return loss.mean()

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
        subset_mask = reference.get("residue_subset_mask")
        if subset_mask is not None:
            alignment_mask = alignment_mask & subset_mask.to(self.device)
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

        # Update blur sigma based on schedule (if provided).
        self._current_image_blur_sigma = self._scheduled_blur_sigma(time=time, step=step, i=i)
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
            subset_mask = reference.get("residue_subset_mask")
            if subset_mask is not None:
                rmsd_mask = rmsd_mask & subset_mask.to(self.device)
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
        if self._current_image_blur_sigma is not None:
            log["cryoimage/blur_sigma"] = float(self._current_image_blur_sigma)
        payload = self._projection_log_payload
        if payload is not None:
            try:
                import wandb  # local import to avoid hard dependency
                gt_imgs = []
                pred_imgs = []
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
                    gt_imgs.append(wandb.Image(for_img[0].numpy(), caption=f"rot={rot_idx}"))
                    pred_imgs.append(wandb.Image(for_img[1].numpy(), caption=f"rot={rot_idx}"))
                log["cryoimage/gt_projections"] = gt_imgs
                log["cryoimage/pred_projections"] = pred_imgs
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
