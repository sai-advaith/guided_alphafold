from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from geomloss import SamplesLoss

from .abstract_loss_funciton import AbstractLossFunction
from ..protenix.metrics.rmsd import self_aligned_rmsd
from ..utils.io import load_pdb_atom_locations_full, alignment_mask_by_chain, create_atom_mask

from cryoforward.atom_stack import AtomStack
from cryoforward.cryoesp_calculator import setup_fast_esp_solver
from cryoforward.lattice import Lattice


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
        self._fast_esp_solver = None
        self._fast_esp_lattice = None
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
        self.datasets = []
        for idx, pt_path in enumerate(image_pt_files):
            json_path = None if image_json_files is None else image_json_files[idx]
            dataset = self._load_dataset(pt_path, json_path)
            self.datasets.append(dataset)

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
        self._last_projection_log_step = None
        self._setup_fast_solver()

    @staticmethod
    def _is_range_pair(value: Any) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return False
        return all(isinstance(v, (int, float)) for v in value)

    def _normalize_residue_ranges_per_chain(self) -> list | None:
        if self.residue_ranges_pdb is None:
            return None
        if not isinstance(self.residue_ranges_pdb, (list, tuple)):
            raise ValueError("residue_ranges_pdb must be a list.")
        if len(self.residue_ranges_pdb) == 0:
            return []

        first = self.residue_ranges_pdb[0]
        if self._is_range_pair(first):
            # Single-chain shorthand: [[start, end], [start, end], ...]
            return [list(self.residue_ranges_pdb)]
        return list(self.residue_ranges_pdb)

    def _build_residue_subset_mask(self, starting_residue_indices: list[int] | None) -> torch.Tensor | None:
        ranges_per_chain = self._normalize_residue_ranges_per_chain()
        if ranges_per_chain is None:
            return None

        regions_per_sequence: list[list[int]] = []
        for chain_idx, sequence in enumerate(self.full_sequences):
            chain_ranges = ranges_per_chain[chain_idx] if chain_idx < len(ranges_per_chain) else None
            if chain_ranges is None:
                regions_per_sequence.append([])
                continue
            if self._is_range_pair(chain_ranges):
                chain_ranges = [chain_ranges]

            start_idx = 1
            if starting_residue_indices is not None and chain_idx < len(starting_residue_indices):
                start_idx = int(starting_residue_indices[chain_idx])

            seq_len = len(sequence)
            selected_residues = set()
            for range_pair in chain_ranges:
                if range_pair is None:
                    continue
                if not self._is_range_pair(range_pair):
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
            self.full_sequences,
            regions_per_sequence,
            sequence_types=self.sequence_types,
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

    def _load_dataset(self, pt_path: str, json_path: str | None):
        pt_path = Path(pt_path)
        data = torch.load(pt_path, map_location="cpu")
        meta = data.get("meta")

        if meta is None:
            if json_path is None:
                json_path = str(pt_path.with_suffix(".json"))
            meta = json.loads(Path(json_path).read_text())

        lattice_meta = meta.get("lattice", {})
        grid_dimensions = lattice_meta.get("grid_dimensions")
        voxel_sizes = lattice_meta.get("voxel_sizes")
        left_bottom = lattice_meta.get("left_bottom")
        right_upper = lattice_meta.get("right_upper")
        sublattice_radius = lattice_meta.get("sublattice_radius", 10.0)

        if grid_dimensions is None or voxel_sizes is None:
            raise ValueError(f"Missing lattice metadata in {pt_path}")

        lattice = Lattice.from_grid_dimensions_and_voxel_sizes(
            grid_dimensions=tuple(grid_dimensions),
            voxel_sizes_in_A=tuple(voxel_sizes),
            left_bottom_point_in_A=left_bottom,
            right_upper_point_in_A=right_upper,
            sublattice_radius_in_A=sublattice_radius,
            dtype=torch.float32,
            device=self.device,
        )

        projection_axis = lattice_meta.get("projection_axis")
        if projection_axis is None:
            axis_map = {"x": 0, "y": 1, "z": 2}
            projection_axis = axis_map[meta["projection_axis"]]

        return {
            "projections": data["projections"].to(self.device),
            "rotations": data["rotations"].to(self.device),
            "lattice": lattice,
            "projection_axis": projection_axis,
            "collapse_projection_axis": lattice_meta.get("collapse_projection_axis", True),
        }

    def _setup_fast_solver(self) -> None:
        if not self.datasets or not self.reference_structures:
            raise RuntimeError("Datasets and reference structures must be loaded before setting up the ESP solver.")

        dataset0 = self.datasets[0]
        reference0 = self.reference_structures[0]
        coords = reference0["coords"]
        atomic_numbers = reference0["atomic_numbers"]
        bfactors = reference0["bfactors"]
        resolved_mask = reference0.get("render_mask")

        if resolved_mask is not None:
            coords, atomic_numbers, bfactors = self._apply_resolved_mask(
                coords, atomic_numbers, bfactors, resolved_mask
            )
        if coords.ndim == 2:
            coords = coords.unsqueeze(0)
        if bfactors.ndim == 1:
            bfactors = bfactors.unsqueeze(0)
        if bfactors.ndim == 2:
            bfactors = bfactors.unsqueeze(-1)
        if bfactors.shape[0] == 1 and coords.shape[0] > 1:
            bfactors = bfactors.expand(coords.shape[0], -1, -1)

        base_stack = AtomStack.from_coords_and_atomic_numbers(
            atom_coordinates=coords,
            atomic_numbers=atomic_numbers,
            device=self.device,
        )
        base_stack.bfactors = bfactors
        if getattr(base_stack, "occupancies", None) is None:
            base_stack.occupancies = torch.ones(
                (coords.shape[0],),
                dtype=coords.dtype,
                device=coords.device,
            )

        lattice = dataset0["lattice"]
        self._fast_esp_solver = setup_fast_esp_solver(
            base_stack,
            lattice,
            per_voxel_averaging=True,
            use_checkpointing=False,
            use_autocast=False,
        )
        self._fast_esp_lattice = lattice

    def _render_all_rotations(
        self,
        coords: torch.Tensor,
        atomic_numbers: torch.Tensor,
        bfactors: torch.Tensor,
        dataset: dict,
        resolved_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if resolved_mask is not None:
            coords, atomic_numbers, bfactors = self._apply_resolved_mask(
                coords, atomic_numbers, bfactors, resolved_mask
            )
        if coords.ndim == 2:
            coords = coords.unsqueeze(0)
        batch = coords.shape[0]
        if bfactors.ndim == 2:
            bfactors = bfactors.unsqueeze(-1)
        if bfactors.shape[0] == 1 and batch > 1:
            bfactors = bfactors.expand(batch, -1, -1)

        if self._fast_esp_solver is None or self._fast_esp_lattice is None:
            raise RuntimeError("Fast ESP solver was not initialized during setup.")
        _, compute_batch_from_coords = self._fast_esp_solver

        rotations = dataset["rotations"]
        num_rots = int(rotations.shape[0])

        base_batch = coords.shape[0]
        center = coords.mean(dim=1, keepdim=True)
        coords_centered = coords - center

        lattice = self._fast_esp_lattice
        grid_dims = tuple(int(x) for x in lattice.grid_dimensions.tolist())
        projection_axis = int(dataset["projection_axis"])
        collapse_projection_axis = bool(dataset["collapse_projection_axis"])
        depth = float(lattice.voxel_sizes_in_A[projection_axis].item())
        proj_axis = projection_axis + 1  # account for [rot] dim

        atomic_numbers_for_solver = atomic_numbers.contiguous()
        if atomic_numbers_for_solver.ndim == 1:
            atomic_numbers_for_solver = atomic_numbers_for_solver.unsqueeze(-1)
        bfactors_for_solver = bfactors.contiguous()
        occupancies_for_solver = torch.ones(
            (base_batch,),
            dtype=coords.dtype,
            device=coords.device,
        )

        max_rots = self.max_rotations_per_batch
        if max_rots is None or max_rots <= 0:
            max_rots = num_rots

        projections = []
        for rot_start in range(0, num_rots, max_rots):
            rot_end = min(rot_start + max_rots, num_rots)
            rot_batch = rotations[rot_start:rot_end]
            # Match cryoforward AtomStack.rotate convention: (x - c) @ R.T + c
            coords_batch = torch.einsum("rji,bnj->rbni", rot_batch, coords_centered) + center.unsqueeze(0)

            vols = compute_batch_from_coords(
                coords_batch,
                bfactors_for_solver,
                atomic_numbers_for_solver,
                occupancies_for_solver,
            )

            if vols.ndim == 2:
                vols = vols.reshape(-1, *grid_dims)
            elif vols.ndim == 5 and vols.shape[1] == 1:
                vols = vols.squeeze(1)

            if collapse_projection_axis:
                proj = vols.squeeze(dim=proj_axis) * depth
            else:
                proj = vols.sum(dim=proj_axis) * depth

            projections.append(proj)

        projections = torch.cat(projections, dim=0)
        return projections

    @staticmethod
    def _apply_resolved_mask(
        coords: torch.Tensor,
        atomic_numbers: torch.Tensor,
        bfactors: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if coords.ndim == 2:
            coords = coords[mask]
        elif coords.ndim == 3:
            coords = coords[:, mask, :]

        if atomic_numbers is not None:
            if atomic_numbers.ndim == 1:
                atomic_numbers = atomic_numbers[mask]
            elif atomic_numbers.ndim == 2:
                if atomic_numbers.shape[0] == mask.shape[0]:
                    atomic_numbers = atomic_numbers[mask]
                elif atomic_numbers.shape[1] == mask.shape[0]:
                    atomic_numbers = atomic_numbers[:, mask]

        if bfactors is not None:
            if bfactors.ndim == 1:
                bfactors = bfactors[mask]
            elif bfactors.ndim == 2:
                bfactors = bfactors[:, mask]
            elif bfactors.ndim == 3:
                bfactors = bfactors[:, mask, :]

        return coords, atomic_numbers, bfactors

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
        do_log = False
        if self._projection_log_every and self._projection_log_every > 0:
            self._projection_log_calls += 1
            do_log = (self._projection_log_calls % self._projection_log_every) == 0

        # Update blur sigma based on schedule (if provided).
        self._current_image_blur_sigma = self._scheduled_blur_sigma(time=time, step=step, i=i)

        for dataset_idx, (dataset, reference) in enumerate(zip(self.datasets, self.reference_structures)):
            aligned = self._align_to_reference(x_0_hat, reference)
            resolved_mask = reference.get("render_mask")

            gt = dataset["projections"]
            rendered = self._render_all_rotations(
                coords=aligned,
                atomic_numbers=reference["atomic_numbers"],
                bfactors=reference["bfactors"].unsqueeze(0).unsqueeze(-1),
                dataset=dataset,
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

        loss = torch.stack(losses).mean()
        self.last_loss_value = loss.detach().item()
        return loss, None, None

    def wandb_log(self, x_0_hat):
        log = {
            "loss": self.last_loss_value,  # keep for MultiLossFunction common_loss
            "cryoimage/loss": self.last_loss_value,
        }
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
