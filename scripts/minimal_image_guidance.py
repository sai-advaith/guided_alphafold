#!/usr/bin/env python3
"""
Minimal wrapper around CryoEM_Images_GuidanceLossFunction.

Loads x0 from a PDB, makes a noised x1, computes image loss/grad,
and checks whether a single gradient step moves x1 toward x0.

Defaults are set to the 7T54 smoketest projections.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path
import math

import torch
import yaml
import torch.nn.functional as F
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Avoid importing src.losses.__init__ (it pulls in heavy deps). Load the module directly.
SRC_DIR = REPO_ROOT / "src"
LOSSES_DIR = SRC_DIR / "losses"

src_pkg = types.ModuleType("src")
src_pkg.__path__ = [str(SRC_DIR)]
sys.modules.setdefault("src", src_pkg)

losses_pkg = types.ModuleType("src.losses")
losses_pkg.__path__ = [str(LOSSES_DIR)]
sys.modules.setdefault("src.losses", losses_pkg)

em_images_path = LOSSES_DIR / "em_images_loss_function.py"
spec = importlib.util.spec_from_file_location(
    "src.losses.em_images_loss_function", em_images_path
)
em_images_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = em_images_module
assert spec.loader is not None
spec.loader.exec_module(em_images_module)

CryoEM_Images_GuidanceLossFunction = em_images_module.CryoEM_Images_GuidanceLossFunction

from src.utils.io import load_pdb_atom_locations_full
from src.utils.process_pipeline_inputs.preprocess_nmr_inputs import get_amino_acid_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal cryoimage guidance sanity check.")
    parser.add_argument(
        "--pdb",
        type=str,
        default="pipeline_inputs/pdbs/7T54/7t54.pdb",
        help="Path to PDB file for x0.",
    )
    parser.add_argument(
        "--images-pt",
        type=str,
        default="src/utils/em_images_loss_dataset_gen/output/smoketest/7T54_esp_projections_1000.pt",
        help="Path to projections .pt file.",
    )
    parser.add_argument(
        "--images-json",
        type=str,
        default="src/utils/em_images_loss_dataset_gen/output/smoketest/7T54_esp_projections_1000.json",
        help="Path to projections .json file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional pipeline YAML to pull sequences/chains from.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=5.0, help="Gaussian noise (A).")
    parser.add_argument("--step-size", type=float, default=None, help="Gradient step size.")
    parser.add_argument("--n-steps", type=int, default=1, help="Number of gradient steps.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Guidance scale factor.")
    parser.add_argument(
        "--loss-type",
        type=str,
        default=None,
        help=(
            "Projection loss type: mse|l1|smooth_l1|fft_mag_mse|fft_log_mag_mse|ncc|ot. "
            "Defaults to YAML loss_type if present."
        ),
    )
    parser.add_argument(
        "--loss-topk",
        type=int,
        default=None,
        help="Use only the K most lossy projections per step.",
    )
    parser.add_argument(
        "--fft-log-eps",
        type=float,
        default=None,
        help="Epsilon for log FFT magnitude loss.",
    )
    parser.add_argument(
        "--fft-bandpass-low",
        type=float,
        default=None,
        help="Low cutoff (0..0.5) for FFT bandpass mask.",
    )
    parser.add_argument(
        "--fft-bandpass-high",
        type=float,
        default=None,
        help="High cutoff (0..0.5) for FFT bandpass mask.",
    )
    parser.add_argument(
        "--ncc-eps",
        type=float,
        default=None,
        help="Epsilon for NCC denominator stability.",
    )
    parser.add_argument(
        "--ot-p",
        type=int,
        default=None,
        help="Sinkhorn OT p-norm.",
    )
    parser.add_argument(
        "--ot-blur",
        type=float,
        default=None,
        help="Sinkhorn OT blur parameter.",
    )
    parser.add_argument(
        "--ot-scaling",
        type=float,
        default=None,
        help="Sinkhorn OT scaling parameter.",
    )
    parser.add_argument(
        "--ot-reach",
        type=float,
        default=None,
        help="Sinkhorn OT reach parameter.",
    )
    parser.add_argument(
        "--ot-backend",
        type=str,
        default=None,
        help="Sinkhorn OT backend (e.g., online, tensorized).",
    )
    parser.add_argument(
        "--ot-debias",
        action="store_true",
        help="Enable OT debiasing.",
    )
    parser.add_argument(
        "--ot-eps",
        type=float,
        default=None,
        help="Epsilon for OT weight normalization.",
    )
    parser.add_argument(
        "--ot-downsample",
        type=int,
        default=None,
        help="Downsample factor for OT loss (average pooling).",
    )
    parser.add_argument(
        "--cosine-lowpass-frac",
        type=float,
        default=None,
        help="Keep this fraction of lowest cosine frequencies (0..1). Applied to noise and gradients.",
    )
    parser.add_argument(
        "--cosine-lowpass-k",
        type=int,
        default=None,
        help="Keep this many lowest cosine frequencies. Applied to noise and gradients.",
    )
    parser.add_argument(
        "--normalize-projections",
        action="store_true",
        help="Z-score normalize rendered/GT projections per rotation before loss.",
    )
    parser.add_argument(
        "--max-rotations",
        type=int,
        default=0,
        help="Limit number of projections/rotations used (<=0 means all).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Rotations per chunk for tqdm/progress. <=0 means one chunk.",
    )
    parser.add_argument(
        "--normalize-gradients",
        action="store_true",
        help="Normalize gradients similar to guided diffusion behavior.",
    )
    parser.add_argument(
        "--resample-rotations",
        action="store_true",
        help="Resample a random subset of rotations each step.",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Skip alignment to reference (debug only).",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging.",
    )
    return parser.parse_args()


def _load_config_sequences(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sequences = cfg["protein"]["sequences"]
    for entry in sequences:
        entry.setdefault("sequence_type", "proteinChain")
    chains_to_read = cfg["protein"].get("chains_to_use")
    should_align_to_chains = cfg["protein"].get("should_align_to_chains")
    return cfg, sequences, chains_to_read, should_align_to_chains


def _fallback_sequences_from_pdb(pdb_path: str):
    seq = get_amino_acid_sequence(pdb_path)
    sequences = [
        {
            "sequence": seq,
            "count": 1,
            "sequence_type": "proteinChain",
            "maps_to": ["A"],
        }
    ]
    return None, sequences, ["A"], None


def _masked_rmsd(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    if a.ndim == 2:
        a = a.unsqueeze(0)
    if b.ndim == 2:
        b = b.unsqueeze(0)
    diff = a[:, mask, :] - b[:, mask, :]
    return torch.sqrt(torch.mean(diff ** 2)).item()

def _masked_stats(t: torch.Tensor, mask: torch.Tensor) -> dict:
    if t.ndim == 3:
        t = t[:, mask, :]
    else:
        t = t[mask]
    return {
        "norm": float(t.norm().item()),
        "mean": float(t.mean().item()),
        "max_abs": float(t.abs().max().item()),
    }

def _mean_per_atom_cosine(grad: torch.Tensor, delta: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute mean cosine similarity per atom between grad and delta.
    grad/delta: [B, N, 3]
    mask: [N]
    """
    if grad.ndim == 2:
        grad = grad.unsqueeze(0)
    if delta.ndim == 2:
        delta = delta.unsqueeze(0)
    grad = grad[:, mask, :]
    delta = delta[:, mask, :]
    grad_norm = grad.norm(dim=-1).clamp_min(eps)
    delta_norm = delta.norm(dim=-1).clamp_min(eps)
    cos = (grad * delta).sum(dim=-1) / (grad_norm * delta_norm)
    return float(cos.mean().item())

def _cosine_lowpass(
    x: torch.Tensor,
    keep_frac: float | None,
    keep_k: int | None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if keep_frac is None and keep_k is None:
        return x
    if x.ndim == 2:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    if mask is not None:
        x = x * mask.unsqueeze(0).unsqueeze(-1)

    # Apply along atom dimension using even extension (cosine transform low-pass)
    x_t = x.transpose(1, 2)  # [B, 3, N]
    v = torch.cat([x_t, x_t.flip(-1)], dim=-1)
    V = torch.fft.rfft(v, dim=-1)
    kmax = V.shape[-1]
    if keep_k is None:
        keep_k = max(1, int(math.ceil(kmax * float(keep_frac))))
    keep_k = min(max(1, int(keep_k)), kmax)
    V[..., keep_k:] = 0
    v_filtered = torch.fft.irfft(V, n=v.shape[-1], dim=-1)
    x_filtered = v_filtered[..., : x_t.shape[-1]].transpose(1, 2)

    if mask is not None:
        x_filtered = x_filtered * mask.unsqueeze(0).unsqueeze(-1)
    if squeeze:
        x_filtered = x_filtered.squeeze(0)
    return x_filtered

def _progress(iterable, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=False)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    pdb_path = Path(args.pdb)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    if args.config:
        cfg, sequences, chains_to_read, should_align_to_chains = _load_config_sequences(args.config)
    else:
        cfg, sequences, chains_to_read, should_align_to_chains = _fallback_sequences_from_pdb(str(pdb_path))

    coords, mask, bfactors, elements = load_pdb_atom_locations_full(
        pdb_file=str(pdb_path),
        full_sequences_dict=sequences,
        chains_to_read=chains_to_read,
        return_bfacs=True,
        return_elements=True,
        return_mask=True,
        device=torch.device(args.device),
    )

    mask = mask.to(dtype=torch.bool)
    coords = coords.to(dtype=torch.float32)

    loss_type = args.loss_type
    normalize_projections = args.normalize_projections
    fft_log_eps = args.fft_log_eps
    fft_bandpass_low = args.fft_bandpass_low
    fft_bandpass_high = args.fft_bandpass_high
    ncc_eps = args.ncc_eps
    loss_topk = args.loss_topk
    ot_p = args.ot_p
    ot_blur = args.ot_blur
    ot_scaling = args.ot_scaling
    ot_reach = args.ot_reach
    ot_backend = args.ot_backend
    ot_debias = args.ot_debias
    ot_eps = args.ot_eps
    ot_downsample = args.ot_downsample
    cosine_lowpass_frac = args.cosine_lowpass_frac
    cosine_lowpass_k = args.cosine_lowpass_k
    if cfg is not None:
        cryo_cfg = cfg.get("loss_function", {}).get("cryoimage_loss_function", {})
        if loss_type is None:
            loss_type = cryo_cfg.get("loss_type", "mse")
        if loss_topk is None:
            loss_topk = cryo_cfg.get("loss_topk")
        if fft_log_eps is None:
            fft_log_eps = cryo_cfg.get("fft_log_eps")
        if fft_bandpass_low is None:
            fft_bandpass_low = cryo_cfg.get("fft_bandpass_low")
        if fft_bandpass_high is None:
            fft_bandpass_high = cryo_cfg.get("fft_bandpass_high")
        if ncc_eps is None:
            ncc_eps = cryo_cfg.get("ncc_eps")
        if ot_p is None:
            ot_p = cryo_cfg.get("ot_p")
        if ot_blur is None:
            ot_blur = cryo_cfg.get("ot_blur")
        if ot_scaling is None:
            ot_scaling = cryo_cfg.get("ot_scaling")
        if ot_reach is None:
            ot_reach = cryo_cfg.get("ot_reach")
        if ot_backend is None:
            ot_backend = cryo_cfg.get("ot_backend")
        if not ot_debias:
            ot_debias = bool(cryo_cfg.get("ot_debias", False))
        if ot_eps is None:
            ot_eps = cryo_cfg.get("ot_eps")
        if ot_downsample is None:
            ot_downsample = cryo_cfg.get("ot_downsample")
        if cosine_lowpass_frac is None:
            cosine_lowpass_frac = cryo_cfg.get("cosine_lowpass_frac")
        if cosine_lowpass_k is None:
            cosine_lowpass_k = cryo_cfg.get("cosine_lowpass_k")

    if loss_type is None:
        loss_type = "mse"
    if fft_log_eps is None:
        fft_log_eps = 1e-6
    if ncc_eps is None:
        ncc_eps = 1e-6
    if ot_p is None:
        ot_p = 1
    if ot_blur is None:
        ot_blur = 0.5
    if ot_scaling is None:
        ot_scaling = 0.9
    if ot_backend is None:
        ot_backend = "online"
    if ot_eps is None:
        ot_eps = 1e-6

    noise = torch.randn_like(coords) * args.noise_std
    # if cosine_lowpass_frac is not None or cosine_lowpass_k is not None:
    #     noise = _cosine_lowpass(
    #         noise,
    #         keep_frac=cosine_lowpass_frac,
    #         keep_k=cosine_lowpass_k,
    #         mask=mask,
    #     )
    coords_noised = coords + noise * mask.unsqueeze(-1)
    x1 = coords_noised.unsqueeze(0).requires_grad_(True)
    x0 = coords.unsqueeze(0)

    loss_fn = CryoEM_Images_GuidanceLossFunction(
        image_pt_files=[args.images_pt],
        image_json_files=[args.images_json],
        reference_pdbs=[str(pdb_path)],
        mask=mask,
        sequences_dictionary=sequences,
        chains_to_read=chains_to_read,
        device=args.device,
        should_align_to_chains=should_align_to_chains,
        atom_batch_size=1024,
        loss_reduction="mean",
        loss_type=loss_type,
        use_checkpointing=False,
        log_projection_every=0,
        log_projection_pairs=0,
        max_rotations_per_batch=None,
        use_resolved_atoms_only=True,
        normalize_projections=normalize_projections,
        fft_log_eps=fft_log_eps,
        fft_bandpass_low=fft_bandpass_low,
        fft_bandpass_high=fft_bandpass_high,
        ncc_eps=ncc_eps,
        ot_p=ot_p,
        ot_blur=ot_blur,
        ot_scaling=ot_scaling,
        ot_reach=ot_reach,
        ot_backend=ot_backend,
        ot_debias=ot_debias,
        ot_eps=ot_eps,
        ot_downsample=ot_downsample,
        loss_topk=loss_topk,
    )

    guidance_scale = args.guidance_scale
    step_size = args.step_size
    normalize_gradients = args.normalize_gradients
    if cfg is not None:
        guidance_cfg = cfg.get("diffusion_process", {}).get("guidance", {})
        if guidance_scale is None:
            guidance_scale = guidance_cfg.get("guidance_direction_scale_factor", 1.0)
        if step_size is None:
            step_size = guidance_cfg.get("step_size", 1.0)
        if not normalize_gradients:
            normalize_gradients = bool(guidance_cfg.get("normalize_gradients", False))
    if guidance_scale is None:
        guidance_scale = 1.0
    if step_size is None:
        step_size = 1.0

    if args.max_rotations and args.max_rotations > 0:
        dataset = loss_fn.datasets[0]
        total_rots = int(dataset["rotations"].shape[0])
        use_rots = min(args.max_rotations, total_rots)
        if use_rots < total_rots:
            dataset["rotations"] = dataset["rotations"][:use_rots]
            dataset["projections"] = dataset["projections"][:use_rots]
        print(f"Using {use_rots}/{total_rots} rotations")
        sys.stdout.flush()

    dataset = loss_fn.datasets[0]
    reference = loss_fn.reference_structures[0]
    total_rots = int(dataset["rotations"].shape[0])
    chunk_size = args.chunk_size if args.chunk_size and args.chunk_size > 0 else total_rots

    def _get_rotation_subset():
        if args.max_rotations and args.max_rotations > 0:
            if args.resample_rotations:
                perm = torch.randperm(total_rots, device=dataset["rotations"].device)
                idx = perm[:args.max_rotations]
                return dataset["rotations"][idx], dataset["projections"][idx]
            return dataset["rotations"][:args.max_rotations], dataset["projections"][:args.max_rotations]
        return dataset["rotations"], dataset["projections"]

    def compute_loss_with_tqdm(x, step_label: str):
        aligned = x if args.no_align else loss_fn._align_to_reference(x, reference)
        rotations, projections = _get_rotation_subset()
        local_rots = int(rotations.shape[0])
        total_loss = 0.0
        per_losses = []
        resolved_mask = reference.get("resolved_mask") if loss_fn.use_resolved_atoms_only else None
        for start in _progress(
            range(0, local_rots, chunk_size),
            desc=f"render+loss {step_label}",
        ):
            end = min(start + chunk_size, local_rots)
            chunk = dict(dataset)
            chunk["rotations"] = rotations[start:end]
            chunk["projections"] = projections[start:end]
            rendered = loss_fn._render_all_rotations(
                coords=aligned,
                atomic_numbers=reference["atomic_numbers"],
                bfactors=reference["bfactors"].unsqueeze(0).unsqueeze(-1),
                dataset=chunk,
                resolved_mask=resolved_mask,
            )
            if loss_fn.loss_topk is not None and loss_fn.loss_topk > 0:
                per = loss_fn._projection_loss_per_rotation(rendered, chunk["projections"])
                per_losses.append(per)
            else:
                loss_chunk = loss_fn._projection_loss(rendered, chunk["projections"])
                if loss_fn.loss_reduction == "mean":
                    weight = (end - start) / local_rots
                else:
                    weight = 1.0
                total_loss = total_loss + loss_chunk * weight
        if loss_fn.loss_topk is not None and loss_fn.loss_topk > 0:
            if not per_losses:
                raise RuntimeError("No projections available for top-k loss.")
            per_all = torch.cat(per_losses, dim=0)
            total_loss = loss_fn._reduce_topk(per_all)
        loss_fn.last_loss_value = float(total_loss.detach().item())
        return total_loss

    use_wandb = not args.no_wandb
    wandb_run = None
    if use_wandb and cfg is not None and cfg.get("wandb"):
        try:
            import wandb

            wandb_cfg = cfg["wandb"]
            mode = wandb_cfg.get("mode", "online")
            project = wandb_cfg.get("project", "guided-alphafold")
            run_name = cfg.get("general", {}).get("name", "cryoimage_minimal")
            login_key = wandb_cfg.get("login_key")
            if login_key and mode != "disabled":
                wandb.login(key=login_key)
            loss_label = loss_type
            if loss_type in ("fft_mag_mse", "fft_magnitude_mse", "fft_mag"):
                loss_label = "fft"
            elif loss_type in ("fft_log_mag_mse", "fft_log_mag", "fft_log_magnitude_mse"):
                loss_label = "fftlog"
            elif loss_type in ("normalized_cross_correlation",):
                loss_label = "ncc"
            elif loss_type in ("optimal_transport", "sinkhorn"):
                loss_label = "ot"

            tag_parts = [loss_label]
            if loss_topk is not None and loss_topk > 0:
                tag_parts.append(f"topk{loss_topk}")
            if args.max_rotations and args.max_rotations > 0:
                tag_parts.append(f"rots{args.max_rotations}")
            if normalize_projections:
                tag_parts.append("normp")
            if normalize_gradients:
                tag_parts.append("normg")
            if cosine_lowpass_frac is not None:
                tag_parts.append(f"lpf{cosine_lowpass_frac:g}")

            run_name_full = f"LossSuite_{'_'.join(tag_parts)}"
            wandb_run = wandb.init(
                project=project,
                name=run_name_full,
                mode=mode,
                config={
                    "noise_std": args.noise_std,
                    "step_size": step_size,
                    "guidance_scale": guidance_scale,
                    "normalize_gradients": normalize_gradients,
                    "n_steps": args.n_steps,
                    "max_rotations": args.max_rotations,
                    "chunk_size": args.chunk_size,
                    "resample_rotations": args.resample_rotations,
                    "loss_topk": loss_topk,
                    "cosine_lowpass_frac": cosine_lowpass_frac,
                    "cosine_lowpass_k": cosine_lowpass_k,
                },
            )
        except Exception:
            wandb_run = None

    loss = None
    for step_idx in range(args.n_steps):
        print(f"Computing loss/grad (step {step_idx})...")
        sys.stdout.flush()
        loss = compute_loss_with_tqdm(x1, f"step{step_idx}")
        loss.backward()

        with torch.no_grad():
            grad_stats = _masked_stats(x1.grad, mask)
            delta = (x0 - x1)
            delta_masked = delta[:, mask, :].reshape(-1)
            grad_masked = (-x1.grad)[:, mask, :].reshape(-1)
            cos = float(F.cosine_similarity(grad_masked, delta_masked, dim=0).item())
            mean_cos = _mean_per_atom_cosine(-x1.grad, delta, mask)
            rmsd_val = _masked_rmsd(x1, x0, mask)
            print(
                "Grad stats (masked): "
                f"norm={grad_stats['norm']:.6f} "
                f"mean={grad_stats['mean']:.6e} "
                f"max_abs={grad_stats['max_abs']:.6f}"
            )
            print(f"cos(-grad, x0-x1) = {cos:.6f} (global)")
            print(f"mean per-atom cos (-grad, x0-x1) = {mean_cos:.6f}")
            print(f"RMSD: {rmsd_val:.6f} A")
            sys.stdout.flush()

            grad_update = x1.grad
            if cosine_lowpass_frac is not None or cosine_lowpass_k is not None:
                grad_update = _cosine_lowpass(
                    grad_update,
                    keep_frac=cosine_lowpass_frac,
                    keep_k=cosine_lowpass_k,
                    mask=mask,
                )
            if normalize_gradients:
                grad_norm = grad_stats["norm"]
                delta_norm = float(delta_masked.norm().item())
                if grad_norm > 0:
                    grad_update = grad_update * (delta_norm / (grad_norm + 1e-8)) * guidance_scale
            else:
                grad_update = grad_update * guidance_scale

            x1 = (x1 - step_size * grad_update).detach().clone().requires_grad_(True)

            if wandb_run is not None:
                wandb.log(
                    {
                        "loss": float(loss.item()),
                        "rmsd": rmsd_val,
                        "grad_norm": grad_stats["norm"],
                        "grad_max_abs": grad_stats["max_abs"],
                        "cos_grad_delta": cos,
                        "mean_atom_cos_grad_delta": mean_cos,
                        "step_idx": step_idx,
                    }
                )

    loss_after = loss

    print(f"Loss before step: {loss.item():.6f}")
    print(f"Loss after  step: {loss_after.item():.6f}")
    print(f"RMSD after  step: {_masked_rmsd(x1, x0, mask):.6f} A")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
