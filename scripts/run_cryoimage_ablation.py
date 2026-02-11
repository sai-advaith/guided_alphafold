#!/usr/bin/env python3
"""Run cryoimage dataset-size ablations from a base config."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

DEFAULT_SIZES = [100, 1000]
DEFAULT_GUIDANCE_SCALES = [0.01, 0.02, 0.05, 0.1]
DEFAULT_SCHEDULE_TYPES = ["constant", "late_ramp", "linear", "sigmoid"]
DEFAULT_RAMP_START_FRACS = [0.5]
DEFAULT_SCHEDULE_SEGMENTS = [4]
DEFAULT_SIGMOID_KS = [6.0]
DEFAULT_START_GUIDANCE = [None]


def _format_guidance_scale(scale: float) -> str:
    return format(scale, "g")


def _format_suffix(
    size: int,
    guidance_scale: float | None,
    schedule_type: str | None,
    ramp_start_frac: float | None,
    n_segments: int | None,
    sigmoid_k: float | None,
    start_guidance_from: int | None,
) -> str:
    suffix = f"n{size}"
    if guidance_scale is not None:
        suffix = f"{suffix}_g{_format_guidance_scale(guidance_scale)}"
    if start_guidance_from is not None:
        suffix = f"{suffix}_start{start_guidance_from}"
    if schedule_type:
        schedule_type = schedule_type.lower()
        if schedule_type == "constant":
            suffix = f"{suffix}_const"
        elif schedule_type == "late_ramp" and ramp_start_frac is not None:
            suffix = f"{suffix}_late{ramp_start_frac:g}"
        elif schedule_type == "linear" and n_segments is not None:
            suffix = f"{suffix}_lin{n_segments}"
        elif schedule_type == "sigmoid" and sigmoid_k is not None:
            suffix = f"{suffix}_sigk{sigmoid_k:g}"
        else:
            suffix = f"{suffix}_{schedule_type}"
    return suffix


def _replace_path_size(path: str, size: int) -> str:
    """
    Replace the numeric suffix in *_esp_projections_<N>.(pt|json).
    """
    pattern = r"(_esp_projections_)(\d+)(\.pt|\.json)$"
    match = re.search(pattern, path)
    if not match:
        raise ValueError(f"Path does not match expected pattern: {path}")
    return re.sub(
        pattern,
        lambda match: f"{match.group(1)}{size}{match.group(3)}",
        path,
    )


def _apply_replacements(text: str, replacements: dict[str, str]) -> str:
    updated = text
    for old, new in replacements.items():
        if old not in updated:
            raise ValueError(f"Expected to find path in base config: {old}")
        updated = updated.replace(old, new)
    return updated


def _replace_guidance_scale(text: str, scale: float) -> str:
    pattern = r"(^\s*guidance_direction_scale_factor:\s*)([^#\n]*)(\s*#.*)?$"
    def _repl(match: re.Match[str]) -> str:
        comment = match.group(3) or ""
        if comment and not comment.startswith(" "):
            comment = f" {comment}"
        return f"{match.group(1)}{_format_guidance_scale(scale)}{comment}"

    updated, count = re.subn(pattern, _repl, text, flags=re.MULTILINE)
    if count == 0:
        raise ValueError(
            "guidance_direction_scale_factor not found in base config text"
        )
    if count > 1:
        raise ValueError(
            "Multiple guidance_direction_scale_factor entries found in base config text"
        )
    return updated


def _build_guidance_schedule(
    schedule_type: str,
    scale: float,
    n_steps: int,
    ramp_start_frac: float,
    n_segments: int,
    sigmoid_k: float,
) -> list[dict] | None:
    schedule_type = schedule_type.lower()
    if schedule_type == "constant":
        return None
    if schedule_type == "late_ramp":
        start = max(0, min(n_steps, int(round(n_steps * ramp_start_frac))))
        return [
            {"scale_factor": 0.0, "step_range": [0, start]},
            {"scale_factor": scale, "step_range": [start, n_steps]},
        ]
    if schedule_type == "linear":
        segments = max(1, n_segments)
        per = n_steps // segments
        schedule = []
        for i in range(segments):
            start = i * per
            end = n_steps if i == segments - 1 else (i + 1) * per
            frac = (i + 1) / segments
            schedule.append({"scale_factor": scale * frac, "step_range": [start, end]})
        return schedule
    if schedule_type == "sigmoid":
        import math
        segments = max(2, n_segments)
        per = n_steps // segments
        schedule = []
        for i in range(segments):
            start = i * per
            end = n_steps if i == segments - 1 else (i + 1) * per
            x = (i + 0.5) / segments
            frac = 1.0 / (1.0 + math.exp(-sigmoid_k * (x - 0.5)))
            schedule.append({"scale_factor": scale * frac, "step_range": [start, end]})
        return schedule
    raise ValueError(f"Unknown schedule_type: {schedule_type}")


def build_config_text(
    base_text: str,
    base_data: dict,
    size: int,
    guidance_scale: float | None,
    schedule_type: str | None,
    ramp_start_frac: float,
    n_segments: int,
    sigmoid_k: float,
    start_guidance_from: int | None,
) -> str:
    # We build a fresh dict (no comment preservation needed for tmp configs).
    data = yaml.safe_load(base_text)

    try:
        cryo_cfg = data["loss_function"]["cryoimage_loss_function"]
    except KeyError as exc:
        raise KeyError("Missing loss_function.cryoimage_loss_function in config") from exc

    for key in ("image_pt_files", "image_json_files"):
        paths = cryo_cfg.get(key, [])
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"{key} must be a non-empty list in base config")
        cryo_cfg[key] = [_replace_path_size(path, size) for path in paths]

    general = data.get("general", {})
    suffix = _format_suffix(
        size,
        guidance_scale,
        schedule_type,
        ramp_start_frac,
        n_segments,
        sigmoid_k,
        start_guidance_from,
    )
    output_folder = general.get("output_folder")
    if not output_folder:
        raise ValueError("general.output_folder is required in base config")
    general["output_folder"] = os.path.join(output_folder, suffix)

    name = general.get("name")
    if name:
        general["name"] = f"{name}_{suffix}"

    guidance = data.get("diffusion_process", {}).get("guidance", {})
    if guidance_scale is not None:
        guidance["guidance_direction_scale_factor"] = guidance_scale
    if start_guidance_from is not None:
        general["denoiser_time_index"] = int(start_guidance_from)
    if schedule_type is not None:
        n_steps = int(data.get("model_manager", {}).get("diffusion_N", 200))
        schedule = _build_guidance_schedule(
            schedule_type=schedule_type,
            scale=guidance_scale or guidance.get("guidance_direction_scale_factor", 1.0),
            n_steps=n_steps,
            ramp_start_frac=ramp_start_frac,
            n_segments=n_segments,
            sigmoid_k=sigmoid_k,
        )
        guidance["guidance_direction_scale_factors"] = schedule

    return yaml.safe_dump(data, sort_keys=False)


def run_ablation(
    base_config: Path,
    pdb_id: str,
    sizes: list[int],
    device: str,
    guidance_scales: list[float] | None,
    schedule_types: list[str] | None,
    ramp_start_fracs: list[float],
    n_segments_list: list[int],
    sigmoid_ks: list[float],
    start_guidance_from_list: list[int | None],
    phenix_manager: str | None,
    dry_run: bool,
) -> None:
    base_text = base_config.read_text()
    base_data = yaml.safe_load(base_text)

    if guidance_scales:
        effective_guidance_scales: list[float | None] = guidance_scales
    else:
        effective_guidance_scales = [None]
    if schedule_types:
        effective_schedule_types: list[str | None] = schedule_types
    else:
        effective_schedule_types = [None]

    output_dir = Path("tmp/ablation_configs")
    output_dir.mkdir(parents=True, exist_ok=True)

    for size in sizes:
        for guidance_scale in effective_guidance_scales:
            for schedule_type in effective_schedule_types:
                for start_guidance_from in start_guidance_from_list:
                    schedule_type_norm = (schedule_type or "constant").lower()
                    if schedule_type_norm == "late_ramp":
                        param_grid = [(r, n_segments_list[0], sigmoid_ks[0]) for r in ramp_start_fracs]
                    elif schedule_type_norm == "linear":
                        param_grid = [(ramp_start_fracs[0], s, sigmoid_ks[0]) for s in n_segments_list]
                    elif schedule_type_norm == "sigmoid":
                        param_grid = [(ramp_start_fracs[0], n_segments_list[0], k) for k in sigmoid_ks]
                    else:
                        param_grid = [(ramp_start_fracs[0], n_segments_list[0], sigmoid_ks[0])]

                for ramp_start_frac, n_segments, sigmoid_k in param_grid:
                    config_text = build_config_text(
                        base_text,
                        base_data,
                        size,
                        guidance_scale,
                        schedule_type,
                        ramp_start_frac,
                        n_segments,
                        sigmoid_k,
                        start_guidance_from,
                    )
                    suffix = _format_suffix(
                        size,
                        guidance_scale,
                        schedule_type,
                        ramp_start_frac,
                        n_segments,
                        sigmoid_k,
                        start_guidance_from,
                    )
                    config_path = output_dir / f"{pdb_id}_cryoimage_{suffix}.yaml"
                    config_path.write_text(config_text)

                    cmd = [
                        sys.executable,
                        "experiment_manager.py",
                        "--configuration_file",
                        str(config_path),
                        "--device",
                        device,
                    ]
                    if phenix_manager:
                        cmd.extend(["--phenix_manager", phenix_manager])

                    print(" ".join(cmd))
                    if not dry_run:
                        subprocess.run(cmd, check=True)


def run_ablation_parallel(
    base_config: Path,
    pdb_id: str,
    sizes: list[int],
    devices: list[str],
    guidance_scales: list[float] | None,
    schedule_types: list[str] | None,
    ramp_start_fracs: list[float],
    n_segments_list: list[int],
    sigmoid_ks: list[float],
    start_guidance_from_list: list[int | None],
    phenix_manager: str | None,
    dry_run: bool,
) -> None:
    if not devices:
        raise ValueError("At least one device is required for parallel runs")

    base_text = base_config.read_text()
    base_data = yaml.safe_load(base_text)

    output_dir = Path("tmp/ablation_configs")
    output_dir.mkdir(parents=True, exist_ok=True)

    if guidance_scales:
        effective_guidance_scales: list[float | None] = guidance_scales
    else:
        effective_guidance_scales = [None]
    if schedule_types:
        effective_schedule_types: list[str | None] = schedule_types
    else:
        effective_schedule_types = [None]

    jobs = []
    for size in sizes:
        for guidance_scale in effective_guidance_scales:
            for schedule_type in effective_schedule_types:
                for start_guidance_from in start_guidance_from_list:
                    schedule_type_norm = (schedule_type or "constant").lower()
                    if schedule_type_norm == "late_ramp":
                        param_grid = [(r, n_segments_list[0], sigmoid_ks[0]) for r in ramp_start_fracs]
                    elif schedule_type_norm == "linear":
                        param_grid = [(ramp_start_fracs[0], s, sigmoid_ks[0]) for s in n_segments_list]
                    elif schedule_type_norm == "sigmoid":
                        param_grid = [(ramp_start_fracs[0], n_segments_list[0], k) for k in sigmoid_ks]
                    else:
                        param_grid = [(ramp_start_fracs[0], n_segments_list[0], sigmoid_ks[0])]

                    for ramp_start_frac, n_segments, sigmoid_k in param_grid:
                        jobs.append((size, guidance_scale, schedule_type, ramp_start_frac, n_segments, sigmoid_k, start_guidance_from))

    device_to_jobs: dict[str, list[tuple[int, float | None, str | None, float, int, float, int | None]]] = {
        device: [] for device in devices
    }
    for index, job in enumerate(jobs):
        device_to_jobs[devices[index % len(devices)]].append(job)

    def worker(device: str, jobs_for_device: list[tuple[int, float | None, str | None, float, int, float, int | None]]) -> None:
        env = os.environ.copy()
        effective_device = device
        if device.startswith("cuda"):
            gpu_id = device.split(":", 1)[1] if ":" in device else device
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            effective_device = "cuda:0"
        for size, guidance_scale, schedule_type, ramp_start_frac, n_segments, sigmoid_k, start_guidance_from in jobs_for_device:
            config_text = build_config_text(
                base_text,
                base_data,
                size,
                guidance_scale,
                schedule_type,
                ramp_start_frac,
                n_segments,
                sigmoid_k,
                start_guidance_from,
            )
            suffix = _format_suffix(
                size,
                guidance_scale,
                schedule_type,
                ramp_start_frac,
                n_segments,
                sigmoid_k,
                start_guidance_from,
            )
            config_path = output_dir / f"{pdb_id}_cryoimage_{suffix}.yaml"
            config_path.write_text(config_text)

            cmd = [
                sys.executable,
                "experiment_manager.py",
                "--configuration_file",
                str(config_path),
                "--device",
                effective_device,
            ]
            if phenix_manager:
                cmd.extend(["--phenix_manager", phenix_manager])

            print(" ".join(cmd))
            if not dry_run:
                subprocess.run(cmd, check=True, env=env)

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(worker, device, jobs_for_device)
            for device, jobs_for_device in device_to_jobs.items()
            if jobs_for_device
        ]
        for future in as_completed(futures):
            future.result()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        required=True,
        help="Path to the base cryoimage config YAML",
    )
    parser.add_argument("--pdb-id", required=True, help="PDB ID label (e.g., 7T55)")
    parser.add_argument(
        "--sizes",
        nargs="*",
        type=int,
        default=DEFAULT_SIZES,
        help=f"Dataset sizes to sweep (default: {DEFAULT_SIZES})",
    )
    parser.add_argument(
        "--schedule-types",
        nargs="*",
        default=DEFAULT_SCHEDULE_TYPES,
        help=f"Guidance schedule types to sweep (default: {DEFAULT_SCHEDULE_TYPES})",
    )
    parser.add_argument(
        "--schedule-ramp-start-fracs",
        nargs="*",
        type=float,
        default=DEFAULT_RAMP_START_FRACS,
        help="Start fractions for late_ramp schedule (0-1).",
    )
    parser.add_argument(
        "--schedule-segments",
        nargs="*",
        type=int,
        default=DEFAULT_SCHEDULE_SEGMENTS,
        help="Number of segments for linear/sigmoid schedules.",
    )
    parser.add_argument(
        "--schedule-sigmoid-ks",
        nargs="*",
        type=float,
        default=DEFAULT_SIGMOID_KS,
        help="Sigmoid steepness values for sigmoid schedule.",
    )
    parser.add_argument(
        "--start-guidance-from",
        nargs="*",
        type=int,
        default=DEFAULT_START_GUIDANCE,
        help="Start guidance after these diffusion steps (uses general.denoiser_time_index).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--devices",
        nargs="*",
        default=None,
        help="Optional list of devices to run in parallel (e.g., cuda:0 cuda:1)",
    )
    parser.add_argument("--phenix-manager", default=None)
    parser.add_argument(
        "--guidance-direction-scale-factors",
        nargs="*",
        type=float,
        default=DEFAULT_GUIDANCE_SCALES,
        help=(
            "Guidance_direction_scale_factor values to sweep "
            f"(default: {DEFAULT_GUIDANCE_SCALES}). Use --guidance-direction-scale-factors "
            "to override."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write configs and print commands",
    )
    args = parser.parse_args()

    if args.devices:
        run_ablation_parallel(
            base_config=Path(args.base_config),
            pdb_id=args.pdb_id,
            sizes=args.sizes,
            devices=args.devices,
            guidance_scales=args.guidance_direction_scale_factors,
            schedule_types=args.schedule_types,
            ramp_start_fracs=args.schedule_ramp_start_fracs,
            n_segments_list=args.schedule_segments,
            sigmoid_ks=args.schedule_sigmoid_ks,
            start_guidance_from_list=args.start_guidance_from,
            phenix_manager=args.phenix_manager,
            dry_run=args.dry_run,
        )
    else:
        run_ablation(
            base_config=Path(args.base_config),
            pdb_id=args.pdb_id,
            sizes=args.sizes,
            device=args.device,
            guidance_scales=args.guidance_direction_scale_factors,
            schedule_types=args.schedule_types,
            ramp_start_fracs=args.schedule_ramp_start_fracs,
            n_segments_list=args.schedule_segments,
            sigmoid_ks=args.schedule_sigmoid_ks,
            start_guidance_from_list=args.start_guidance_from,
            phenix_manager=args.phenix_manager,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
