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


def _format_guidance_scale(scale: float) -> str:
    return format(scale, "g")


def _format_suffix(size: int, guidance_scale: float | None) -> str:
    suffix = f"n{size}"
    if guidance_scale is not None:
        suffix = f"{suffix}_g{_format_guidance_scale(guidance_scale)}"
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


def build_config_text(
    base_text: str,
    base_data: dict,
    size: int,
    guidance_scale: float | None,
) -> str:
    replacements: dict[str, str] = {}

    try:
        cryo_cfg = base_data["loss_function"]["cryoimage_loss_function"]
    except KeyError as exc:
        raise KeyError("Missing loss_function.cryoimage_loss_function in config") from exc

    for key in ("image_pt_files", "image_json_files"):
        paths = cryo_cfg.get(key, [])
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"{key} must be a non-empty list in base config")
        for path in paths:
            replacements[path] = _replace_path_size(path, size)

    general = base_data.get("general", {})
    suffix = _format_suffix(size, guidance_scale)
    output_folder = general.get("output_folder")
    if not output_folder:
        raise ValueError("general.output_folder is required in base config")
    replacements[output_folder] = os.path.join(output_folder, suffix)

    name = general.get("name")
    if name:
        replacements[name] = f"{name}_{suffix}"

    updated = _apply_replacements(base_text, replacements)

    if guidance_scale is not None:
        updated = _replace_guidance_scale(updated, guidance_scale)

    return updated


def run_ablation(
    base_config: Path,
    pdb_id: str,
    sizes: list[int],
    device: str,
    guidance_scales: list[float] | None,
    phenix_manager: str | None,
    dry_run: bool,
) -> None:
    base_text = base_config.read_text()
    base_data = yaml.safe_load(base_text)

    if guidance_scales:
        effective_guidance_scales: list[float | None] = guidance_scales
    else:
        effective_guidance_scales = [None]

    output_dir = Path("tmp/ablation_configs")
    output_dir.mkdir(parents=True, exist_ok=True)

    for size in sizes:
        for guidance_scale in effective_guidance_scales:
            config_text = build_config_text(
                base_text, base_data, size, guidance_scale
            )
            suffix = _format_suffix(size, guidance_scale)
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

    jobs = [
        (size, guidance_scale)
        for size in sizes
        for guidance_scale in effective_guidance_scales
    ]

    device_to_jobs: dict[str, list[tuple[int, float | None]]] = {
        device: [] for device in devices
    }
    for index, job in enumerate(jobs):
        device_to_jobs[devices[index % len(devices)]].append(job)

    def worker(device: str, jobs_for_device: list[tuple[int, float | None]]) -> None:
        env = os.environ.copy()
        effective_device = device
        if device.startswith("cuda"):
            gpu_id = device.split(":", 1)[1] if ":" in device else device
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            effective_device = "cuda:0"
        for size, guidance_scale in jobs_for_device:
            config_text = build_config_text(
                base_text, base_data, size, guidance_scale
            )
            suffix = _format_suffix(size, guidance_scale)
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
            phenix_manager=args.phenix_manager,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
