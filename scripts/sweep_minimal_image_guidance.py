#!/usr/bin/env python3
"""
Sweep a small, curated set of minimal_image_guidance configs across noise/blur,
running one job per GPU at a time.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SweepConfig:
    name: str
    args: list[str]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip() != ""]


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip() != ""]


def _configs() -> list[SweepConfig]:
    # Curated, promising configurations (not a full grid).
    return [
        SweepConfig(
            name="l1_topk",
            args=[
                "--loss-type",
                "l1",
                "--loss-topk",
                "32",
                "--max-rotations",
                "128",
                "--resample-rotations"
            ],
        ),
        SweepConfig(
            name="l1",
            args=[
                "--loss-type",
                "l1",
                "--max-rotations",
                "128",
                "--resample-rotations"
            ],
        ),
        SweepConfig(
            name="l1-cosine",
            args=[
                "--loss-type",
                "l1",
                "--cosine-lowpass-frac",
                "0.1",
                "--max-rotations",
                "128",
                "--resample-rotations"
            ],
        ),
        SweepConfig(
            name="fftlog",
            args=[
                "--loss-type",
                "fft_log_mag_mse",
                "--fft-bandpass-low",
                "0.02",
                "--fft-bandpass-high",
                "0.25",
                "--max-rotations",
                "512",
                "--resample-rotations"
            ],
        ),
        SweepConfig(
            name="ncc_topk",
            args=[
                "--loss-type",
                "ncc",
                "--loss-topk",
                "64",
                "--max-rotations",
                "512",
                "--resample-rotations"
            ],
        ),
    ]


def _iter_jobs(
    configs: Iterable[SweepConfig],
    noise_stds: list[float],
    blur_sigmas: list[float],
    base_cmd: list[str],
    extra_args: list[str],
) -> list[tuple[str, list[str]]]:
    jobs: list[tuple[str, list[str]]] = []
    for cfg in configs:
        for noise in noise_stds:
            for blur in blur_sigmas:
                cmd = base_cmd + cfg.args + ["--noise-std", f"{noise:g}"]
                if blur > 0:
                    cmd += ["--image-blur-sigma", f"{blur:g}"]
                cmd += extra_args
                label = f"{cfg.name}_noise{noise:g}_blur{blur:g}"
                jobs.append((label, cmd))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep curated minimal_image_guidance configs across noise/blur.")
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU indices (e.g., 0,1,2).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="tmp/ablation_configs/7T54_cryoimage_n100_g0.01.yaml",
        help="YAML config for minimal_image_guidance.py.",
    )
    parser.add_argument(
        "--images-pt",
        type=str,
        default="src/utils/em_images_loss_dataset_gen/output/smoketest/7t54_esp_projections_1000.pt",
        help="Path to projections .pt file.",
    )
    parser.add_argument(
        "--images-json",
        type=str,
        default="src/utils/em_images_loss_dataset_gen/output/smoketest/7t54_esp_projections_1000.json",
        help="Path to projections .json file.",
    )
    parser.add_argument(
        "--noise-stds",
        type=str,
        default="0.5,2,5",
        help="Comma-separated noise stds to sweep.",
    )
    parser.add_argument(
        "--blur-sigmas",
        type=str,
        default="1.0,3.0,5.0",
        help="Comma-separated Gaussian blur sigmas to sweep.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=200,
        help="Number of steps per run.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging for all runs.",
    )
    parser.add_argument(
        "--wandb-prefix",
        type=str,
        default=None,
        help="Prefix for wandb run names (e.g., Sweep0).",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default="0",
        help="Sweep identifier used when --wandb-prefix is not provided.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of jobs (for quick tests).",
    )
    args = parser.parse_args()

    gpus = _parse_csv_ints(args.gpus)
    noise_stds = _parse_csv_floats(args.noise_stds)
    blur_sigmas = _parse_csv_floats(args.blur_sigmas)

    base_cmd = [
        "python",
        "scripts/minimal_image_guidance.py",
        "--config",
        args.config,
        "--images-pt",
        args.images_pt,
        "--images-json",
        args.images_json,
        "--device",
        "cuda:0",
        "--n-steps",
        str(args.n_steps),
    ]
    extra_args: list[str] = []
    if args.no_wandb:
        extra_args.append("--no-wandb")
    if args.wandb_prefix:
        extra_args += ["--wandb-name-prefix", args.wandb_prefix]
    else:
        extra_args += ["--wandb-name-prefix", f"Sweep{args.sweep_id}"]

    jobs = _iter_jobs(_configs(), noise_stds, blur_sigmas, base_cmd, extra_args)
    if args.limit is not None:
        jobs = jobs[: args.limit]

    if args.dry_run:
        for label, cmd in jobs:
            print(label)
            print(" ".join(cmd))
        return

    if not jobs:
        print("No jobs to run.")
        return

    running: dict[int, subprocess.Popen] = {}
    queue = list(jobs)

    job_index = 0
    while queue or running:
        # Launch on free GPUs
        for gpu in gpus:
            if gpu in running:
                if running[gpu].poll() is None:
                    continue
                running.pop(gpu)
            if not queue:
                continue
            label, cmd = queue.pop(0)
            cmd = cmd + ["--wandb-name-suffix", f"{job_index:03d}"]
            job_index += 1
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"[GPU {gpu}] start: {label}")
            running[gpu] = subprocess.Popen(cmd, env=env)

        time.sleep(2.0)

    print("Sweep completed.")


if __name__ == "__main__":
    main()
