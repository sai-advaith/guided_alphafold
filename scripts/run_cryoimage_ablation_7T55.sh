#!/usr/bin/env bash
set -euo pipefail

python scripts/run_cryoimage_ablation.py \
  --base-config pipeline_configurations/cryoem_test_configs/image_loss/7T55_benchmark_cryoimage.yaml \
  --pdb-id 7T55 \
  "$@"
