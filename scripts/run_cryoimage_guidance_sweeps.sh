#!/usr/bin/env bash
set -euo pipefail

# Single entrypoint to run both sweeps.
# 1) guidance-scale sweep (constant schedule)
# 2) schedule-parameter sweep (late_ramp/linear/sigmoid with 4 params each)

BASE_CONFIG="pipeline_configurations/cryoem_test_configs/image_loss/7T54_benchmark_cryoimage.yaml"
PDB_ID="7T54"
SIZES=(512)
DEVICES=(cuda:0 cuda:1 cuda:2 cuda:3)
START_GUIDANCE=(0 50 100)
PHENIX_MANAGER="/mnt/nfs/clustersw/shared/phenix/1.21.2-5419/build/setpaths.sh"

# Sweep A: guidance scale only (constant schedule)
GUIDANCE_SCALES=(0.002 0.005 0.01 0.02)

# Sweep B: schedule params only (use fixed scale)
SCHEDULE_SCALE=(0.01)
RAMP_START_FRACS=(0.3 0.5 0.7 0.85)   # late_ramp params
LINEAR_SEGMENTS=(2 3 4 6)            # linear params
SIGMOID_KS=(3 5 7 9)                 # sigmoid params

python scripts/run_cryoimage_ablation.py \
  --base-config "$BASE_CONFIG" \
  --pdb-id "$PDB_ID" \
  --sizes "${SIZES[@]}" \
  --devices "${DEVICES[@]}" \
  --guidance-direction-scale-factors "${GUIDANCE_SCALES[@]}" \
  --schedule-types constant \
  --start-guidance-from "${START_GUIDANCE[@]}" \
  --phenix-manager "$PHENIX_MANAGER"

python scripts/run_cryoimage_ablation.py \
  --base-config "$BASE_CONFIG" \
  --pdb-id "$PDB_ID" \
  --sizes "${SIZES[@]}" \
  --devices "${DEVICES[@]}" \
  --guidance-direction-scale-factors "${SCHEDULE_SCALE[@]}" \
  --schedule-types late_ramp linear sigmoid \
  --schedule-ramp-start-fracs "${RAMP_START_FRACS[@]}" \
  --schedule-segments "${LINEAR_SEGMENTS[@]}" \
  --schedule-sigmoid-ks "${SIGMOID_KS[@]}" \
  --start-guidance-from "${START_GUIDANCE[@]}" \
  --phenix-manager "$PHENIX_MANAGER"
