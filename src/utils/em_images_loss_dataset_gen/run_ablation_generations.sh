#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

for pdb_id in 7T54 7T55; do
  for num_rotations in 1000 2000; do
    python dataset-gen.py --pdb-id "$pdb_id" --num-rotations "$num_rotations" --out-dir output/smoketest
  done
done
