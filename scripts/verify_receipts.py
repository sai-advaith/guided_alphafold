#!/usr/bin/env python3
"""Verify a chained run-receipt manifest (recompute event + chain digests)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.run_receipts import verify_manifest


def main() -> int:
    p = argparse.ArgumentParser(description="Verify SHA-256 run-receipt manifest")
    p.add_argument("manifest", type=str, help="Path to *_receipt_manifest.json")
    args = p.parse_args()
    summary = verify_manifest(args.manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("VERIFY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
