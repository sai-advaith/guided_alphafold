#!/usr/bin/env python3
"""Lightweight unit tests for run-receipt chaining (CPU-only, no model weights)."""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.run_receipts import ReceiptLedger, sha256_json_hex, verify_manifest


def test_chain_is_stable_and_verifiable() -> None:
    with tempfile.TemporaryDirectory() as td:
        led = ReceiptLedger(td, run_name="unit")
        led.record_genesis({"seed": 0, "diffusion_N": 2})
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        led.record_tensor("initial_latents", t, stage_index=0)
        led.record_scalar("experimental_likelihood", 1.25, stage_index=1)
        led.record_tensor("final_ensemble", t * 2, stage_index=2)
        m1 = led.finalize()

        # Second ledger with identical scientific payload must match chain
        led2 = ReceiptLedger(td + "_b", run_name="unit")
        led2.record_genesis({"seed": 0, "diffusion_N": 2})
        led2.record_tensor("initial_latents", t, stage_index=0)
        led2.record_scalar("experimental_likelihood", 1.25, stage_index=1)
        led2.record_tensor("final_ensemble", t * 2, stage_index=2)
        m2 = led2.finalize()
        assert m1["chain_sha256"] == m2["chain_sha256"]

        summary = verify_manifest(led.manifest_path)
        assert summary["ok"] is True
        assert summary["numerical_stability_status"] == "pass"


def test_non_finite_scalar_flagged() -> None:
    with tempfile.TemporaryDirectory() as td:
        led = ReceiptLedger(td, run_name="nan")
        led.record_genesis({"seed": 0})
        led.record_scalar("experimental_likelihood", float("nan"), stage_index=0)
        m = led.finalize()
        assert m["numerical_stability_status"] == "fail"
        assert "experimental_likelihood" in m["non_finite_stages"]
        verify_manifest(led.manifest_path)


def test_tamper_detection() -> None:
    with tempfile.TemporaryDirectory() as td:
        led = ReceiptLedger(td, run_name="tamper")
        led.record_genesis({"seed": 0})
        led.record_scalar("experimental_likelihood", 0.5, stage_index=0)
        m = led.finalize()
        # Tamper with stored chain digest
        path = Path(led.manifest_path)
        data = path.read_text()
        data = data.replace(m["chain_sha256"], "0" * 64)
        path.write_text(data)
        try:
            verify_manifest(str(path))
            raise AssertionError("expected verify_manifest to fail on tamper")
        except AssertionError as e:
            assert "chain mismatch" in str(e)


def test_canonical_json() -> None:
    a = sha256_json_hex({"b": 1, "a": 2})
    b = sha256_json_hex({"a": 2, "b": 1})
    assert a == b


if __name__ == "__main__":
    test_canonical_json()
    test_chain_is_stable_and_verifiable()
    test_non_finite_scalar_flagged()
    test_tamper_detection()
    print("TEST_RUN_RECEIPTS_OK")
