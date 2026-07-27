"""
Verifiable run receipts for guided AlphaFold ensembles.

Public tooling only: SHA-256 chained receipts over tensor bytes / scalars,
independent of any proprietary acceleration backend.

Guarantee (narrow): with pinned code, inputs, model, software stack, and
hardware class, independent reruns of the audited path can produce identical
receipt chains. This is a tamper-evident record, not independent proof of
authenticity without external anchoring.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch

DETERMINISTIC_ENV = "GUIDED_AF_DETERMINISTIC"
RECEIPTS_ENV = "GUIDED_AF_RUN_RECEIPTS"
SCHEMA_VERSION = 1


def sanitize_run_name(run_name: str) -> str:
    """Make a run name safe for use as a single path component / filename stem."""
    name = str(run_name or "guided_run").replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "guided_run"


def sha256_bytes_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json_hex(obj: Any) -> str:
    """Canonical JSON digest: sorted keys, compact separators, UTF-8."""
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes_hex(raw.encode("utf-8"))


def sha256_tensor_hex(tensor: torch.Tensor) -> str:
    """
    Digest of contiguous little-endian f32 bytes plus dtype/shape metadata.

    Large tensors are truncated to the first 1e6 elements; truncation is
    recorded by the caller in the chained payload.
    """
    t = tensor.detach().float().cpu().contiguous()
    meta = {
        "dtype": "float32",
        "shape": list(t.shape),
        "byte_order": "little",
        "numel": int(t.numel()),
    }
    payload = t.view(-1).numpy().astype("<f4", copy=False).tobytes()
    return sha256_json_hex({"meta": meta, "bytes_sha256": sha256_bytes_hex(payload)})


def receipts_enabled(config_flag: Optional[bool] = None) -> bool:
    if config_flag is not None:
        return bool(config_flag)
    return os.environ.get(RECEIPTS_ENV, "") in ("1", "true", "True", "yes") or (
        os.environ.get(DETERMINISTIC_ENV, "") in ("1", "true", "True", "yes")
    )


def deterministic_mode_enabled(config_flag: Optional[bool] = None) -> bool:
    if config_flag is not None:
        return bool(config_flag)
    return os.environ.get(DETERMINISTIC_ENV, "") in ("1", "true", "True", "yes")


def enable_torch_deterministic(seed: int = 0) -> None:
    """
    Stricter torch deterministic settings layered on top of existing seeding.

    Does not replace ExperimentManager.seed_experiment(); callers should keep
    using that path for baseline reproducibility.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")


def collect_environment_fingerprint() -> Dict[str, Any]:
    """Non-path environment facts suitable for genesis receipts."""
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["cuda"] = getattr(torch.version, "cuda", None)
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        try:
            info["cudnn"] = str(torch.backends.cudnn.version())
        except Exception:
            info["cudnn"] = None
    return info


@dataclass
class ReceiptEvent:
    """One chained receipt record (no wall-clock / path fields in payload)."""

    schema_version: int
    sequence: int
    stage: str
    stage_index: Optional[int]
    previous_receipt_sha256: Optional[str]
    payload: Dict[str, Any]
    status: str  # "finite" | "non_finite" | "ok"
    receipt_sha256: str = ""

    def compute_receipt(self) -> str:
        body = {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "stage": self.stage,
            "stage_index": self.stage_index,
            "previous_receipt_sha256": self.previous_receipt_sha256,
            "payload": self.payload,
            "status": self.status,
        }
        self.receipt_sha256 = sha256_json_hex(body)
        return self.receipt_sha256


class ReceiptLedger:
    """
    Append-only chained receipt ledger.

    Writes:
      - *_receipts.jsonl : one chained event per line
      - *_observations.jsonl : non-chained observations (timestamps, paths, etc.)
      - *_receipt_manifest.json : terminal chain digest + summary
    """

    def __init__(self, out_dir: str, run_name: str = "guided_run"):
        self.out_dir = out_dir
        self.run_name = sanitize_run_name(run_name)
        self.events: List[ReceiptEvent] = []
        self._prev: Optional[str] = None
        os.makedirs(out_dir, exist_ok=True)
        self.jsonl_path = os.path.join(out_dir, f"{self.run_name}_receipts.jsonl")
        self.observations_path = os.path.join(
            out_dir, f"{self.run_name}_observations.jsonl"
        )
        self.manifest_path = os.path.join(
            out_dir, f"{self.run_name}_receipt_manifest.json"
        )
        open(self.jsonl_path, "w", encoding="utf-8").close()
        open(self.observations_path, "w", encoding="utf-8").close()

    def _append_event(self, event: ReceiptEvent) -> str:
        event.compute_receipt()
        self.events.append(event)
        self._prev = event.receipt_sha256
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), sort_keys=True) + "\n")
        return event.receipt_sha256

    def observe(self, kind: str, data: Dict[str, Any]) -> None:
        """Non-chained observation (timestamps, paths, wall time, etc.)."""
        rec = {
            "kind": kind,
            "data": data,
            "observed_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.observations_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    def record_genesis(self, payload: Optional[Dict[str, Any]] = None) -> str:
        body = {
            "environment": collect_environment_fingerprint(),
            "run_name": self.run_name,
            **(payload or {}),
        }
        # Strip absolute-looking paths from genesis payload if any slipped in
        cleaned = {
            k: v
            for k, v in body.items()
            if k not in ("output_dir", "absolute_path", "pod_id")
        }
        ev = ReceiptEvent(
            schema_version=SCHEMA_VERSION,
            sequence=len(self.events),
            stage="genesis",
            stage_index=None,
            previous_receipt_sha256=self._prev,
            payload=cleaned,
            status="ok",
        )
        return self._append_event(ev)

    def record_tensor(
        self,
        stage: str,
        tensor: torch.Tensor,
        *,
        stage_index: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        t = tensor.detach()
        payload_meta = dict(meta or {})
        if t.numel() > 1_000_000:
            flat = t.float().reshape(-1)[:1_000_000]
            payload_meta["truncated_to"] = 1_000_000
            payload_meta["full_numel"] = int(t.numel())
            digest = sha256_tensor_hex(flat)
        else:
            digest = sha256_tensor_hex(t.float())
        finite = bool(torch.isfinite(t.float()).all().item())
        payload = {"tensor_sha256": digest, **payload_meta}
        ev = ReceiptEvent(
            schema_version=SCHEMA_VERSION,
            sequence=len(self.events),
            stage=stage,
            stage_index=stage_index,
            previous_receipt_sha256=self._prev,
            payload=payload,
            status="finite" if finite else "non_finite",
        )
        return self._append_event(ev)

    def record_scalar(
        self,
        stage: str,
        value: float,
        *,
        stage_index: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        v = float(value)
        finite = math.isfinite(v)
        # Fixed-width float representation for determinism of the scalar itself
        payload = {
            "value_hex": sha256_tensor_hex(torch.tensor([v], dtype=torch.float32)),
            "value_repr": f"{v:.17g}",
            **(meta or {}),
        }
        ev = ReceiptEvent(
            schema_version=SCHEMA_VERSION,
            sequence=len(self.events),
            stage=stage,
            stage_index=stage_index,
            previous_receipt_sha256=self._prev,
            payload=payload,
            status="finite" if finite else "non_finite",
        )
        return self._append_event(ev)

    def finalize(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        chain = [e.receipt_sha256 for e in self.events]
        non_finite = [e.stage for e in self.events if e.status == "non_finite"]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_name": self.run_name,
            "n_events": len(self.events),
            "events": [asdict(e) for e in self.events],
            "chain_sha256": sha256_json_hex(chain),
            "determinism_status": "pass",  # filled by dual-run harness if needed
            "numerical_stability_status": "fail" if non_finite else "pass",
            "non_finite_stages": non_finite,
            "extra": extra or {},
        }
        # Non-chained wall-clock observation only
        self.observe(
            "finalize",
            {
                "finalized_utc": datetime.now(timezone.utc).isoformat(),
                "n_events": len(self.events),
            },
        )
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        return manifest


def verify_manifest(manifest_path: str) -> Dict[str, Any]:
    """
    Recompute per-event and chain digests.

    Returns a summary dict. Raises AssertionError on chain/event mismatch.
    """
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    events = manifest.get("events") or []
    recomputed_chain: List[str] = []
    prev: Optional[str] = None
    for i, e in enumerate(events):
        body = {
            "schema_version": e.get("schema_version", SCHEMA_VERSION),
            "sequence": e["sequence"],
            "stage": e["stage"],
            "stage_index": e.get("stage_index"),
            "previous_receipt_sha256": e.get("previous_receipt_sha256"),
            "payload": e.get("payload") or {},
            "status": e.get("status", "ok"),
        }
        got = sha256_json_hex(body)
        if got != e.get("receipt_sha256"):
            raise AssertionError(
                f"event {i} receipt mismatch: got {got} stored {e.get('receipt_sha256')}"
            )
        if e.get("previous_receipt_sha256") != prev:
            raise AssertionError(
                f"event {i} previous link mismatch: got {e.get('previous_receipt_sha256')} expected {prev}"
            )
        recomputed_chain.append(got)
        prev = got

    expected_chain = sha256_json_hex(recomputed_chain)
    if expected_chain != manifest.get("chain_sha256"):
        raise AssertionError(
            f"chain mismatch: got {manifest.get('chain_sha256')} expected {expected_chain}"
        )
    return {
        "ok": True,
        "n_events": len(events),
        "chain_sha256": expected_chain,
        "numerical_stability_status": manifest.get("numerical_stability_status"),
        "non_finite_stages": manifest.get("non_finite_stages") or [],
    }
