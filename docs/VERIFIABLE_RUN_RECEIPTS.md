# Verifiable run receipts

## Problem

This project already configures seeded deterministic PyTorch execution via
`ExperimentManager.seed_experiment()`. That is necessary but not sufficient for
auditability: there is still no portable, machine-verifiable record that two
runs used the same scientific payload and produced the same artifacts.

## Solution

Opt-in **SHA-256 chained run receipts**:

| Flag | Effect |
|------|--------|
| *(none)* | Unchanged. Existing `seed_experiment()` behavior remains. |
| `--run-receipts` | Write a chained receipt ledger during the run. |
| `--deterministic-mode` | Stricter deterministic torch settings **and** receipts. |

Environment variables (optional):

- `GUIDED_AF_RUN_RECEIPTS=1`
- `GUIDED_AF_DETERMINISTIC=1`

### Outputs

Under `{experiment_save_dir}/receipts/`:

- `*_receipts.jsonl` — chained events (scientific payload only)
- `*_observations.jsonl` — non-chained notes (timestamps, paths, wall time)
- `*_receipt_manifest.json` — terminal chain digest + stability summary

### Verify

```bash
python3 scripts/verify_receipts.py path/to/*_receipt_manifest.json
python3 scripts/test_run_receipts.py
```

## Guarantee (narrow)

With **pinned** code, inputs, model checkpoint, software stack, and hardware
class, independent reruns of the audited path can produce **identical receipt
chains**.

This is **not** a claim of bitwise reproducibility across PyTorch versions,
CUDA versions, GPU architectures, or CPU vs GPU. PyTorch documents that
deterministic algorithms are relative to the same environment.

A receipt chain is **tamper-evident** once the terminal hash is anchored
somewhere trustworthy. Without external anchoring, an adversary who can rewrite
the run can regenerate the chain. Call it a *tamper-evident run receipt chain*,
not independent cryptographic proof of occurrence.

## What is / is not in the chain

**Chained (must match across legitimate reruns):**

- schema version, sequence index, stage name
- previous event digest
- tensor digests (dtype, shape, little-endian f32 bytes)
- scalar digests / fixed `value_repr`
- finite vs non-finite status

**Observations only (may differ):**

- timestamps
- absolute paths
- wall-clock duration
- pod / host identifiers

## Separating determinism from numerical stability

Matching chains prove **repeatability**. They do **not** imply scientific
quality.

Manifest fields:

- `numerical_stability_status`: `pass` | `fail` (any non-finite receipt status)
- `non_finite_stages`: stages that recorded non-finite values

Example interpretation of a dual-run:

| Check | Result |
|-------|--------|
| Receipt chain match (A vs B) | pass |
| Numerical stability | fail if loss became NaN |
| Structure quality / restraint fit | not evaluated by receipts alone |

## Usage

```bash
export CUBLAS_WORKSPACE_CONFIG=:16:8
# Baseline seeding still always runs inside entrypoints:
python3 run_nmr.py 1u0p --deterministic-mode --device cuda:0
# Or receipts only:
python3 run_nmr.py 1u0p --run-receipts --device cuda:0
```

## Design notes for maintainers

- No proprietary kernels or vendor-specific attention backends.
- Integration is opt-in; default CLI behavior is unchanged.
- Genesis payload includes environment fingerprint (Python/torch/CUDA/GPU)
  and scientific knobs (seed, diffusion_N, N_cycle, batch_size, use_msa).
- Absolute model/input paths are intentionally not hashed into the chain
  in this first version; binding checkpoint/input digests is a natural follow-up.

## Validation evidence (external)

On NVIDIA H100 NVL (RunPod), public 7dac NMR-guided dual-runs with fixed seed:

| diffusion_N | Receipt chain match | Numerical stability |
|-------------|---------------------|---------------------|
| 6 | pass | pass (no non-finite events observed) |
| 100 | pass | fail (non-finite NMR loss near end of trajectory) |

Full pod logs and manifests available from the validation environment; not
required to review the code.

Developed and validated by LuxiEdge.
