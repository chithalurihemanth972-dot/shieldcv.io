# SHIELD-CV — Session Report: Recalibration + Phase 4

**Date:** 2026-09-14
**Scope:** near-duplicate recalibration, risk-score redesign, CLI, dashboard.

---

## 1. Executive summary

The open question from the previous session — whether to recalibrate the
near-duplicate detector before building the UI — was answered by **doing the
recalibration**. That was the right call: chasing the 0.69 clean-data score
uncovered three further defects that would each have shipped as confident,
wrong output, including a **security hole in the provenance chain**.

Phase 4 is now complete: `src/cli.py` (six commands), `src/dashboard.py` (five
tabs) and `run.py` are built and working.

**Headline result — clean data now reads clean, poisoned still reads poisoned:**

| Metric | Clean (before) | Clean (after) | Poisoned (after) |
|---|---|---|---|
| Dataset risk | 0.998 | **0.179** | 0.990 |
| Verdict | SUSPICIOUS | **CAUTION** | COMPROMISED |
| Immunity score | — | **76.9 ADEQUATE** | 49.1 COMPROMISED |
| Contributor risk | 0.69 | **0.40–0.48** | 0.76–0.86 |
| Findings | 54 | **27** | 71 |

Detection performance was **not** traded away for this:

| Attack | Precision | Recall | F1 |
|---|---|---|---|
| TRIGGER_INJECTION | 0.78 | 0.78 | 0.78 |
| LABEL_FLIPPING | 0.67 | 1.00 | 0.80 |
| NEAR_DUPLICATE_FLOODING | 1.00 | 1.00 | **1.00** |

Phase 2 regression: **23/23**.

---

## 2. The near-duplicate recalibration

### 2.1 Root cause

The detector emitted a finding for **every** pHash cluster, including benign
size-2 pairs, at a floor confidence of 0.35. Measuring the corpora showed why
that was hopeless:

| Corpus | Cluster sizes | Real attack |
|---|---|---|
| Clean | 26×size-2, 8×size-3, 1×size-4 | none |
| Poisoned | 24×size-2, 9×size-3, **2×size-8** | the two size-8 clusters |

Natural imagery *always* contains near-identical pairs. The absolute
`flood_cluster_min_size: 4` rule even flagged a naturally-occurring size-4
cluster in clean data.

### 2.2 Two fixes

**SSIM confirmation (was entirely missing).** The specification requires
"pHash, Hamming <5, plus SSIM" — SSIM appeared nowhere in the codebase. pHash
alone is a 64-bit lossy summary, so different photographs of the same subject
land within a few bits of each other. I implemented SSIM directly on
numpy/cv2 with a Gaussian window rather than adding a `scikit-image`
dependency to an air-gapped install (13 ms per comparison).

**Population-relative thresholds.** Both thresholds are now derived from the
dataset's own distribution, following the principle already established
elsewhere in this project — *score against the population, never an absolute
constant*:

- Flood threshold = `median + 3×MAD` of cluster sizes, floored at the config value.
- SSIM cut-off = calibrated against a random-pair baseline from the same corpus.

The SSIM calibration turned out to be essential: **random unrelated pairs in
this corpus already score 0.92 SSIM**. A fixed 0.90 constant would have
confirmed virtually everything.

**Result:** SSIM rejected 46 spurious pHash links on clean data (35 → 9
clusters, **0 floods**), while the poisoned corpus retained exactly the 2 real
attack clusters (mean SSIM 0.985–0.988, both `contributor_alpha`, matching
ground truth).

A contributor-level share rule (>20% duplicated material) was also added, as
required by the spec and previously absent.

---

## 3. Three further defects found

### 3.1 The "2+ methods" rule was never enforced

**Every** clean-data trigger finding was single-method (spectral only), yet
reaching **0.80 confidence / HIGH severity**. The spec requires 2 of 3 methods
to agree for high confidence. Measuring precision by agreement count on the
poisoned corpus proved the spec right:

| Methods agreeing | Findings | Precision |
|---|---|---|
| 1 | 8 | 0.63 |
| 2 | 7 | 0.86 |
| 3 | 3 | **1.00** |

Single-method findings are now capped at 0.45 — reported as a *lead for
review*, never as a confirmed detection. Clean data dropped from 5 HIGH
findings to **zero**, with no loss of recall.

### 3.2 The risk score rewarded volume over severity

`summarize_findings` used a noisy-OR whose docstring claimed "many
low-confidence findings cannot outrank a single CRITICAL one". It did exactly
that:

| Input | Old risk | New risk |
|---|---|---|
| 1 CRITICAL @ 0.95 | 0.950 | 0.713 |
| 100 LOW @ 0.25 | **0.978** | **0.038** |
| 26 CRITICAL @ 0.90 | ~1.000 | 0.897 |

Replaced with the severity-dominated saturating curve already validated in
`immunity.py`: `risk = 0.75·worst + 0.25·breadth·worst`, where breadth is
`1 - exp(-tail/6)`. Scaling breadth *by worst* is the key property — a pile of
trivial findings has no severe finding to amplify, so it cannot manufacture a
high score.

### 3.3 Security hole: the chain never bound its own payload

`verify_chain` hashed `output_hash` but **never checked that `output_hash`
matched `output`**. An attacker could rewrite the recorded inference result
and every link still verified:

```
edit output only     -> valid=True   ← the recorded answer is a lie
edit output + rehash -> valid=True
```

This is the "hashing a digest-of-a-digest leaves a hole" trap. The high-level
`crypto_chain` engine did check this (INFERENCE_TAMPER, 0.97), but the
low-level primitive did not — and my dashboard tamper demo called it directly,
so the demo would have displayed *"that edit was not detected."*

Added a `PAYLOAD_MISMATCH` break type that re-derives the digest from the
stored payload. Now:

```
edit output only     -> valid=False  PAYLOAD_MISMATCH @ index 5
edit output + rehash -> valid=False  PAYLOAD_MISMATCH @ index 5
```

All five chain fixtures still classify correctly; the clean chain still verifies.

---

## 4. Phase 4 deliverables

### 4.1 `src/cli.py` + `run.py`

Six commands — `scan`, `audit`, `verify`, `drift`, `report`, `office` — with
rich colour output, progress bars, severity-coloured tables and evidence
drill-down. Every command prints its coverage and limitations rather than
hiding them behind a verbose flag.

**Meaningful exit codes** so the tool can gate a pipeline: `0` clean, `1`
findings need review, `2` compromised, `3` error.

That last code exposed a real problem during testing: `scan <nonexistent-path>`
reported **"CLEAN, exit 0"**. A mistyped path would have silently passed a CI
gate. Added `_assessment_failed()`, which detects a run that examined nothing
(no samples, no records, unavailable access, no detector available) and returns
`NOT ASSESSED — this is not a clean result` with exit 3.

### 4.2 `src/dashboard.py`

Streamlit, dark military theme, five tabs: DATA SCANNER, MODEL AUDIT, CRYPTO
LOCK (live editable tamper demo), DRIFT MONITOR, THREAT STORY (immunity gauge,
narratives, LLM briefing, downloadable report). Results are cached in
`st.session_state` so unrelated widget interactions never trigger a rescan.
Running on port 8501.

### 4.3 Memory-aware worker sizing

The office command crashed with *"a process in the process pool was terminated
abruptly"* — workers were being OOM-killed. Each worker loads its own torch
runtime plus a ResNet-18 backbone (~700 MB), and sizing the pool purely by CPU
count ignores that. `OfficeManager` now budgets by `MemAvailable` and caps
workers accordingly (it correctly self-limited to 1 worker at 890 MB free).
`BrokenProcessPool` now triggers a full serial retry instead of losing the batch.

---

## 5. Files created / modified

**Created:** `src/cli.py`, `src/dashboard.py`, `run.py`, `.streamlit/config.toml`,
`docs/session_report_phase4.md`

**Modified:** `src/analysis/outlier.py` (SSIM + adaptive thresholds),
`src/scanners/data_scanner.py` (material findings only, corroboration rule,
contributor share), `src/reporting/schema.py` (risk redesign),
`src/crypto/chain.py` (PAYLOAD_MISMATCH), `src/agents/manager.py` (memory-aware
workers), `src/agents/meeting.py` (accurate probe limitation),
`config/settings.yaml`

---

## 6. Remaining work (Phase 5)

- `README.md`, `setup.py`, `.gitignore`
- `demo/run_demo.py`, `demo/annotations.json`
- 5 × `tests/test_*.py`
- 5 × `docs/*.md`
