# SHIELD-CV — Session Report

**Scope of this report:** the work completed in the preceding work session —
closing out Phase 3 (Intelligence & Reporting) and building the first half of
Phase 4 (the multi-agent office).

**Date:** 2026-09-14
**Project:** SHIELD-CV — Secure Holistic Integrity Evaluation Layer for Defence Computer Vision
**Problem Statement:** SIH 2025, PS ID 26228, MoD — Indian Army (DGIS)

---

## 1. Executive summary

Phase 3 is complete and Phase 4 is underway. Two bodies of work were finished:

1. **`report_generator.py` was executed for the first time and repaired.**
   It had been written but never run. Running it exposed three defects, one of
   which was silently disabling the tamper-evident audit trail — the single
   most important integrity guarantee in the framework. All three are fixed and
   the generator now emits schema-valid, hash-stamped reports.

2. **The multi-agent office (`src/agents/`) was built and validated.** Three
   new modules totalling ~1,420 lines: per-contributor scanning agents, a
   parallel office manager, and a meeting room that performs cross-contributor
   analysis and issues the consolidated verdict.

Three defects found during validation were *silent* — they produced
plausible-looking clean output rather than errors. They are detailed in
Section 4 because they are the substantive findings of the session.

Phase 2 regression suite: **23/23 passing**, re-run twice (before and after the
agents work). No regressions introduced.

---

## 2. Phase 3 completion — reporting layer

### 2.1 Defects found by first execution

| # | Defect | Consequence if shipped |
|---|--------|------------------------|
| 1 | `append_audit(detail=...)` — the real keyword is `details` | **Audit trail never recorded.** The report claimed an audit entry while the hash chain stayed empty. A tamper-evidence feature that silently does nothing is worse than none, because it is trusted. |
| 2 | `target` passed as a string; `REPORT_SCHEMA` requires an object | Every report failed validation for a cosmetic reason, training the operator to ignore `schema_valid: false`. |
| 3 | Schema requires a `coverage` key with `supported` / `unsupported` lists; the generator only produced `coverage_statement` | Reports rejected by the validator. |

### 2.2 Fixes applied

- **Audit trail** now records correctly. Verified: chain of 50 entries,
  `chain_valid: true`, `audit_trail_hash` populated.
- **`target`** is normalised — a plain string is wrapped into
  `{"description": ...}`, so either form validates.
- **`coverage`** is emitted as the schema-mandated key, with
  `coverage_statement` retained as an alias so the dashboard and existing
  consumers keep working.
- **Unassessed modules are now explicitly enumerated as `unsupported`**, each
  tagged `"(module not run)"`, rather than omitted. This enforces the standing
  design rule that *absence of evidence must never read as a clean result*.

### 2.3 Verification

- `schema_valid: True`.
- `report_hash` recomputes identically from the persisted JSON file
  (hash is taken over the report *excluding* the hash field itself).
- `output/reports/` created; 3 reports written during testing.
- Full-attack run: 124 findings, verdict `COMPROMISED`, 8 known blind spots,
  7 limitations recorded.

### 2.4 Package exports closed out

`src/scanners/__init__.py` now exports all five engines (data, model, crypto,
drift, trigger). Created `src/reporting/__init__.py` and `src/agents/__init__.py`;
`src/intelligence/__init__.py` gained real exports. All packages import cleanly.

One error caught here: `iso_timestamp` is a **method on `Finding`**, not a
module-level function — the re-export was corrected rather than left to fail at
import time.

---

## 3. Phase 4 (part 1) — the multi-agent office

### 3.1 `src/agents/agent.py` (292 lines)

`AgentReport` dataclass + `agent_scan_contributor()`.

**Design rationale.** Each contributor is scanned *in isolation*. A
dataset-wide scan lets a heavy contributor dominate the population statistics
the detectors score against — so a contributor poisoning a large share of the
corpus pulls the baseline toward itself and appears normal. Per-contributor
scanning removes that attack path.

- Reports hold only primitives → picklable across process boundaries
  (verified explicitly).
- Agents **never raise**: failures become `status='ERROR'` with the reason
  recorded, so one bad folder cannot abort an office-wide run.
- A contributor with zero readable samples is marked `EMPTY`, **not** clean,
  and is excluded from peer statistics.
- Unregistered contributors default to `UNVERIFIED` — an undeclared source is
  never treated as trusted.

### 3.2 `src/agents/manager.py` (337 lines)

`OfficeManager` — discovery + parallel dispatch.

- **Three-pass discovery:** configured glob patterns → non-structural
  sub-folders → the root itself as a single contributor. A `_STRUCTURAL_DIRS`
  guard prevents a plain `images/` + `labels/` layout being mistaken for two
  contributors.
- **Workers capped at `min(config, os.cpu_count())`.** Each worker loads its
  own embedding backbone; oversubscribing the target 8 GB / 4-core i5 causes
  swapping, which is slower than running sequentially.
- **Process pool, not threads** — the detectors are CPU-bound and hold the GIL.
- Single worker or single contributor runs **in-process**, keeping the pipeline
  debuggable.
- If the pool cannot start (restricted environments, no `/dev/shm`, frozen
  executables) the whole batch **retries serially**. That is an infrastructure
  problem, not an analysis result, and must not be reported as scan errors.

**Measured:** 3 contributors, 2 workers, 7.6 s.

### 3.3 `src/agents/meeting.py` (794 lines)

`MeetingRoom` — four cross-contributor analyses that are invisible to any
single agent:

| Analysis | Signal |
|---|---|
| **Disparity** | One contributor standing clear of its peers |
| **Class targeting** | Multiple contributors attacking the *same* label |
| **Trust inversion** | A TRUSTED source behaving worse than an UNVERIFIED one |
| **Model disagreement** | Candidate models diverging on identical probes |

Statistical care taken: comparisons use the **median and MAD**, not mean and
standard deviation, because the mean is dragged by the very outlier being
hunted. Disparity requires both a ratio test *and* an absolute risk floor, so a
cosmetic 0.02-vs-0.01 gap between near-clean contributors is never escalated.
All confidences are capped below certainty.

---

## 4. Substantive findings — three silent defects

These are recorded in full because each produced *confident, plausible, wrong*
output rather than an error.

### 4.1 The disparity z-score threshold was mathematically unattainable

For `n` contributors, the largest attainable absolute z-score is `(n-1)/√n`:

| n | max attainable \|z\| |
|---|---|
| 2 | 1.000 |
| 3 | **1.155** |
| 4 | 1.732 |
| 5 | 2.000 |

The configured threshold is **1.5**. With three contributors the test **could
never fire** — and would have reported "no disparity found," which reads as a
clean result.

**Resolution:** a robust median-ratio rule (>1.8× peer median *and* risk >0.45)
is now the operative test; the z-test is retained as a secondary signal; the
attainability ceiling is computed at runtime and recorded as an explicit
limitation whenever it binds.

This is the same class of trap as the previously-documented Neural Cleanse MAD
ceiling — a threshold that cannot be reached given the sample size.

### 4.2 Uniform-noise probes caused a false negative on model comparison

Comparing the clean and backdoored models on uniform random noise gave a
**0.0% mismatch rate** — i.e. "these models are identical." They are not; one
carries a backdoor.

**Cause:** uniform noise is far off the data manifold, where both models
saturate to the same output class. This is the same failure mode already
documented for Neural Cleanse probes.

**Resolution:** structured synthetic probes (oriented gratings + Gaussian blobs
+ gradients) that excite convolutional filters the way natural imagery does,
with real reference images preferred when available.

| Probe set | Mismatch rate | Detected? |
|---|---|---|
| Uniform noise (before) | 0.0% | **No — false negative** |
| Structured synthetic | **79.0%** | Yes |
| Real reference images | **21.7%** | Yes |

An honest limitation is recorded alongside the result: this proves two models
*differ*, but cannot say which is authentic without a trusted reference.

### 4.3 `str(Enum)` produced schema-invalid findings

`str(AttackClass.LABEL_FLIPPING)` returns `"AttackClass.LABEL_FLIPPING"`, not
`"LABEL_FLIPPING"`. This broke the class-targeting filter (matching zero
findings) and would have written invalid `attack_class` values into reports.

**Resolution:** hardened `Finding.create()` itself to normalise an `Enum`
member or a string, rather than only patching the call sites — no future caller
can reintroduce the bug. Also normalised class-label namespaces, since
detectors name the same class differently (`"class_2"` vs `2`).

---

## 5. Validation results

### 5.1 A/B control — poisoned vs clean corpus

| | Poisoned | Clean (control) |
|---|---|---|
| Verdict | `COMPROMISED` | `CAUTION` |
| Meeting findings | 4 | 1 |
| Class targeting | classes 1 & 2 by all 3 contributors | **none** |
| False disparity claims | none | **none** |
| False inversion claims | none | **none** |

Poisoned-corpus findings:

```
MEET-001  COORDINATED_ATTACK    HIGH      0.79  REVIEW      class:0
MEET-002  COORDINATED_ATTACK    CRITICAL  0.92  QUARANTINE  class:1
MEET-003  COORDINATED_ATTACK    CRITICAL  0.92  QUARANTINE  class:2
MEET-004  MODEL_DISAGREEMENT    HIGH      0.72  REVIEW      backdoored vs clean
```

All findings validate against the Finding schema. The clean control raised only
the model-disagreement finding (correct — the backdoored model genuinely is
different), and manufactured no targeting or disparity claims.

### 5.2 Regression

`python3 demo/validate_phase2.py` → **23/23**, run twice (~47 s). No regressions.

---

## 6. Files created / modified

**Created**
- `src/agents/agent.py` (292)
- `src/agents/manager.py` (337)
- `src/agents/meeting.py` (794)
- `src/agents/__init__.py` (18)
- `src/reporting/__init__.py` (24)

**Modified**
- `src/reporting/report_generator.py` (427) — three defect fixes
- `src/reporting/schema.py` — `Finding.create()` enum hardening
- `src/scanners/__init__.py` — all five engines exported
- `src/intelligence/__init__.py` — real exports

All work observes the standing constraints: no Docker, no TODO placeholders,
full docstrings, module-level try/except, `pathlib` throughout, config-driven
thresholds, no network access.

---

## 7. Outstanding work

**Phase 4 (remainder)**
- `src/cli.py` — `scan`, `audit`, `verify`, `drift`, `report`, `office` with rich output
- `src/dashboard.py` — Streamlit, 5 tabs

**Phase 5**
- `README.md`, `setup.py`, `.gitignore`, `run.py`
- `demo/run_demo.py`, `demo/annotations.json`
- 5 × `tests/test_*.py`, 5 × `docs/*.md`

---

## 8. Open question for the user

On the **clean** corpus, contributors *bravo* and *charlie* score **0.69**,
driven by the known near-duplicate false positives. That is high enough to look
alarming in a live demo, even though the meeting room correctly declines to
raise targeting or disparity findings.

Two options:

1. **Recalibrate near-duplicate confidence** before building the dashboard, so
   clean data scores visibly clean.
2. **Leave the detector as-is** and let the coverage statement carry the caveat.

Option 1 costs time now but makes the demo's clean/poisoned contrast much
sharper. Awaiting a decision.
