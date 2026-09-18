# SHIELD-CV

**Secure Holistic Integrity Evaluation Layer for Defence Computer Vision**

Smart India Hackathon 2025 · Problem Statement **26228** · Ministry of Defence — Indian Army (DGIS) · Theme: Blockchain & Cybersecurity

SHIELD-CV is an **offline, air-gapped, model-agnostic assurance framework** for multi-contributor computer-vision pipelines. When several agencies contribute training data and trained models into a shared defence CV pipeline, SHIELD-CV answers one question: *can this pipeline be trusted?*

It independently evaluates the integrity of three assets:

| Asset | Question answered |
|---|---|
| **Training datasets** | Has anyone poisoned the data — backdoor triggers, flipped labels, duplicate floods, out-of-distribution injections? |
| **Trained models** | Does this model contain an implanted backdoor, anomalous weights, or behaviour inconsistent with its claimed function? |
| **Inference records** | Has the operational record been edited, replayed, deleted, or re-signed after the fact? |

**It never retrains or modifies a contributed model, and it never sends a byte off the host.**

---

## 1. Quick start (5 minutes)

You need **Python 3.10 or newer** and about **2 GB of disk**. No Docker, no GPU, no internet after install.

### Linux / macOS

```bash
cd shield-cv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python demo/run_demo.py --quick
```

### Windows (PowerShell)

```powershell
cd shield-cv
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python demo\run_demo.py --quick
```

That last command is the one to run first. It is a **narrated 8-stage demonstration** that exercises the entire system end to end and prints what each stage proves. It finishes in roughly a minute on a laptop and exits `0` when everything passed.

> **PyTorch tip.** `pip install -r requirements.txt` pulls the default PyTorch build, which on Linux includes large CUDA packages you do not need. For a much smaller CPU-only install:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

---

## 2. What the demo shows you

`python demo/run_demo.py` walks through eight stages. Run a single one with `--stage N`.

| Stage | What it demonstrates |
|---|---|
| 1 | Environment check — every dependency and demo asset present |
| 2 | Clean dataset scan (the control: what "healthy" looks like) |
| 3 | Poisoned dataset scan + detection benchmark (precision/recall/F1) |
| 4 | Model audit, clean vs backdoored, **with no retraining** |
| 5 | Provenance verification across 5 chain fixtures (clean + 4 tamper types) |
| 6 | Drift detection: natural operational drift vs deliberate manipulation |
| 7 | Multi-agent office scan and cross-contributor meeting |
| 8 | Correlation, immunity score, LLM briefing, signed report |

Useful flags:

```bash
python demo/run_demo.py              # full demo
python demo/run_demo.py --quick      # smaller sample, skips the slow trigger search
python demo/run_demo.py --stage 4    # just the model audit
python demo/run_demo.py --no-llm     # deterministic template briefing, no Ollama
python demo/run_demo.py --isolate    # run each stage in its own process (low memory)
```

**The headline contrast** the demo produces — the clean and poisoned pipelines are cleanly separated:

| | Clean pipeline | Poisoned pipeline |
|---|---|---|
| Verdict | `CAUTION` | `COMPROMISED` |
| Risk score | 0.179 | 0.990 |
| Findings | 27 (0 HIGH) | 71 (26 CRITICAL) |
| Immunity score | 76.9 / 100 | 49.1 / 100 |

---

## 3. Using the CLI on your own data

All commands run through `run.py`. Add `--help` to any of them.

```bash
python run.py scan    demo/data/poisoned        # scan a dataset
python run.py audit   demo/models/backdoored_model_scripted.pt
python run.py verify  demo/records/tampered_edit.json
python run.py drift   demo/data/clean demo/data/drift/injected
python run.py office  demo/data/poisoned        # parallel multi-contributor scan
python run.py report  --help                    # signed assurance report
```

Common options: `--json [PATH]` to emit machine-readable output, `--limit N` to cap images.

### Exit codes (scriptable)

| Code | Meaning |
|---|---|
| `0` | Assessed, no findings |
| `1` | Assessed, findings raised |
| `2` | Assessed, pipeline compromised |
| `3` | **Assessment failed** — target unreadable or no detector ran |

Exit code `3` matters. SHIELD-CV never reports "clean" when it could not actually read the data — absence of evidence is not evidence of integrity.

---

## 4. The dashboard

```bash
streamlit run src/dashboard.py
```

Opens a dark military-theme GUI at `http://localhost:8501` with five tabs: **DATA SCANNER**, **MODEL AUDIT**, **CRYPTO LOCK** (an interactive tamper demo — edit a record and watch the chain break), **DRIFT MONITOR**, and **THREAT STORY**.

---

## 5. Air-gapped installation

The target machine never touches the internet. Stage the wheels on a connected host first:

```bash
# On a connected staging host, same OS and Python version as the target:
pip download -r requirements.txt -d wheels/

# Transfer the whole project folder (including wheels/) by approved media, then:
python3 -m venv .venv
source .venv/bin/activate
pip install --no-index --find-links wheels/ -r requirements.txt
```

`requirements.txt` lists **15 dependencies, every one verified as actually imported** by the codebase. Unused packages were deliberately removed to keep the transfer bundle small.

### Proving it is offline

The only outbound call anywhere in the codebase is the *optional* LLM briefing to `localhost:11434` (Ollama), made with the standard library. If Ollama is absent, the briefing falls back to a deterministic template. You can prove there is no egress by running with networking removed entirely:

```bash
unshare -rn bash -c 'ip link set lo up; cd shield-cv && python3 run.py scan demo/data/poisoned'
```

Optional briefing model, if you want it:

```bash
ollama pull llama3.2:3b     # or: phi3:mini
```

Every AI-generated briefing carries the disclaimer *"AI-generated. Verify with analyst."*

---

## 6. Testing

```bash
python tests/test_edge_cases.py        # 30 hostile-input / degradation cases
python demo/validate_phase2.py         # 23 detection-correctness checks
python demo/benchmark.py demo/data/poisoned   # precision / recall / F1
```

Expected: `30 ok / 0 failures` and `23/23`. Under memory pressure, run validation per module: `python demo/validate_phase2.py --modules data`.

Benchmark results on the bundled poisoned corpus (284 images, 9.1 s):

| Attack | Precision | Recall | F1 |
|---|---|---|---|
| Trigger injection | 0.78 | 0.78 | 0.78 |
| Label flipping | 0.67 | 1.00 | 0.80 |
| Near-duplicate flooding | 1.00 | 1.00 | 1.00 |

---

## 7. How it works

### Detection modules

**Data integrity** (`src/scanners/data_scanner.py`) — Trigger detection combines three independent methods: FFT high/low frequency ratio, 16×16 block-variance Z-scores, and SVD spectral signatures. **Two or more methods must agree** before a finding reaches HIGH confidence; a single method is capped at 0.45, below the HIGH cutoff. Also detects label flipping (class-centroid distance and KNN disagreement), near-duplicate flooding (pHash prefilter confirmed by SSIM), and out-of-distribution samples (Mahalanobis plus energy score). Contributor risk aggregates as `0.4·trigger + 0.3·flip + 0.2·dup + 0.1·ood`.

**Model integrity** (`src/scanners/model_auditor.py`) — Auto-detects the access level available (white-box → grey-box → black-box) and **degrades gracefully**, always reporting which methods it could actually use. Runs SHA-256 weight fingerprinting, per-layer statistics, Neural Cleanse trigger reconstruction, activation clustering, and black-box behavioural fingerprinting. No retraining ever occurs.

**Inference provenance** (`src/scanners/crypto_chain.py`) — Ed25519-signed, hash-chained records in tamper-evident SQLite. Detects `HASH_MISMATCH`, `BROKEN_LINK`, `SEQUENCE_GAP`, `DUPLICATE_NONCE`, `TIMESTAMP_REGRESSION`, `MISSING_GENESIS`, and `PAYLOAD_MISMATCH` — the last one re-derives the output digest from the stored payload, so an attacker who edits data *and* recomputes the record hash is still caught.

**Drift** (`src/scanners/drift_detector.py`) — Distinguishes `NATURAL_OPERATIONAL_DRIFT` (dusk, monsoon, sensor ageing) from `SUSPICIOUS_MANIPULATION`, so that genuine field conditions are not escalated as attacks.

### Every finding is explainable

No finding is ever a bare score. Each carries `reason`, `evidence`, `confidence`, and `recommended_disposition` (QUARANTINE ≥ 0.80, REVIEW ≥ 0.50), plus the detection method and attributed contributor. Reports include an explicit **coverage statement**: what was checked *and* what could not be.

---

## 8. Project layout

```
shield-cv/
├── run.py                  # CLI launcher — start here
├── requirements.txt
├── config/                 # settings.yaml, thresholds.yaml, contributors.yaml
├── src/
│   ├── cli.py  dashboard.py  config.py  database.py
│   ├── scanners/           # data, model, crypto, drift, trigger
│   ├── analysis/           # embeddings, statistics, frequency, spectral, neural_cleanse
│   ├── crypto/             # hashing, signing, merkle, chain
│   ├── loaders/            # COCO, YOLO, model, record
│   ├── intelligence/       # threat_story, immunity, briefing
│   ├── agents/             # agent, manager, meeting
│   └── reporting/          # schema, report_generator
├── attacks/                # 5 deterministic attack simulators
├── demo/                   # run_demo.py, data generators, benchmark, fixtures
├── tests/
└── output/                 # keys, reports (generated)
```

No hardcoded paths — everything is read from `config/settings.yaml`.

---

## 9. Requirements and limitations

**Runs on**: Intel i5 / 8 GB RAM / no GPU, Windows, Linux, or macOS. CUDA is optional and never required.

**Memory note**: a white-box model audit peaks near 0.9 GB. On a machine with under 3 GB free, `run_demo.py` automatically runs each stage in a separate process to keep peak usage low; force this with `--isolate`.

**Honest limitations**: black-box auditing cannot inspect weights and reports reduced coverage accordingly. Neural Cleanse requires white-box access. Detection thresholds are calibrated against each corpus's own distribution rather than fixed absolutes, so very small datasets (under ~30 images per contributor) yield lower statistical confidence — and SHIELD-CV says so in the report rather than guessing.

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the venv; run commands from the `shield-cv/` root |
| Killed / exit 137 | Out of memory — use `--quick --isolate`, close other apps |
| `libGL.so.1` error on a server | `pip install opencv-python-headless` instead of `opencv-python` |
| Briefing says `(template)` | Ollama is not running — expected and harmless |
| Dashboard won't open | Check `http://localhost:8501`; port in `.streamlit/config.toml` |
