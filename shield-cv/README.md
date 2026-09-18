<div align="center">

# 🛡️ SHIELD-CV 🔒

### **S**ecure **H**olistic **I**ntegrity **E**valuation **L**ayer for **D**efence **C**omputer **V**ision

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-✅%20Operational-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=for-the-badge)

<br>

🎯 **Smart India Hackathon 2025** · Problem Statement **#26228**
Ministry of Defence — Indian Army (DGIS) · Theme: **Blockchain & Cybersecurity**

</div>

---

## 🌟 What is SHIELD-CV?

SHIELD-CV is an **offline, air-gapped, model-agnostic assurance framework** for multi-contributor computer-vision pipelines. When several agencies contribute training data and trained models into a shared defence CV pipeline, SHIELD-CV answers one critical question:

> 🔍 ***Can this pipeline be trusted?***

---

## 🎯 What Does It Protect?

<table>
<tr>
<th width="5%">🎯</th>
<th width="25%">Asset</th>
<th>Question Answered</th>
</tr>
<tr>
<td>📊</td>
<td><strong>Training Datasets</strong></td>
<td>Has anyone poisoned the data — backdoor triggers, flipped labels, duplicate floods, out-of-distribution injections?</td>
</tr>
<tr>
<td>🧠</td>
<td><strong>Trained Models</strong></td>
<td>Does this model contain an implanted backdoor, anomalous weights, or behaviour inconsistent with its claimed function?</td>
</tr>
<tr>
<td>📋</td>
<td><strong>Inference Records</strong></td>
<td>Has the operational record been edited, replayed, deleted, or re-signed after the fact?</td>
</tr>
</table>

> 🔒 **It never retrains or modifies a contributed model, and it never sends a byte off the host.**

---

## 📑 Table of Contents

- [🚀 Quick Start](#-quick-start-5-minutes)
- [🎭 What the Demo Shows](#-what-the-demo-shows-you)
- [⚡ CLI Usage](#-using-the-cli-on-your-own-data)
- [🖥️ Dashboard](#️-the-dashboard)
- [🔌 Air-Gapped Installation](#-air-gapped-installation)
- [🧪 Testing](#-testing)
- [⚙️ How It Works](#-how-it-works)
- [📁 Project Layout](#-project-layout)
- [📋 Requirements](#-requirements-and-limitations)
- [🔧 Troubleshooting](#-troubleshooting)

---

## 🚀 Quick Start (5 Minutes)

You need **Python 3.10 or newer** and about **2 GB of disk**. No Docker, no GPU, no internet after install.

<details>
<summary><strong>🐧 Linux / macOS</strong></summary>

```bash
cd shield-cv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python demo/run_demo.py --quick
```

</details>

<details>
<summary><strong>🪟 Windows (PowerShell)</strong></summary>

```powershell
cd shield-cv
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python demo\run_demo.py --quick
```

</details>

> 💡 **That last command is the magic one!** It runs a **narrated 8-stage demonstration** that exercises the entire system end to end and prints what each stage proves. Finishes in ~1 minute. Exits `0` when everything passed. ✅

---

### 💡 Pro Tip — Smaller Install

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 🎭 What the Demo Shows You

`python demo/run_demo.py` walks through **8 powerful stages**. Run a single one with `--stage N`.

| Stage | Icon | Description |
|:-----:|:----:|-------------|
| **1** | 🔍 | Environment check — every dependency and demo asset present |
| **2** | ✅ | Clean dataset scan (the control: what "healthy" looks like) |
| **3** | ⚠️ | Poisoned dataset scan + detection benchmark (precision/recall/F1) |
| **4** | 🧠 | Model audit, clean vs backdoored, **with no retraining** |
| **5** | 🔗 | Provenance verification across 5 chain fixtures (clean + 4 tamper types) |
| **6** | 📈 | Drift detection: natural operational drift vs deliberate manipulation |
| **7** | 🤖 | Multi-agent office scan and cross-contributor meeting |
| **8** | 📊 | Correlation, immunity score, LLM briefing, signed report |

### 🎛️ Useful Flags

```bash
python demo/run_demo.py              # 🎬 full demo
python demo/run_demo.py --quick      # ⚡ smaller sample, skips slow trigger search
python demo/run_demo.py --stage 4    # 🧠 just the model audit
python demo/run_demo.py --no-llm     # 📝 deterministic template briefing, no Ollama
python demo/run_demo.py --isolate    # 🔒 run each stage in its own process (low memory)
```

### 📊 The Headline Results

The demo produces a **clean contrast** — clean vs poisoned pipelines are clearly separated:

<table>
<tr>
<th></th>
<th>🟢 Clean Pipeline</th>
<th>🔴 Poisoned Pipeline</th>
</tr>
<tr>
<td><strong>Verdict</strong></td>
<td><code>CAUTION</code></td>
<td><code>COMPROMISED</code></td>
</tr>
<tr>
<td><strong>Risk Score</strong></td>
<td><code>0.179</code></td>
<td><code>0.990</code></td>
</tr>
<tr>
<td><strong>Findings</strong></td>
<td><code>27 (0 HIGH)</code></td>
<td><code>71 (26 CRITICAL)</code></td>
</tr>
<tr>
<td><strong>Immunity Score</strong></td>
<td><code>76.9 / 100</code></td>
<td><code>49.1 / 100</code></td>
</tr>
</table>

---

## ⚡ Using the CLI on Your Own Data

All commands run through `run.py`. Add `--help` to any of them.

```bash
python run.py scan    demo/data/poisoned          # 📊 scan a dataset
python run.py audit   demo/models/backdoored_model_scripted.pt   # 🧠 audit a model
python run.py verify  demo/records/tampered_edit.json           # 🔗 verify provenance
python run.py drift   demo/data/clean demo/data/drift/injected  # 📈 detect drift
python run.py office  demo/data/poisoned          # 🏢 parallel multi-contributor scan
python run.py report  --help                      # 📋 signed assurance report
```

Common options: `--json [PATH]` to emit machine-readable output, `--limit N` to cap images.

### 🚦 Exit Codes (Scriptable)

| Code | Status | Meaning |
|:----:|:------:|---------|
| `0` | ✅ | Assessed, no findings |
| `1` | ⚠️ | Assessed, findings raised |
| `2` | 🔴 | Assessed, pipeline compromised |
| `3` | ❌ | **Assessment failed** — target unreadable or no detector ran |

> ⚠️ **Exit code `3` matters.** SHIELD-CV never reports "clean" when it could not actually read the data — *absence of evidence is not evidence of integrity.*

---

## 🖥️ The Dashboard

```bash
streamlit run src/dashboard.py
```

Opens a **dark military-theme GUI** at `http://localhost:8501` with **5 powerful tabs**:

| Tab | Icon | Description |
|:----|:----:|-------------|
| DATA SCANNER | 📊 | Scan datasets for poisoning |
| MODEL AUDIT | 🧠 | Audit models for backdoors |
| CRYPTO LOCK | 🔗 | Interactive tamper demo — edit a record and watch the chain break |
| DRIFT MONITOR | 📈 | Monitor operational drift |
| THREAT STORY | 📖 | Visualize threat intelligence |

---

## 🔌 Air-Gapped Installation

The target machine **never touches the internet**. Stage the wheels on a connected host first:

```bash
# 📥 On a connected staging host, same OS and Python version as the target:
pip download -r requirements.txt -d wheels/

# 🚚 Transfer the whole project folder (including wheels/) by approved media, then:
python3 -m venv .venv
source .venv/bin/activate
pip install --no-index --find-links wheels/ -r requirements.txt
```

> 📦 `requirements.txt` lists **15 dependencies, every one verified as actually imported** by the codebase. Unused packages were deliberately removed to keep the transfer bundle small.

### 🔒 Proving It Is Offline

The only outbound call anywhere in the codebase is the *optional* LLM briefing to `localhost:11434` (Ollama). If Ollama is absent, the briefing falls back to a deterministic template.

```bash
# Prove zero network egress:
unshare -rn bash -c 'ip link set lo up; cd shield-cv && python3 run.py scan demo/data/poisoned'
```

Optional briefing model:
```bash
ollama pull llama3.2:3b     # or: phi3:mini
```

> 🤖 Every AI-generated briefing carries the disclaimer *"AI-generated. Verify with analyst."*

---

## 🧪 Testing

```bash
python tests/test_edge_cases.py        # 🧱 30 hostile-input / degradation cases
python demo/validate_phase2.py         # ✅ 23 detection-correctness checks
python demo/benchmark.py demo/data/poisoned   # 📈 precision / recall / F1
```

> ✅ Expected: `30 ok / 0 failures` and `23/23`

Under memory pressure, run validation per module:
```bash
python demo/validate_phase2.py --modules data
```

### 📊 Benchmark Results (284 images, 9.1s)

| Attack | Precision | Recall | F1 |
|:-------|:---------:|:------:|:--:|
| 🎯 Trigger injection | `0.78` | `0.78` | `0.78` |
| 🔄 Label flipping | `0.67` | `1.00` | `0.80` |
| 📋 Near-duplicate flooding | `1.00` | `1.00` | `1.00` |

---

## ⚙️ How It Works

### 🔍 Detection Modules

<details>
<summary><strong>📊 Data Integrity</strong> — <code>src/scanners/data_scanner.py</code></summary>

Trigger detection combines **three independent methods**:
- 📡 **FFT** high/low frequency ratio
- 📊 **16×16 block-variance Z-scores**
- 📈 **SVD spectral signatures**

**Two or more methods must agree** before a finding reaches HIGH confidence; a single method is capped at `0.45`, below the HIGH cutoff.

Also detects:
- 🔄 Label flipping (class-centroid distance + KNN disagreement)
- 📋 Near-duplicate flooding (pHash prefilter + SSIM)
- 📊 Out-of-distribution samples (Mahalanobis + energy score)

Contributor risk = `0.4·trigger + 0.3·flip + 0.2·dup + 0.1·ood`

</details>

<details>
<summary><strong>🧠 Model Integrity</strong> — <code>src/scanners/model_auditor.py</code></summary>

Auto-detects the access level available:
- 🟢 **White-box** → full weight inspection
- 🟡 **Grey-box** → limited inspection
- 🔴 **Black-box** → behavioural fingerprinting

**Degrades gracefully**, always reporting which methods it could actually use.

Methods include:
- 🔐 SHA-256 weight fingerprinting
- 📊 Per-layer statistics
- 🧠 Neural Cleanse trigger reconstruction
- 📈 Activation clustering
- 🔍 Black-box behavioural fingerprinting

> ❌ **No retraining ever occurs.**

</details>

<details>
<summary><strong>🔗 Inference Provenance</strong> — <code>src/scanners/crypto_chain.py</code></summary>

Ed25519-signed, hash-chained records in tamper-evident SQLite.

Detects:
- ⚠️ `HASH_MISMATCH`
- 🔗 `BROKEN_LINK`
- 📊 `SEQUENCE_GAP`
- 🔁 `DUPLICATE_NONCE`
- ⏰ `TIMESTAMP_REGRESSION`
- ❌ `MISSING_GENESIS`
- 🔄 `PAYLOAD_MISMATCH` — re-derives the output digest from the stored payload, so an attacker who edits data **and** recomputes the record hash is still caught.

</details>

<details>
<summary><strong>📈 Drift Detection</strong> — <code>src/scanners/drift_detector.py</code></summary>

Distinguishes:
- 🌿 `NATURAL_OPERATIONAL_DRIFT` — dusk, monsoon, sensor ageing
- ⚠️ `SUSPICIOUS_MANIPULATION`

So genuine field conditions are not escalated as attacks.

</details>

### 🔎 Every Finding Is Explainable

No finding is ever a bare score. Each carries:

| Field | Description |
|-------|-------------|
| `reason` | Why this finding exists |
| `evidence` | The data that supports it |
| `confidence` | How certain we are |
| `recommended_disposition` | QUARANTINE ≥ 0.80, REVIEW ≥ 0.50 |
| `detection_method` | Which detector found it |
| `contributor` | Who contributed the data |

Reports include an explicit **coverage statement**: what was checked *and* what could not be.

---

## 📁 Project Layout

```
🛡️ shield-cv/
├── 🚀 run.py                  # CLI launcher — start here
├── 📋 requirements.txt
├── ⚙️ config/                 # settings.yaml, thresholds.yaml, contributors.yaml
├── 📦 src/
│   ├── 🖥️ cli.py  dashboard.py  config.py  database.py
│   ├── 🔍 scanners/           # data, model, crypto, drift, trigger
│   ├── 📊 analysis/           # embeddings, statistics, frequency, spectral, neural_cleanse
│   ├── 🔐 crypto/             # hashing, signing, merkle, chain
│   ├── 📥 loaders/            # COCO, YOLO, model, record
│   ├── 🧠 intelligence/       # threat_story, immunity, briefing
│   ├── 🤖 agents/             # agent, manager, meeting
│   └── 📋 reporting/          # schema, report_generator
├── ⚔️ attacks/                # 5 deterministic attack simulators
├── 🎭 demo/                   # run_demo.py, data generators, benchmark, fixtures
├── 🧪 tests/
└── 📂 output/                 # keys, reports (generated)
```

> ✅ No hardcoded paths — everything is read from `config/settings.yaml`.

---

## 📋 Requirements and Limitations

### ✅ Runs On

| Component | Requirement |
|-----------|-------------|
| 💻 CPU | Intel i5 or better |
| 🧠 RAM | 8 GB minimum |
| 🎮 GPU | ❌ Not required (CUDA optional) |
| 🖥️ OS | Windows, Linux, or macOS |

### ⚠️ Memory Note

A white-box model audit peaks near **0.9 GB**. On a machine with under 3 GB free, `run_demo.py` automatically runs each stage in a separate process. Force this with `--isolate`.

### ⚖️ Honest Limitations

| Limitation | Impact |
|------------|--------|
| Black-box auditing | Cannot inspect weights — reports reduced coverage |
| Neural Cleanse | Requires white-box access |
| Small datasets | Under ~30 images per contributor → lower statistical confidence |

> 💡 SHIELD-CV **says so in the report** rather than guessing.

---

## 🔧 Troubleshooting

| Symptom | 🔧 Fix |
|---------|--------|
| `ModuleNotFoundError` | Activate the venv; run commands from the `shield-cv/` root |
| Killed / exit 137 | Out of memory — use `--quick --isolate`, close other apps |
| `libGL.so.1` error on a server | `pip install opencv-python-headless` instead of `opencv-python` |
| Briefing says `(template)` | Ollama is not running — expected and harmless |
| Dashboard won't open | Check `http://localhost:8501`; port in `.streamlit/config.toml` |

---

<div align="center">

### 🛡️ Built for Defence. Built for Trust. Built for India.

---

**Smart India Hackathon 2025** · Problem Statement **#26228**
Ministry of Defence — Indian Army (DGIS)

![Made with ❤️](https://img.shields.io/badge/Made_with_❤️_for_India-FF6B35?style=for-the-badge)

</div>
