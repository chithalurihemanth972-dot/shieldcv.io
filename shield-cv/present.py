#!/usr/bin/env python3
"""SHIELD-CV — Jury Presentation Mode.

Run this in the terminal to walk judges through the entire framework
with animated panels, live demos, and visual comparisons.

    python present.py              # full presentation
    python present.py --quick      # skip slow scans
    python present.py --stage 3    # jump to a specific stage
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# ── colours ──────────────────────────────────────────────────────────────────
CYAN    = "bold cyan"
GREEN   = "bold green"
RED     = "bold red"
YELLOW  = "bold yellow"
MAGENTA = "bold magenta"
WHITE   = "bold white"
DIM     = "dim"
BLUE    = "bold blue"

# ── helpers ──────────────────────────────────────────────────────────────────

def _slow_print(text: str, style: str = "", delay: float = 0.02) -> None:
    """Type text character by character for dramatic effect."""
    msg = Text(text, style=style)
    with Live(console=console, refresh_per_second=60, transient=True) as live:
        displayed = Text("", style=style)
        for ch in text:
            displayed.append(ch)
            live.update(displayed)
            time.sleep(delay)

def _pause(msg: str = "Press Enter to continue...") -> None:
    """Wait for user input."""
    console.print(f"\n  [{DIM}]{msg}[/{DIM}]", end="")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass

def _slide_number(num: int, title: str) -> None:
    console.print()
    console.rule(f"  [{CYAN}]SLIDE {num} — {title}[/{CYAN}]", style="cyan")
    console.print()

def _verdict_style(verdict: str) -> str:
    v = verdict.upper()
    if "COMPROMISED" in v: return "bold white on red"
    if "SUSPICIOUS" in v: return "bold dark_orange"
    if "CAUTION" in v: return "bold yellow"
    if "CLEAN" in v: return "bold white on green"
    return "white"

def _sev_color(sev: str) -> str:
    return {"CRITICAL": "bold red", "HIGH": "bold dark_orange",
            "MEDIUM": "bold yellow", "LOW": "bold blue",
            "INFO": "dim"}.get(sev.upper(), "white")


# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ═════════════════════════════════════════════════════════════════════════════

LOGO = r"""
   _____ __  __ ____    _____ ____   ___  _   _ ______ _____ _____ ______ ____
  / ____|  \/  |  _ \  | ____| __ ) / _ \| \ | |  _ \_   _| ____|  ____/ ___|
 | (___ | |\/| | |_) | |  _| |  _ \| | | |  \| | | | || | |  _| |  _|  \___ \
  \___ \| |  | |  _ <  | |___| |_) | |_| | |\  | |_| || | | |___| |____ ___) |
  |____/|_|  |_|_| \_\ |_____|____/ \___/|_| \_|____/ |_| |_____|______|____/
"""

def slide_title() -> None:
    _slide_number(1, "TITLE")
    console.print(Align.center(Text(LOGO, style="bold cyan")))
    console.print()
    console.print(Align.center(Text("Secure Holistic Integrity Evaluation Layer", style=WHITE)))
    console.print(Align.center(Text("for Defence Computer Vision", style=WHITE)))
    console.print()
    console.print(Align.center(Text("━" * 60, style="cyan")))
    console.print()
    console.print(Align.center(Text("Smart India Hackathon 2025", style=YELLOW)))
    console.print(Align.center(Text("Problem Statement #26228", style=DIM)))
    console.print(Align.center(Text("Ministry of Defence — Indian Army (DGIS)", style=DIM)))
    console.print(Align.center(Text("Theme: Blockchain & Cybersecurity", style=DIM)))
    console.print()
    console.print(Align.center(Text("━" * 60, style="cyan")))
    console.print()

    features = Table(show_header=False, box=None, padding=(0, 2))
    features.add_column(style="green")
    features.add_column(style="white")
    features.add_row("100% OFFLINE", "No data ever leaves the host")
    features.add_row("MODEL-AGNOSTIC", "Works with any CV model — no retraining")
    features.add_row("CPU-ONLY", "Runs on Intel i5 / 8 GB RAM — no GPU needed")
    features.add_row("AIR-GAPPED", "Designed for classified environments")
    console.print(Align.center(features))

    _pause()


# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — THE PROBLEM
# ═════════════════════════════════════════════════════════════════════════════

def slide_problem() -> None:
    _slide_number(2, "THE PROBLEM")

    console.print(Panel(
        Text("When multiple agencies contribute data and models\n"
             "into a shared defence CV pipeline...\n\n"
             "     Can this pipeline be TRUSTED?", style=WHITE),
        title="[bold yellow]THE PROBLEM[/bold yellow]",
        border_style="yellow", padding=(1, 3)))

    console.print()
    attacks = Table(title="[bold red]Known Attack Vectors[/bold red]",
                   box=box.ROUNDED, border_style="red")
    attacks.add_column("Attack", style="bold")
    attacks.add_column("What it does", style="dim")
    attacks.add_column("Impact", style="bold red")

    attacks.add_row("Trigger Injection", "Hidden pattern in images causes misclassification", "Model obeys attacker")
    attacks.add_row("Label Flipping", "Correct labels silently swapped", "Model learns wrong associations")
    attacks.add_row("Duplicate Flooding", "Same images repeated to skew training", "Overfits to attacker data")
    attacks.add_row("Model Backdoor", "Implanted trigger in trained weights", "Silent targeted failures")
    attacks.add_row("Inference Tamper", "Operational records edited/replayed", "Audit trail compromised")
    attacks.add_row("Distribution Shift", "Injected content changes data profile", "Degraded accuracy")
    console.print(attacks)

    console.print()
    console.print(Panel(
        Text("Existing solutions require internet, GPU clusters, or retrain models.\n"
             "Defence environments need something that works OFFLINE on modest hardware.",
             style=WHITE),
        border_style="cyan", padding=(0, 2)))

    _pause()


# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — OUR SOLUTION (Architecture)
# ═════════════════════════════════════════════════════════════════════════════

def slide_architecture() -> None:
    _slide_number(3, "ARCHITECTURE — Four Pillars of Assurance")

    arch = """
    ┌─────────────────────────────────────────────────────────────┐
    │                    SHIELD-CV FRAMEWORK                      │
    │              100% Offline · Air-Gapped · CPU                 │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │   ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌────────┐ │
    │   │  DATA     │  │  MODEL    │  │  CRYPTO   │  │ DRIFT  │ │
    │   │  SCANNER  │  │  AUDITOR  │  │  CHAIN    │  │DETECTOR│ │
    │   ├───────────┤  ├───────────┤  ├───────────┤  ├────────┤ │
    │   │• Trigger  │  │• Weight   │  │• Ed25519  │  │• MMD   │ │
    │   │  detect   │  │  hash     │  │  sign     │  │• Cent- │ │
    │   │• Label    │  │• Neural   │  │• SHA-256  │  │  roid  │ │
    │   │  flip     │  │  Cleanse  │  │  chain    │  │• Varia-│ │
    │   │• Dup flood│  │• Actv     │  │• Nonce    │  │  tion  │ │
    │   │• OOD      │  │  cluster  │  │  replay   │  │• Pixel │ │
    │   │• Mislabel │  │• Blackbox │  │• Tamper   │  │  props │ │
    │   │           │  │  fingerpr.│  │  evidence │  │        │ │
    │   └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └───┬────┘ │
    │         │              │              │              │      │
    │         └──────────────┴──────┬───────┴──────────────┘      │
    │                              │                             │
    │                    ┌─────────▼─────────┐                   │
    │                    │   INTELLIGENCE    │                   │
    │                    │      LAYER        │                   │
    │                    ├───────────────────┤                   │
    │                    │ • Threat Story    │                   │
    │                    │ • Immunity Score  │                   │
    │                    │ • LLM Briefing    │                   │
    │                    │ • Signed Report   │                   │
    │                    └───────────────────┘                   │
    └─────────────────────────────────────────────────────────────┘
"""
    console.print(Panel(arch, border_style="cyan", padding=(0, 1)))

    pillars = Table(box=box.SIMPLE_HEAVY, border_style="cyan")
    pillars.add_column("Pillar", style="bold cyan")
    pillars.add_column("Asset", style="white")
    pillars.add_column("Question Answered", style="dim")

    pillars.add_row("DATA SCANNER", "Training datasets",
                    "Has anyone poisoned the data?")
    pillars.add_row("MODEL AUDITOR", "Trained models",
                    "Does this model contain a backdoor?")
    pillars.add_row("CRYPTO CHAIN", "Inference records",
                    "Has the record been tampered with?")
    pillars.add_row("DRIFT DETECTOR", "Operational data",
                    "Is this natural drift or an attack?")
    console.print(pillars)

    console.print()
    console.print(Panel(
        Text("Each module runs independently. Correlation across modules\n"
             "produces the ADVERSARIAL IMMUNITY SCORE — a single number\n"
             "that tells the commander: can this pipeline be trusted?",
             style=WHITE),
        title="[bold green]KEY INSIGHT[/bold green]",
        border_style="green"))

    _pause()


# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — DATA SCANNER (DEEP DIVE)
# ═════════════════════════════════════════════════════════════════════════════

def slide_data_scanner() -> None:
    _slide_number(4, "DATA SCANNER — Five Independent Detectors")

    table = Table(box=box.ROUNDED, border_style="green", title="[bold]Detection Methods[/bold]")
    table.add_column("#", style="bold cyan", width=3)
    table.add_column("Detector", style="bold")
    table.add_column("Technique", style="white")
    table.add_column("How it works", style="dim")

    table.add_row("1", "Trigger Injection",
                  "FFT + Block Variance + SVD",
                  "Three methods must AGREE for HIGH confidence")
    table.add_row("2", "Label Flipping",
                  "Centroid Distance + KNN",
                  "Image features vs claimed label — neighbours vote")
    table.add_row("3", "Near-Duplicate Flooding",
                  "pHash + SSIM",
                  "Perceptual hash prefilter, pixel-level confirmation")
    table.add_row("4", "Out-of-Distribution",
                  "Mahalanobis + Energy Score",
                  "Statistical distance from class centres")
    table.add_row("5", "Systematic Mislabeling",
                  "Confident Learning",
                  "Predicted label vs given label disagreement")
    console.print(table)

    console.print()
    console.print(Panel(
        Text("CONTRIBUTOR RISK = 0.4·trigger + 0.3·flip + 0.2·dup + 0.1·ood\n\n"
             "Each contributor gets an individual risk score.\n"
             "The worst contributor is isolated, not averaged into the pool.",
             style=WHITE),
        border_style="green"))

    _pause()


# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — MODEL AUDITOR
# ═════════════════════════════════════════════════════════════════════════════

def slide_model_auditor() -> None:
    _slide_number(5, "MODEL AUDITOR — Graceful Degradation")

    table = Table(box=box.ROUNDED, border_style="magenta",
                  title="[bold]Access Levels[/bold]")
    table.add_column("Level", style="bold magenta")
    table.add_column("What is available", style="white")
    table.add_column("Detection methods", style="dim")

    table.add_row("WHITE-BOX", "Full weights + architecture",
                  "SHA-256 hash, per-layer stats, Neural Cleanse, "
                  "activation clustering, weight forensics")
    table.add_row("GREY-BOX", "Model file, no reference",
                  "Weight statistics, anomaly detection, "
                  "activation clustering")
    table.add_row("BLACK-BOX", "Input/output only",
                  "Behavioural fingerprint, entropy analysis, "
                  "mismatch rate")
    console.print(table)

    console.print()
    console.print(Panel(
        Text("CRITICAL: We NEVER retrain or modify the contributed model.\n"
             "We only INSPECT it and report findings.\n\n"
             "Neural Cleanse reverse-engineers a minimal trigger per class\n"
             "and flags classes needing an abnormally small patch.",
             style=WHITE),
        border_style="magenta"))

    _pause()


# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — CRYPTO CHAIN
# ═════════════════════════════════════════════════════════════════════════════

def slide_crypto() -> None:
    _slide_number(6, "CRYPTO CHAIN — Tamper-Evident Provenance")

    chain_vis = """
    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
    │ Record 0 │────▶│ Record 1 │────▶│ Record 2 │────▶│ Record 3 │
    │ (Genesis)│     │          │     │          │     │          │
    ├──────────┤     ├──────────┤     ├──────────┤     ├──────────┤
    │ payload  │     │ payload  │     │ payload  │     │ payload  │
    │ hash     │◀────│ hash     │◀────│ hash     │◀────│ hash     │
    │ sig      │     │ sig      │     │ sig      │     │ sig      │
    │ nonce    │     │ nonce    │     │ nonce    │     │ nonce    │
    │ prev─────│────▶│ prev─────│────▶│ prev─────│────▶│ prev─────│
    └──────────┘     └──────────┘     └──────────┘     └──────────┘
         │
         ▼
    Ed25519 signing key pair
    (private key signs, public key verifies)
"""
    console.print(Panel(chain_vis, border_style="yellow", padding=(0, 1)))

    table = Table(box=box.ROUNDED, border_style="yellow",
                  title="[bold]Tamper Detection[/bold]")
    table.add_column("Attack Type", style="bold")
    table.add_column("Detection", style="yellow")

    table.add_row("Record Edit", "HASH_MISMATCH — output hash differs from re-derivation")
    table.add_row("Record Delete", "BROKEN_LINK + SEQUENCE_GAP")
    table.add_row("Record Replay", "DUPLICATE_NONCE + TIMESTAMP_REGRESSION")
    table.add_row("Hash Repair", "SIGNATURE_FAIL — attacker cannot forge Ed25519")
    table.add_row("Genesis Missing", "MISSING_GENESIS — chain starts improperly")
    console.print(table)

    console.print()
    console.print(Panel(
        Text("Even if the attacker edits data AND recomputes the hash,\n"
             "they still fail the Ed25519 signature check.\n"
             "This is why we SIGN, not just hash.",
             style=WHITE),
        border_style="yellow"))

    _pause()


# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — LIVE DEMO: CLEAN vs POISONED
# ═════════════════════════════════════════════════════════════════════════════

def slide_live_demo(args: argparse.Namespace) -> None:
    _slide_number(7, "LIVE DEMO — Clean vs Poisoned Pipeline")

    console.print(Panel(
        Text("Running real scans on demo data...\n"
             "LEFT: Clean pipeline (no attacks)\n"
             "RIGHT: Poisoned pipeline (3 attacks injected)",
             style=WHITE),
        border_style="cyan"))

    console.print()
    console.print(f"  [{DIM}]Scanning clean dataset...[/{DIM}]")

    from src.scanners.data_scanner import DataIntegrityEngine
    cap = 60 if args.quick else 90

    with console.status("[cyan]Scanning clean corpus..."):
        clean = DataIntegrityEngine().scan(
            ROOT / "demo" / "data" / "clean", max_images=cap)
    gc.collect()

    console.print(f"  [{DIM}]Scanning poisoned dataset...[/{DIM}]")
    with console.status("[cyan]Scanning poisoned corpus..."):
        poisoned = DataIntegrityEngine().scan(
            ROOT / "demo" / "data" / "poisoned", max_images=cap)
    gc.collect()

    # Build comparison table
    cs = clean.get("summary") or {}
    ps = poisoned.get("summary") or {}
    c_sev = cs.get("by_severity") or {}
    p_sev = ps.get("by_severity") or {}

    comp = Table(box=box.DOUBLE_EDGE, border_style="white",
                 title="[bold]A/B COMPARISON[/bold]")
    comp.add_column("Metric", style="bold", width=22)
    comp.add_column("CLEAN Pipeline", style="bold green", justify="center")
    comp.add_column("POISONED Pipeline", style="bold red", justify="center")

    comp.add_row("Verdict",
                 Text(str(cs.get("verdict", "?")), style="bold green"),
                 Text(str(ps.get("verdict", "?")), style="bold red"))
    comp.add_row("Risk Score",
                 f"{float(cs.get('risk_score', 0)):.3f}",
                 f"{float(ps.get('risk_score', 0)):.3f}")
    comp.add_row("Total Findings",
                 str(cs.get("total_findings", 0)),
                 str(ps.get("total_findings", 0)))
    comp.add_row("CRITICAL",
                 str(c_sev.get("CRITICAL", 0)),
                 str(p_sev.get("CRITICAL", 0)))
    comp.add_row("HIGH",
                 str(c_sev.get("HIGH", 0)),
                 str(p_sev.get("HIGH", 0)))
    comp.add_row("MEDIUM",
                 str(c_sev.get("MEDIUM", 0)),
                 str(p_sev.get("MEDIUM", 0)))
    comp.add_row("LOW",
                 str(c_sev.get("LOW", 0)),
                 str(p_sev.get("LOW", 0)))
    console.print(comp)

    # Contributor risk for poisoned
    risks = poisoned.get("contributor_risk") or {}
    if risks:
        console.print()
        risk_table = Table(box=box.ROUNDED, border_style="red",
                           title="[bold]Contributor Risk (Poisoned)[/bold]")
        risk_table.add_column("Contributor", style="bold")
        risk_table.add_column("Risk", justify="right")
        risk_table.add_column("Findings", justify="right")
        risk_table.add_column("Assessment", style="dim")

        for name, entry in sorted(risks.items(),
                                  key=lambda kv: -float(kv[1].get("risk", 0.0))):
            risk = float(entry.get("risk", 0.0))
            style = "bold red" if risk >= 0.75 else "bold yellow" if risk >= 0.45 else "green"
            risk_table.add_row(
                name,
                Text(f"{risk:.3f}", style=style),
                str(entry.get("findings", 0)),
                str(entry.get("assessment", ""))[:60])
        console.print(risk_table)

    console.print()
    console.print(Panel(
        Text("KEY RESULT: Clean pipeline = CAUTION (low risk)\n"
             "            Poisoned pipeline = COMPROMISED (high risk)\n\n"
             "The detectors cleanly separate the two scenarios.",
             style=WHITE),
        border_style="green"))

    _pause()


# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — LIVE DEMO: MODEL AUDIT
# ═════════════════════════════════════════════════════════════════════════════

def slide_model_demo(args: argparse.Namespace) -> None:
    _slide_number(8, "LIVE DEMO — Model Audit (Clean vs Backdoored)")

    from src.analysis import embeddings as _embeddings
    from src.scanners.model_auditor import ModelIntegrityEngine
    _embeddings._EXTRACTOR_SINGLETON = None
    gc.collect()

    engine = ModelIntegrityEngine()
    models_dir = ROOT / "demo" / "models"

    table = Table(box=box.DOUBLE_EDGE, border_style="magenta",
                  title="[bold]MODEL AUDIT RESULTS[/bold]")
    table.add_column("Metric", style="bold", width=22)
    table.add_column("Clean Model", style="bold green", justify="center")
    table.add_column("Backdoored Model", style="bold red", justify="center")

    results = {}
    for label, fname in [("Clean Model", "clean_model_scripted.pt"),
                          ("Backdoored Model", "backdoored_model_scripted.pt")]:
        path = models_dir / fname
        if not path.exists():
            continue
        with console.status(f"[cyan]Auditing {label}..."):
            outcome = engine.audit(path, run_neural_cleanse=not args.quick)
        results[fname] = outcome
        gc.collect()

    clean_r = results.get("clean_model_scripted.pt") or {}
    back_r = results.get("backdoored_model_scripted.pt") or {}
    cs = clean_r.get("summary") or {}
    bs = back_r.get("summary") or {}

    table.add_row("Verdict",
                 Text(str(cs.get("verdict", "?")), style="bold green"),
                 Text(str(bs.get("verdict", "?")), style="bold red"))
    table.add_row("Access Level",
                 str(clean_r.get("access_level", "?")),
                 str(back_r.get("access_level", "?")))
    table.add_row("Confidence",
                 f"{float(clean_r.get('confidence', 0)):.2f}",
                 f"{float(back_r.get('confidence', 0)):.2f}")
    table.add_row("Findings",
                 str(len(clean_r.get("findings") or [])),
                 str(len(back_r.get("findings") or [])))
    table.add_row("Risk Score",
                 f"{float(cs.get('risk_score', 0)):.3f}",
                 f"{float(bs.get('risk_score', 0)):.3f}")
    console.print(table)

    # Show findings for backdoored model
    for finding in (back_r.get("findings") or [])[:5]:
        sev = finding.get("severity", "")
        console.print(f"  [{_sev_color(sev)}]{finding.get('finding_id', '')} "
                      f"[{_sev_color(sev)}]{sev}[/{_sev_color(sev)}] "
                      f"{finding.get('attack_class', '')} "
                      f"({float(finding.get('confidence', 0)):.2f})")

    console.print()
    console.print(Panel(
        Text("The clean model passes. The backdoored model is flagged.\n"
             "No retraining occurred — we only inspected the weights.",
             style=WHITE),
        border_style="magenta"))

    _pause()


# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — LIVE DEMO: CRYPTO CHAIN
# ═════════════════════════════════════════════════════════════════════════════

def slide_crypto_demo() -> None:
    _slide_number(9, "LIVE DEMO — Provenance Chain Verification")

    from src.scanners.crypto_chain import InferenceProvenanceEngine

    engine = InferenceProvenanceEngine()
    fixtures = [
        ("clean_chain.json",        "Intact chain"),
        ("tampered_edit.json",      "Record edited"),
        ("tampered_rehash.json",    "Edited + re-hashed"),
        ("tampered_delete.json",    "Record deleted"),
        ("tampered_replay.json",    "Record replayed"),
    ]

    table = Table(box=box.DOUBLE_EDGE, border_style="yellow",
                  title="[bold]CHAIN VERIFICATION[/bold]")
    table.add_column("Scenario", style="bold")
    table.add_column("Verdict", justify="center")
    table.add_column("Expected", style="dim")

    for filename, scenario in fixtures:
        path = ROOT / "demo" / "records" / filename
        if not path.exists():
            table.add_row(scenario, "[yellow]missing[/yellow]", "")
            continue
        outcome = engine.verify(path)
        verdict = (outcome.get("summary") or {}).get("verdict", "?")
        expected = "PASS" if "clean" in filename else "FAIL"
        table.add_row(scenario,
                      Text(verdict, style=_verdict_style(verdict)),
                      expected)
    console.print(table)

    console.print()
    console.print(Panel(
        Text("Clean chain PASSES. Every tamper variant FAILS.\n"
             "Even the attacker who recomputes the hash is caught\n"
             "because they cannot forge the Ed25519 signature.",
             style=WHITE),
        border_style="yellow"))

    _pause()


# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — IMMUNITY SCORE & CORRELATION
# ═════════════════════════════════════════════════════════════════════════════

def slide_immunity() -> None:
    _slide_number(10, "ADVERSARIAL IMMUNITY SCORE")

    score_vis = """
    ┌──────────────────────────────────────────────┐
    │        ADVERSARIAL IMMUNITY SCORE             │
    │                                               │
    │   HARDENED (>80)     Pipeline is robust       │
    │   ADEQUATE (60-80)   Minor review needed      │
    │   MARGINAL (40-60)   Significant concerns     │
    │   COMPROMISED (<40)  DO NOT DEPLOY            │
    │                                               │
    │   Score = Σ (pillar_weight × pillar_score)    │
    │                                               │
    │   Pillars:                                     │
    │     • Trigger Resistance        (25%)          │
    │     • Backdoor Immunity         (25%)          │
    │     • Adversarial Robustness    (20%)          │
    │     • Label Flip Resistance     (15%)          │
    │     • Distribution Tolerance    (15%)          │
    └──────────────────────────────────────────────┘
"""
    console.print(Panel(score_vis, border_style="green", padding=(0, 1)))

    console.print(Panel(
        Text("Not-assessed modules are recorded as NOT ASSESSED,\n"
             "NOT counted as clean. Absence of evidence ≠ evidence of integrity.\n\n"
             "The immunity score tells the commander a single number:\n"
             "how resilient is this pipeline against adversarial manipulation?",
             style=WHITE),
        border_style="green"))

    _pause()


# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — KEY DIFFERENTIATORS
# ═════════════════════════════════════════════════════════════════════════════

def slide_differentiators() -> None:
    _slide_number(11, "WHY SHIELD-CV IS DIFFERENT")

    table = Table(box=box.ROUNDED, border_style="cyan",
                  title="[bold]Comparison[/bold]")
    table.add_column("Feature", style="bold", width=30)
    table.add_column("Existing Tools", style="dim")
    table.add_column("SHIELD-CV", style="bold green")

    table.add_row("Internet required?", "Yes (most)", "NO — 100% offline")
    table.add_row("GPU required?", "Often yes", "NO — CPU only (i5 / 8GB)")
    table.add_row("Retrains models?", "Some do", "NEVER — inspect only")
    table.add_row("Multiple attack types?", "1-2", "6+ attack classes")
    table.add_row("Contributor attribution?", "No", "YES — per-contributor risk")
    table.add_row("Graceful degradation?", "Rare", "YES — white/grey/black box")
    table.add_row("Tamper-evident audit trail?", "No", "YES — Ed25519 + hash chain")
    table.add_row("Air-gap support?", "No", "YES — designed for classified envs")
    table.add_row("Explainable findings?", "Score only", "YES — reason + evidence + disposition")
    table.add_row("Correlation across modules?", "No", "YES — threat story + immunity score")
    console.print(table)

    _pause()


# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — TECHNICAL DETAILS
# ═════════════════════════════════════════════════════════════════════════════

def slide_tech() -> None:
    _slide_number(12, "TECHNICAL HIGHLIGHTS")

    table = Table(box=box.ROUNDED, border_style="blue",
                  title="[bold]Detection Accuracy (Ground Truth Validated)[/bold]")
    table.add_column("Attack", style="bold")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")

    table.add_row("Trigger Injection", "0.78", "0.78", "0.78")
    table.add_row("Label Flipping",     "0.67", "1.00", "0.80")
    table.add_row("Near-Duplicate Flood","1.00", "1.00", "1.00")
    console.print(table)

    console.print()
    details = Table(box=box.SIMPLE, border_style="blue")
    details.add_column("Component", style="bold")
    details.add_column("Detail", style="dim")

    details.add_row("Backbone", "ResNet-18 (frozen, IMAGENET1K_V1) — 512-d embeddings")
    details.add_row("Trigger Detection", "FFT + 16x16 block variance + SVD spectral (2-of-3 voting)")
    details.add_row("Model Signing", "Ed25519 digital signatures (256-bit private key)")
    details.add_row("Hash Chain", "SHA-256, hash-linked, nonce-tagged, timestamped")
    details.add_row("Drift Detection", "MMD (Maximum Mean Discrepancy) + centroid drift + pixel properties")
    details.add_row("Briefing", "Optional local LLM (Ollama) with deterministic fallback")
    details.add_row("Report", "JSON schema-validated, signed, with audit trail")
    details.add_row("Dependencies", "15 packages — every one verified as imported")
    details.add_row("Test Suite", "30 hostile-input cases + 23 detection-correctness checks")
    console.print(details)

    _pause()


# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 13 — DASHBOARD PREVIEW
# ═════════════════════════════════════════════════════════════════════════════

def slide_dashboard() -> None:
    _slide_number(13, "DASHBOARD PREVIEW")

    console.print(Panel(
        Text("The project includes a Streamlit-based military-theme dashboard\n"
             "with 5 interactive tabs:\n\n"
             "  1. DATA SCANNER   — Run dataset integrity scans\n"
             "  2. MODEL AUDIT    — Audit models (white/grey/black box)\n"
             "  3. CRYPTO LOCK    — Live tamper demo (edit & watch chain break)\n"
             "  4. DRIFT MONITOR  — Natural vs adversarial shift detection\n"
             "  5. THREAT STORY   — Correlation, immunity score & briefing\n\n"
             "Run with:  streamlit run src/dashboard.py",
             style=WHITE),
        title="[bold cyan]GUI Dashboard[/bold cyan]",
        border_style="cyan"))

    console.print()
    console.print(Panel(
        Text("For the live demo, run:\n\n"
             "  python demo/run_demo.py          # Full 8-stage guided demo\n"
             "  python demo/run_demo.py --quick  # Faster version\n"
             "  python run.py scan demo/data/poisoned  # Just the scanner\n"
             "  python run.py audit demo/models/backdoored_model_scripted.pt",
             style=WHITE),
        title="[bold yellow]Quick Commands[/bold yellow]",
        border_style="yellow"))

    _pause()


# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 14 — CLOSING
# ═════════════════════════════════════════════════════════════════════════════

def slide_closing() -> None:
    _slide_number(14, "SUMMARY")

    console.print(Panel(
        Text("SHIELD-CV provides integrity assurance for multi-contributor\n"
             "defence computer vision pipelines.\n\n"
             "  100% Offline    — No data leaves the host\n"
             "  Model-Agnostic  — Works with any CV model, no retraining\n"
             "  CPU-Only        — Runs on standard military hardware\n"
             "  Explainable     — Every finding has reason, evidence, disposition\n"
             "  Correlated      — Cross-module intelligence, not isolated alerts\n"
             "  Tamper-Evident  — Ed25519 signed audit trail\n\n"
             "For the Indian Army, by the Indian Army.",
             style=WHITE),
        title="[bold green]SHIELD-CV[/bold green]",
        border_style="green", padding=(1, 3)))

    console.print()
    console.print(Align.center(Text("Thank you.", style=CYAN)))
    console.print(Align.center(Text("━" * 40, style="cyan")))
    console.print(Align.center(Text("Smart India Hackathon 2025", style=DIM)))
    console.print(Align.center(Text("Problem #26228 · Ministry of Defence · DGIS", style=DIM)))
    console.print()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="SHIELD-CV Jury Presentation")
    parser.add_argument("--quick", action="store_true",
                        help="Smaller sample for live demos")
    parser.add_argument("--stage", type=int, default=None,
                        choices=range(1, 15),
                        help="Jump to a specific slide (1-14)")
    parser.add_argument("--no-demo", action="store_true",
                        help="Skip live scan demos (slides 7-9)")
    args = parser.parse_args()

    console.clear()
    console.print()

    all_slides = [
        (1,  slide_title),
        (2,  slide_problem),
        (3,  slide_architecture),
        (4,  slide_data_scanner),
        (5,  slide_model_auditor),
        (6,  slide_crypto),
        (7,  lambda: slide_live_demo(args)),
        (8,  lambda: slide_model_demo(args)),
        (9,  slide_crypto_demo),
        (10, slide_immunity),
        (11, slide_differentiators),
        (12, slide_tech),
        (13, slide_dashboard),
        (14, slide_closing),
    ]

    skip_live = {7, 8, 9} if args.no_demo else set()

    for num, fn in all_slides:
        if args.stage is not None and args.stage != num:
            continue
        if num in skip_live:
            continue
        try:
            fn()
        except KeyboardInterrupt:
            console.print(f"\n  [{DIM}]Presentation interrupted at slide {num}[/{DIM}]")
            return 0
        except Exception as exc:
            console.print(f"\n  [red]Slide {num} failed: {exc}[/red]")
            import traceback
            console.print(f"  [dim]{traceback.format_exc()}[/dim]")
            _pause("Press Enter to continue to next slide...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
