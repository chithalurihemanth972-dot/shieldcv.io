#!/usr/bin/env python3
"""SHIELD-CV Interactive CLI — Shield Command Tool.

Type commands and get instant results:

    shield scan      → scan dataset
    shield audit     → audit model
    shield verify    → verify chain
    shield report    → generate full report
    shield demo      → run guided demo
    shield dashboard → open GUI

Run:  python shield.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# ── Banner ───────────────────────────────────────────────────────────────────

BANNER = r"""
  ███████╗██╗  ██╗██╗███████╗██╗     ██████╗ ███████╗██╗  ██╗
  ██╔════╝██║  ██║██║██╔════╝██║     ██╔══██╗██╔════╝██║  ██║
  ███████╗███████║██║█████╗  ██║     ██║  ██║███████╗███████║
  ╚════██║██╔══██║██║██╔══╝  ██║     ██║  ██║╚════██║██╔══██║
  ███████║██║  ██║██║███████╗███████╗██████╔╝███████║██║  ██║
  ╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═════╝ ╚══════╝╚═╝  ╚═╝
  ─── Secure Holistic Integrity Evaluation Layer ────────────────
"""

def show_banner() -> None:
    console.clear()
    console.print(Align.center(Text(BANNER, style="bold cyan")))
    console.print(Align.center(Text("Defence Computer Vision Integrity Assurance", style="dim")))
    console.print(Align.center(Text("100% Offline · Air-Gapped · CPU-Only", style="dim")))
    console.print()

# ── Menu ─────────────────────────────────────────────────────────────────────

def show_menu() -> None:
    table = Table(box=box.ROUNDED, border_style="cyan", show_header=False,
                  padding=(0, 2))
    table.add_column("Command", style="bold cyan", width=18)
    table.add_column("Description", style="white")

    table.add_row("shield scan",     "Scan a dataset for poison attacks")
    table.add_row("shield audit",    "Audit a model for backdoors")
    table.add_row("shield verify",   "Verify inference provenance chain")
    table.add_row("shield drift",    "Detect distribution shift")
    table.add_row("shield report",   "Generate full assurance report")
    table.add_row("shield demo",     "Run 8-stage guided demo")
    table.add_row("shield dashboard","Open Streamlit GUI dashboard")
    table.add_row("shield help",     "Show detailed help")
    table.add_row("exit",            "Quit the tool")
    console.print(table)
    console.print()

# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_input(prompt: str, default: str = "") -> str:
    try:
        val = input(f"  {prompt}: ").strip()
    except (EOFError, KeyboardInterrupt):
        return default
    return val if val else default


def _resolve_path(raw: str) -> Path:
    """Resolve path — handles both absolute and relative paths, strips quotes."""
    cleaned = raw.strip().strip('"').strip("'")
    p = Path(cleaned)
    if p.is_absolute():
        return p
    return ROOT / p

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_scan() -> None:
    """Run data integrity scan with visual step-by-step flow."""
    console.print(Panel("[bold cyan]SHIELD SCAN[/bold cyan] — Dataset Integrity Check\n\n"
                        "  Step 1: Load dataset + enumerate images\n"
                        "  Step 2: Extract embeddings (ResNet-18 frozen backbone)\n"
                        "  Step 3: Trigger injection detection (FFT + Block Var + SVD)\n"
                        "  Step 4: Label flipping detection (Centroid + KNN)\n"
                        "  Step 5: Near-duplicate flooding detection (pHash + SSIM)\n"
                        "  Step 6: Out-of-distribution detection (Mahalanobis + Energy)\n"
                        "  Step 7: Systematic mislabeling detection\n"
                        "  Step 8: Contributor risk aggregation + report",
                        border_style="cyan"))

    path = _get_input("Dataset path", "demo/data/poisoned")
    limit = _get_input("Max images (0=all)", "0")
    max_imgs = int(limit) if limit.isdigit() and int(limit) > 0 else None

    target = _resolve_path(path)
    if not target.exists():
        console.print(f"\n  [red]Path not found: {target}[/red]")
        return

    console.print()
    console.print(f"  [bold cyan]Target:[/bold cyan] {target}")

    # ── Step-by-step visual scan ──
    total_steps = 8
    console.print()
    console.print(f"  [bold]Starting scan pipeline...[/bold]")
    console.print()

    _step(1, total_steps, f"Loading dataset from {target.name}/")
    _step(2, total_steps, "Extracting embeddings (ResNet-18, frozen, 512-d)")

    from src.scanners.data_scanner import DataIntegrityEngine

    t0 = time.time()
    result = DataIntegrityEngine().scan(str(target), max_images=max_imgs)
    elapsed = time.time() - t0

    _step(3, total_steps, "Trigger injection detection (FFT + Block Variance + SVD)", "done")
    _step(4, total_steps, "Label flipping detection (Centroid Distance + KNN)", "done")
    _step(5, total_steps, "Near-duplicate flooding detection (pHash + SSIM)", "done")
    _step(6, total_steps, "Out-of-distribution detection (Mahalanobis + Energy Score)", "done")
    _step(7, total_steps, "Systematic mislabeling detection (Confident Learning)", "done")

    summary = result.get("summary") or {}
    verdict = str(summary.get("verdict", "?"))
    risk = float(summary.get("risk_score", 0))
    findings = summary.get("total_findings", 0)
    counts = summary.get("by_severity") or {}
    _step(8, total_steps, f"Verdict: {verdict} (risk={risk:.3f}, findings={findings})", "done")

    summary = result.get("summary") or {}
    verdict = str(summary.get("verdict", "?"))
    risk = float(summary.get("risk_score", 0))
    findings = summary.get("total_findings", 0)
    counts = summary.get("by_severity") or {}

    # Result panel
    v_style = {"COMPROMISED": "bold white on red", "SUSPICIOUS": "bold dark_orange",
               "CAUTION": "bold yellow", "CLEAN": "bold white on green"
               }.get(verdict.upper(), "white")

    result_table = Table(box=box.HEAVY_EDGE, border_style="cyan",
                         title="[bold]SCAN RESULTS[/bold]")
    result_table.add_column("Metric", style="bold", width=20)
    result_table.add_column("Value")
    result_table.add_row("Verdict",    Text(verdict, style=v_style))
    result_table.add_row("Risk Score", f"{risk:.3f}")
    result_table.add_row("Findings",   str(findings))
    result_table.add_row("CRITICAL",   str(counts.get("CRITICAL", 0)))
    result_table.add_row("HIGH",       str(counts.get("HIGH", 0)))
    result_table.add_row("MEDIUM",     str(counts.get("MEDIUM", 0)))
    result_table.add_row("LOW",        str(counts.get("LOW", 0)))
    result_table.add_row("Time",       f"{elapsed:.1f}s")
    console.print(result_table)

    # Contributor risk
    risks = result.get("contributor_risk") or {}
    if risks:
        console.print()
        risk_table = Table(box=box.SIMPLE, border_style="cyan",
                           title="[bold]Contributor Risk[/bold]")
        risk_table.add_column("Contributor", style="bold")
        risk_table.add_column("Risk", justify="right")
        risk_table.add_column("Findings", justify="right")
        for name, entry in sorted(risks.items(),
                                  key=lambda kv: -float(kv[1].get("risk", 0.0))):
            r = float(entry.get("risk", 0.0))
            style = "bold red" if r >= 0.75 else "bold yellow" if r >= 0.45 else "green"
            risk_table.add_row(name, Text(f"{r:.3f}", style=style),
                               str(entry.get("findings", 0)))
        console.print(risk_table)

    # Top findings
    findings_list = result.get("findings") or []
    if findings_list:
        console.print()
        console.print("  [bold]Top findings:[/bold]")
        for f in findings_list[:8]:
            sev = f.get("severity", "")
            sev_style = {"CRITICAL": "bold red", "HIGH": "bold dark_orange",
                         "MEDIUM": "bold yellow", "LOW": "blue"}.get(sev, "dim")
            console.print(f"    [{sev_style}]{f.get('finding_id',''):10s} {sev:8s}[/{sev_style}] "
                          f"{f.get('attack_class',''):30s} "
                          f"conf={float(f.get('confidence',0)):.2f}")

    # Save JSON
    out_dir = ROOT / "output" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"SHIELD-SCAN-{_timestamp()}.json"
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    console.print(f"\n  [green]Report saved:[/green] {out_file}")


def _step(num: int, total: int, label: str, status: str = "running") -> None:
    """Show a single audit step with progress."""
    icons = {"running": "[cyan]...[/cyan]", "done": "[green]OK[/green]",
             "skip": "[yellow]SKIP[/yellow]", "fail": "[red]FAIL[/red]"}
    icon = icons.get(status, "[dim]?[/dim]")
    console.print(f"  [dim]{num}/{total}[/dim] {icon} {label}")


def cmd_audit() -> None:
    """Run model audit with visual step-by-step flow."""
    console.print(Panel("[bold magenta]SHIELD AUDIT[/bold magenta] — Model Integrity Check\n\n"
                        "  Step 1: Load model + detect access level\n"
                        "  Step 2: SHA-256 weight hash + substitution check\n"
                        "  Step 3: Per-layer statistics (std, kurtosis, sparsity)\n"
                        "  Step 4: Neural Cleanse — reverse-engineer triggers\n"
                        "  Step 5: Activation clustering — KMeans k=2 imbalance\n"
                        "  Step 6: Black-box behavioural fingerprint\n"
                        "  Step 7: Generate verdict + report",
                        border_style="magenta"))

    path = _get_input("Model path", "demo/models/backdoored_model_scripted.pt")
    ref = _get_input("Reference model (Enter=skip)", "")
    skip_nc = _get_input("Skip Neural Cleanse? (y/n)", "n").lower() == "y"

    target = _resolve_path(path)
    if not target.exists():
        console.print(f"\n  [red]Model not found: {target}[/red]")
        return

    reference = _resolve_path(ref) if ref else None
    if reference and not reference.exists():
        console.print(f"  [yellow]Reference not found, skipping: {reference}[/yellow]")
        reference = None

    console.print()
    console.print(f"  [bold magenta]Target:[/bold magenta]  {target.name}")
    if reference:
        console.print(f"  [bold magenta]Reference:[/bold magenta] {reference.name}")
    console.print()

    # ── Visual progress steps ──
    total_steps = 7
    console.print(f"  [bold]Starting audit pipeline...[/bold]")
    console.print()

    _step(1, total_steps, f"Loading model: {target.name}")
    _step(2, total_steps, "Computing SHA-256 weight hash")
    _step(3, total_steps, "Analyzing per-layer weight statistics")
    if not skip_nc:
        _step(4, total_steps, "Running Neural Cleanse (trigger reverse-engineering)")
    else:
        _step(4, total_steps, "Neural Cleanse — skipped by user", "skip")
    _step(5, total_steps, "Activation clustering (KMeans k=2, imbalance check)")
    _step(6, total_steps, "Black-box behavioural fingerprinting")
    _step(7, total_steps, "Running full audit engine...")
    console.print()

    # ── Run actual audit ──
    from src.scanners.model_auditor import ModelIntegrityEngine
    from src.analysis import embeddings as _embeddings

    _embeddings._EXTRACTOR_SINGLETON = None
    gc.collect()

    t0 = time.time()
    engine = ModelIntegrityEngine()
    result = engine.audit(
        str(target),
        reference_path=str(reference) if reference else None,
        run_neural_cleanse=not skip_nc)
    elapsed = time.time() - t0

    # ── Results ──
    summary = result.get("summary") or {}
    verdict = str(summary.get("verdict", "?"))
    access = str(result.get("access_level", "?"))
    confidence = float(result.get("confidence", 0))

    v_style = {"COMPROMISED": "bold white on red", "SUSPICIOUS": "bold dark_orange",
               "CAUTION": "bold yellow", "CLEAN": "bold white on green"
               }.get(verdict.upper(), "white")

    result_table = Table(box=box.HEAVY_EDGE, border_style="magenta",
                         title="[bold]AUDIT RESULTS[/bold]")
    result_table.add_column("Metric", style="bold", width=20)
    result_table.add_column("Value")
    result_table.add_row("Verdict",    Text(verdict, style=v_style))
    result_table.add_row("Access",     access)
    result_table.add_row("Confidence", f"{confidence:.2f}")
    result_table.add_row("Findings",   str(len(result.get("findings") or [])))
    result_table.add_row("Risk Score", f"{float(summary.get('risk_score', 0)):.3f}")
    result_table.add_row("Time",       f"{elapsed:.1f}s")
    console.print(result_table)

    # Neural Cleanse info
    nc = result.get("neural_cleanse") or {}
    if nc.get("available"):
        flagged_classes = nc.get("flagged_classes") or []
        console.print()
        if flagged_classes:
            console.print("  [bold red]Neural Cleanse — BACKDOOR DETECTED:[/bold red]")
            for fc in flagged_classes:
                console.print(f"    [red]Class {fc.get('target_class', '?')}[/red] — "
                              f"anomaly index {fc.get('anomaly_index', 0):.2f}, "
                              f"attack success {fc.get('attack_success_rate', 0):.0%}, "
                              f"L1 norm {fc.get('mask_l1', 0):.2f} vs median {fc.get('median_mask_l1', 0):.2f}")
        else:
            console.print("  [green]Neural Cleanse — no backdoor triggers found[/green]")

    # Findings
    findings_list = result.get("findings") or []
    if findings_list:
        console.print()
        console.print("  [bold]Findings:[/bold]")
        for f in findings_list[:10]:
            sev = f.get("severity", "")
            sev_style = {"CRITICAL": "bold red", "HIGH": "bold dark_orange",
                         "MEDIUM": "bold yellow", "LOW": "blue"}.get(sev, "dim")
            console.print(f"    [{sev_style}]{f.get('finding_id',''):10s} {sev:8s}[/{sev_style}] "
                          f"{f.get('attack_class',''):30s} "
                          f"conf={float(f.get('confidence',0)):.2f}")

    # Limitations
    limitations = result.get("limitations") or []
    if limitations:
        console.print()
        console.print("  [bold yellow]Limitations:[/bold yellow]")
        for lim in limitations[:5]:
            console.print(f"    [yellow]! {lim}[/yellow]")

    # Save
    out_dir = ROOT / "output" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"SHIELD-AUDIT-{_timestamp()}.json"
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    console.print(f"\n  [green]Report saved:[/green] {out_file}")


def cmd_verify() -> None:
    """Verify provenance chain with visual flow."""
    console.print(Panel("[bold yellow]SHIELD VERIFY[/bold yellow] — Provenance Chain Check\n\n"
                        "  Step 1: Load signed records from JSON\n"
                        "  Step 2: Verify genesis record (first in chain)\n"
                        "  Step 3: Check hash links (each record → next)\n"
                        "  Step 4: Verify Ed25519 signatures\n"
                        "  Step 5: Check nonce uniqueness + timestamps\n"
                        "  Step 6: Re-derive output hashes (detect payload edit)\n"
                        "  Step 7: Generate chain integrity report",
                        border_style="yellow"))

    path = _get_input("Records file", "demo/records/clean_chain.json")
    target = _resolve_path(path)
    if not target.exists():
        console.print(f"\n  [red]File not found: {target}[/red]")
        return

    console.print()
    console.print(f"  [bold yellow]Target:[/bold yellow] {target}")

    # ── Step-by-step visual verify ──
    total_steps = 7
    console.print()
    console.print(f"  [bold]Starting chain verification...[/bold]")
    console.print()

    _step(1, total_steps, f"Loading {target.name}")

    from src.scanners.crypto_chain import InferenceProvenanceEngine

    t0 = time.time()
    result = InferenceProvenanceEngine().verify(str(target))
    elapsed = time.time() - t0

    valid = bool(result.get("chain_valid", False))
    num = result.get("num_records", 0)
    first_break = result.get("first_break_index")

    _step(2, total_steps, "Genesis record check", "done")
    _step(3, total_steps, "Hash link verification", "done")
    _step(4, total_steps, "Ed25519 signature verification", "done")
    _step(5, total_steps, "Nonce + timestamp checks", "done")
    _step(6, total_steps, "Payload re-derivation", "done")

    if valid:
        _step(7, total_steps, f"Chain VALID — {num} records, no breaks", "done")
    else:
        _step(7, total_steps, f"Chain BROKEN at index {first_break}", "fail")

    valid = bool(result.get("chain_valid", False))
    num = result.get("num_records", 0)
    first_break = result.get("first_break_index")

    result_table = Table(box=box.HEAVY_EDGE, border_style="yellow",
                         title="[bold]CHAIN VERIFICATION[/bold]")
    result_table.add_column("Metric", style="bold", width=20)
    result_table.add_column("Value")
    result_table.add_row("Chain Valid",
                         Text("VALID" if valid else "BROKEN",
                              style="bold green" if valid else "bold red"))
    result_table.add_row("Records",    str(num))
    result_table.add_row("First Break", str(first_break) if first_break is not None else "none")
    result_table.add_row("Time",       f"{elapsed:.1f}s")
    console.print(result_table)

    findings_list = result.get("findings") or []
    if findings_list:
        console.print()
        console.print("  [bold]Breaks detected:[/bold]")
        for f in findings_list[:10]:
            sev = f.get("severity", "")
            sev_style = {"CRITICAL": "bold red", "HIGH": "bold dark_orange"}.get(sev, "dim")
            console.print(f"    [{sev_style}]{f.get('finding_id',''):10s} {sev:8s}[/{sev_style}] "
                          f"{f.get('attack_class',''):25s} "
                          f"{str(f.get('affected_asset',''))[:50]}")

    # Save
    out_dir = ROOT / "output" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"SHIELD-VERIFY-{_timestamp()}.json"
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    console.print(f"\n  [green]Report saved:[/green] {out_file}")


def cmd_drift() -> None:
    """Run drift detection."""
    console.print(Panel("[bold blue]SHIELD DRIFT[/bold blue] — Distribution Shift Detection",
                        border_style="blue"))

    baseline = _get_input("Baseline corpus", "demo/data/clean")
    current = _get_input("Current corpus", "demo/data/poisoned")
    max_imgs = int(_get_input("Max images", "120"))

    base_path = _resolve_path(baseline)
    cur_path = _resolve_path(current)
    if not base_path.exists() or not cur_path.exists():
        console.print(f"  [red]Path(s) not found[/red]")
        return

    console.print()
    console.print(f"  [blue]Comparing: {baseline}  vs  {current}[/blue]")
    console.print()

    from src.scanners.drift_detector import DistributionShiftDetector

    t0 = time.time()
    with console.status("[blue]Comparing pixel properties and feature distributions..."):
        result = DistributionShiftDetector().compare(
            str(base_path), str(cur_path), max_images=max_imgs)
    elapsed = time.time() - t0

    shift_type = str(result.get("shift_type", "?"))
    risk = float(result.get("risk_score", 0))
    confidence = float(result.get("confidence", 0))

    s_style = {"SUSPICIOUS_MANIPULATION": "bold red", "NATURAL_OPERATIONAL_DRIFT": "bold green",
               "MIXED": "bold yellow"}.get(shift_type, "white")

    result_table = Table(box=box.HEAVY_EDGE, border_style="blue",
                         title="[bold]DRIFT RESULTS[/bold]")
    result_table.add_column("Metric", style="bold", width=20)
    result_table.add_column("Value")
    result_table.add_row("Shift Type",  Text(shift_type, style=s_style))
    result_table.add_row("Risk Score",  f"{risk:.3f}")
    result_table.add_row("Confidence",  f"{confidence:.2f}")
    result_table.add_row("Severity",    str(result.get("severity", "?")))
    result_table.add_row("Time",        f"{elapsed:.1f}s")
    console.print(result_table)

    console.print(f"\n  [bold]Characterization:[/bold] {result.get('characterization', '')}")
    console.print(f"  [bold]Recommendation:[/bold] {result.get('recommendation', '')}")

    # Save
    out_dir = ROOT / "output" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"SHIELD-DRIFT-{_timestamp()}.json"
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    console.print(f"\n  [green]Report saved:[/green] {out_file}")


def cmd_report() -> None:
    """Generate full assurance report with visual flow."""
    console.print(Panel("[bold green]SHIELD REPORT[/bold green] — Full Assurance Report\n\n"
                        "  Step 1:  Data Scanner   — scan dataset for poison attacks\n"
                        "  Step 2:  Model Auditor  — audit model for backdoors\n"
                        "  Step 3:  Crypto Chain   — verify inference provenance\n"
                        "  Step 4:  Drift Detector — check distribution shift\n"
                        "  Step 5:  Threat Story   — correlate findings across modules\n"
                        "  Step 6:  Immunity Score — compute adversarial immunity\n"
                        "  Step 7:  Briefing       — generate commander's briefing\n"
                        "  Step 8:  Report         — sign + save final report",
                        border_style="green"))

    console.print()
    console.print("  [bold green]Running all 8 modules...[/bold green]")
    console.print()

    from src.scanners.data_scanner import DataIntegrityEngine
    from src.scanners.model_auditor import ModelIntegrityEngine
    from src.scanners.crypto_chain import InferenceProvenanceEngine
    from src.scanners.drift_detector import DistributionShiftDetector
    from src.intelligence.threat_story import build_threat_story
    from src.intelligence.immunity import compute_immunity_score
    from src.intelligence.briefing import generate_briefing
    from src.reporting.report_generator import generate_report

    modules: Dict[str, Any] = {}
    t0 = time.time()
    total = 8

    # Step 1: Data scan
    _step(1, total, "Scanning dataset for poison attacks...")
    data_path = ROOT / "demo" / "data" / "poisoned"
    if data_path.exists():
        with console.status("[cyan]Running 5 data detectors..."):
            modules["data_scanner"] = DataIntegrityEngine().scan(str(data_path))
        _step(1, total, f"Data scan done — {len(modules.get('data_scanner',{}).get('findings',[]))} findings", "done")
    else:
        _step(1, total, "Dataset not found — skipped", "skip")
    gc.collect()

    # Step 2: Model audit
    _step(2, total, "Auditing model for backdoors...")
    model_path = ROOT / "demo" / "models" / "backdoored_model_scripted.pt"
    if model_path.exists():
        with console.status("[magenta]Running weight hash + Neural Cleanse..."):
            modules["model_auditor"] = ModelIntegrityEngine().audit(
                str(model_path), run_neural_cleanse=True)
        _step(2, total, f"Model audit done — {len(modules.get('model_auditor',{}).get('findings',[]))} findings", "done")
        gc.collect()
    else:
        _step(2, total, "Model not found — skipped", "skip")

    # Step 3: Chain verify
    _step(3, total, "Verifying inference provenance chain...")
    chain_path = ROOT / "demo" / "records" / "tampered_replay.json"
    if chain_path.exists():
        with console.status("[yellow]Checking hash links + signatures..."):
            modules["crypto_chain"] = InferenceProvenanceEngine().verify(str(chain_path))
        chain_valid = modules.get("crypto_chain", {}).get("chain_valid", False)
        _step(3, total, f"Chain verify done — {'VALID' if chain_valid else 'BROKEN'}", "done")
    else:
        _step(3, total, "Chain file not found — skipped", "skip")

    # Step 4: Drift
    _step(4, total, "Detecting distribution shift...")
    drift_base = ROOT / "demo" / "data" / "clean"
    drift_cur = ROOT / "demo" / "data" / "poisoned"
    if drift_base.exists() and drift_cur.exists():
        with console.status("[blue]Comparing pixel + feature distributions..."):
            modules["drift_detector"] = DistributionShiftDetector().compare(
                str(drift_base), str(drift_cur))
        shift = modules.get("drift_detector", {}).get("shift_type", "?")
        _step(4, total, f"Drift done — {shift}", "done")
    else:
        _step(4, total, "Drift paths not found — skipped", "skip")

    # Step 5-8: Intelligence
    _step(5, total, "Building threat story (cross-module correlation)...")
    story = build_threat_story(modules)
    _step(5, total, f"Threat story built — {len(story.get('stories',[]))} narratives", "done")

    _step(6, total, "Computing adversarial immunity score...")
    immunity = compute_immunity_score(modules, story)
    score = float(immunity.get("score", 0))
    band = str(immunity.get("band", "?"))
    _step(6, total, f"Immunity: {score:.1f}/100 — {band}", "done")

    _step(7, total, "Generating commander's briefing...")
    briefing = generate_briefing(modules, story, immunity, use_llm=False)
    _step(7, total, f"Briefing generated ({briefing.get('source', 'template')})", "done")

    _step(8, total, "Signing and saving final report...")
    report = generate_report(modules, story, immunity, briefing,
                             target={"modules": list(modules)}, persist=True)
    _step(8, total, f"Report saved to {report.get('report_path', 'output/')}", "done")

    elapsed = time.time() - t0

    # Results
    score = float(immunity.get("score", 0))
    band = str(immunity.get("band", "?"))
    band_style = {"HARDENED": "bold green", "ADEQUATE": "green",
                  "MARGINAL": "bold yellow", "COMPROMISED": "bold red"
                  }.get(band, "white")

    result_table = Table(box=box.DOUBLE_EDGE, border_style="green",
                         title="[bold]FULL REPORT[/bold]")
    result_table.add_column("Metric", style="bold", width=22)
    result_table.add_column("Value")
    result_table.add_row("Immunity Score", Text(f"{score:.1f} / 100", style=band_style))
    result_table.add_row("Band",           Text(band, style=band_style))
    result_table.add_row("Modules Run",    str(len(modules)))
    result_table.add_row("Total Findings", str(len(report.get("findings") or [])))
    result_table.add_row("Schema Valid",   str(report.get("schema_valid")))
    result_table.add_row("Time",           f"{elapsed:.1f}s")
    console.print(result_table)

    # Briefing
    console.print()
    console.print(Panel(briefing.get("briefing", ""),
                        title="[bold cyan]Commander's Briefing[/bold cyan]",
                        border_style="cyan"))

    if report.get("report_path"):
        console.print(f"\n  [green]Report saved:[/green] {report['report_path']}")


def cmd_demo() -> None:
    """Run the guided demo."""
    console.print(Panel("[bold cyan]SHIELD DEMO[/bold cyan] — 8-Stage Guided Demo",
                        border_style="cyan"))
    console.print()
    quick = _get_input("Quick mode? (y/n)", "y").lower() == "y"
    console.print()
    os.system(f'"{sys.executable}" demo/run_demo.py {"--quick" if quick else ""}')


def cmd_dashboard() -> None:
    """Open the Streamlit dashboard."""
    console.print(Panel("[bold green]SHIELD DASHBOARD[/bold green] — Opening GUI...",
                        border_style="green"))
    console.print()
    console.print("  [cyan]Opening Streamlit dashboard at http://localhost:8501[/cyan]")
    console.print("  [dim]Press Ctrl+C to stop the server[/dim]")
    console.print()
    os.system(f'"{sys.executable}" -m streamlit run src/dashboard.py')


def cmd_help() -> None:
    """Show help."""
    console.print(Panel("[bold]SHIELD-CV COMMAND REFERENCE[/bold]",
                        border_style="cyan"))

    table = Table(box=box.SIMPLE, border_style="cyan")
    table.add_column("Command", style="bold cyan")
    table.add_column("What it does", style="white")
    table.add_column("Example", style="dim")

    table.add_row("shield scan", "Scans a dataset for 5 types of attacks",
                  "shield scan → demo/data/poisoned")
    table.add_row("shield audit", "Audits a model (white/grey/black box)",
                  "shield audit → demo/models/backdoored_model_scripted.pt")
    table.add_row("shield verify", "Verifies Ed25519 signed hash chain",
                  "shield verify → demo/records/clean_chain.json")
    table.add_row("shield drift", "Detects natural vs adversarial drift",
                  "shield drift → clean vs poisoned")
    table.add_row("shield report", "Runs all modules + generates signed report",
                  "shield report (auto-runs everything)")
    table.add_row("shield demo", "8-stage guided demo with narration",
                  "shield demo → quick mode")
    table.add_row("shield dashboard", "Opens browser GUI at localhost:8501",
                  "shield dashboard")
    console.print(table)

    console.print()
    console.print(Panel("[bold]Key Features[/bold]\n\n"
                        "  100% OFFLINE — no data leaves this host\n"
                        "  NEVER retrains models — inspect only\n"
                        "  CPU-ONLY — runs on Intel i5 / 8GB RAM\n"
                        "  EXPLAINABLE — every finding has reason + evidence\n"
                        "  CORRELATED — cross-module intelligence\n"
                        "  SIGNED — Ed25519 audit trail",
                        border_style="green"))


# ── Main Loop ────────────────────────────────────────────────────────────────

def main() -> int:
    show_banner()
    show_menu()

    commands = {
        "scan":      cmd_scan,
        "audit":     cmd_audit,
        "verify":    cmd_verify,
        "drift":     cmd_drift,
        "report":    cmd_report,
        "demo":      cmd_demo,
        "dashboard": cmd_dashboard,
        "help":      cmd_help,
    }

    while True:
        try:
            console.print()
            raw = input("  shield> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [dim]Goodbye.[/dim]")
            break

        if not raw:
            continue
        if raw in ("exit", "quit", "q"):
            console.print("  [dim]Goodbye.[/dim]")
            break

        # Parse command — supports "shield scan", "scan", "shield-scan"
        parts = raw.split()
        if parts[0] in ("shield", "shield-", "shield_") and len(parts) > 1:
            cmd = parts[1]
        else:
            cmd = parts[0].replace("shield-", "").replace("shield_", "")

        if cmd in commands:
            try:
                commands[cmd]()
            except KeyboardInterrupt:
                console.print("\n  [yellow]Interrupted.[/yellow]")
            except Exception as exc:
                console.print(f"\n  [red]Error: {type(exc).__name__}: {exc}[/red]")
                import traceback
                console.print(f"  [dim]{traceback.format_exc()}[/dim]")
        else:
            console.print(f"  [yellow]Unknown command: {raw}[/yellow]")
            console.print("  [dim]Type 'help' or 'exit'[/dim]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
