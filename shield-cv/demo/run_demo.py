"""SHIELD-CV guided end-to-end demonstration.

Runs the complete assurance workflow in one command and narrates what is
happening at each step, so a reviewer can see the framework work without
knowing any of the individual commands.

The demo is built around an A/B contrast, because a security tool that only
ever shows red teaches nothing: the same detectors are run against a clean
corpus and a poisoned one, and the difference between the two verdicts is the
actual result.

Stages:

1. Environment check — dependencies, config, demo assets
2. Clean dataset scan (the control)
3. Poisoned dataset scan (the attack)
4. Model audit — clean vs backdoored
5. Provenance chain — intact vs tampered
6. Distribution shift — natural weather vs injected content
7. Multi-agent office + cross-contributor meeting
8. Correlation, immunity score, commander's briefing, signed report

Run::

    python demo/run_demo.py              # full demo
    python demo/run_demo.py --quick      # smaller sample, skips slow stages
    python demo/run_demo.py --stage 3    # run one stage only
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

VERDICT_STYLE = {
    "COMPROMISED": "bold white on red",
    "SUSPICIOUS": "bold dark_orange",
    "CAUTION": "bold yellow",
    "MINOR_ANOMALIES": "yellow",
    "CLEAN": "bold white on green",
    "NOT_ASSESSED": "bold white on grey37",
    "UNKNOWN": "dim",
}


def _verdict(verdict: str) -> Text:
    """Render a verdict label in its severity colour.

    Args:
        verdict: Verdict string.

    Returns:
        Styled text.
    """
    return Text(f" {verdict} ", style=VERDICT_STYLE.get(str(verdict).upper(), "white"))


def _stage(number: int, title: str, explanation: str) -> None:
    """Print a stage header with an explanation of what it proves.

    Args:
        number: Stage number.
        title: Stage title.
        explanation: Why this stage matters.
    """
    console.print()
    console.rule(f"[bold cyan]STAGE {number} — {title}[/bold cyan]", style="cyan")
    console.print(Text(explanation, style="dim"))
    console.print()


def _result_line(label: str, verdict: str, extra: str = "") -> None:
    """Print one labelled verdict line.

    Args:
        label: What was assessed.
        verdict: Verdict produced.
        extra: Optional trailing detail.
    """
    console.print(Text.assemble(f"  {label:<34}", _verdict(verdict),
                                Text(f"  {extra}", style="dim")))



_HEAVY_KEYS = ("per_image", "features", "embeddings", "cluster_members",
               "cluster_size_stats", "cluster_members", "weight_statistics",
               "pixel_analysis", "feature_analysis", "candidates")


def _slim(payload: Any) -> Any:
    """Strip bulky intermediate arrays from a module result.

    Correlation, immunity scoring and reporting consume findings, summaries,
    verdicts and limitations. The raw per-image measurements and embedding
    matrices that produced them are not read again, but they dominate memory.
    Dropping them keeps peak usage at roughly one stage rather than all eight.

    Args:
        payload: A module result, or a mapping of results.

    Returns:
        The same structure with heavy intermediates removed.
    """
    try:
        if isinstance(payload, dict):
            if "findings" in payload or "summary" in payload:
                trimmed = {k: v for k, v in payload.items() if k not in _HEAVY_KEYS}
                detectors = trimmed.get("detectors")
                if isinstance(detectors, dict):
                    trimmed["detectors"] = {
                        name: ({k: v for k, v in entry.items() if k not in _HEAVY_KEYS}
                               if isinstance(entry, dict) else entry)
                        for name, entry in detectors.items()}
                return trimmed
            return {k: _slim(v) for k, v in payload.items()}
        return payload
    except Exception:  # pragma: no cover - defensive
        return payload



def _json_safe(value: Any) -> Any:
    """Convert a result tree into something ``json.dump`` accepts.

    Module results contain numpy scalars, enums and Path objects. The handoff
    file between isolated stages must survive a round trip, so unknown leaves
    are degraded to strings rather than raising.

    Args:
        value: Arbitrary nested result data.

    Returns:
        A JSON-serialisable equivalent.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    for attr in ("value", "item", "tolist"):
        if hasattr(value, attr):
            try:
                converted = getattr(value, attr)
                converted = converted() if callable(converted) else converted
                return _json_safe(converted)
            except Exception:
                break
    return str(value)


def _available_memory_mb() -> Optional[int]:
    """Read MemAvailable, in megabytes, when the platform exposes it.

    Returns:
        Available memory in MB, or None on Windows/macOS where the Linux
        proc interface is absent.
    """
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return None
    return None


def _run_isolated(number: int, argv_base: List[str], handoff: Path) -> bool:
    """Run one stage in a child process so the OS reclaims its memory.

    A white-box model audit peaks near 0.9 GB on its own. Running all eight
    stages in a single interpreter stacks those peaks and is killed by the
    out-of-memory reaper on small machines. Each child writes its slim result
    into the handoff directory, then exits, returning every byte to the OS.

    Args:
        number: Stage number to execute.
        argv_base: Shared command-line flags to forward to the child.
        handoff: Directory where stage results are exchanged.

    Returns:
        True when the child exited cleanly.
    """
    command = [sys.executable, str(Path(__file__).resolve()),
               "--stage", str(number), *argv_base]
    env = dict(os.environ)
    env["SHIELD_DEMO_HANDOFF"] = str(handoff)
    env["SHIELD_DEMO_CHILD"] = "1"
    completed = subprocess.run(command, env=env)
    if completed.returncode == -9 or completed.returncode == 137:
        console.print(f"[red]Stage {number} was killed by the out-of-memory "
                      f"reaper. Close other applications and retry.[/red]")
        return False
    return completed.returncode == 0


def stage_environment(_: argparse.Namespace) -> Dict[str, Any]:
    """Check dependencies, configuration and demo assets.

    Args:
        _: Parsed arguments (unused).

    Returns:
        Dictionary describing the environment state.
    """
    _stage(1, "ENVIRONMENT CHECK",
           "Confirms every dependency imports and the demo corpora exist. "
           "A missing package degrades detection silently, so this is checked "
           "up front rather than discovered mid-scan.")

    table = Table(header_style="bold", expand=False)
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")

    packages = [("numpy", "numerics"), ("scipy", "statistics"),
                ("torch", "model loading + embeddings"), ("cv2", "imaging"),
                ("PIL", "imaging"), ("imagehash", "near-duplicate pHash"),
                ("sklearn", "clustering"), ("cryptography", "Ed25519 signing"),
                ("onnxruntime", "ONNX inference"), ("yaml", "configuration"),
                ("jsonschema", "report validation"), ("rich", "CLI output"),
                ("streamlit", "dashboard")]
    missing: List[str] = []
    for module, purpose in packages:
        try:
            __import__(module)
            table.add_row(module, "[green]ok[/green]", purpose)
        except Exception as exc:
            missing.append(module)
            table.add_row(module, "[red]MISSING[/red]", f"{purpose} — {exc}")

    console.print(table)

    assets = {
        "clean dataset": ROOT / "demo" / "data" / "clean",
        "poisoned dataset": ROOT / "demo" / "data" / "poisoned",
        "drift corpora": ROOT / "demo" / "data" / "drift",
        "models": ROOT / "demo" / "models",
        "records": ROOT / "demo" / "records",
        "config": ROOT / "config" / "settings.yaml",
    }
    asset_table = Table(header_style="bold", expand=False)
    asset_table.add_column("Demo asset")
    asset_table.add_column("Status")
    asset_table.add_column("Path", overflow="fold")
    absent: List[str] = []
    for name, path in assets.items():
        if path.exists():
            asset_table.add_row(name, "[green]found[/green]", str(path))
        else:
            absent.append(name)
            asset_table.add_row(name, "[red]MISSING[/red]", str(path))
    console.print(asset_table)

    if missing:
        console.print(Panel(
            f"[red]Missing packages: {', '.join(missing)}[/red]\n"
            "Install them with:  pip install -r requirements.txt",
            border_style="red"))
    if absent:
        console.print(Panel(
            f"[yellow]Missing demo data: {', '.join(absent)}[/yellow]\n"
            "Regenerate with:  python demo/generate_demo_data.py",
            border_style="yellow"))
    if not missing and not absent:
        console.print("[bold green]Environment is complete — "
                      "all dependencies and demo assets present.[/bold green]")

    return {"missing_packages": missing, "missing_assets": absent}


def stage_clean_scan(args: argparse.Namespace) -> Dict[str, Any]:
    """Scan the clean corpus as an experimental control.

    Args:
        args: Parsed arguments.

    Returns:
        Scan result dictionary.
    """
    _stage(2, "CLEAN DATASET (the control)",
           "A detector that flags everything is useless. This establishes the "
           "false-positive baseline: the same five detectors are run against "
           "data with no injected attack.")

    from src.scanners.data_scanner import DataIntegrityEngine

    cap = 90 if args.quick else None
    with console.status("[cyan]Scanning clean corpus…"):
        result = DataIntegrityEngine().scan(ROOT / "demo" / "data" / "clean",
                                            max_images=cap)
    summary = result.get("summary") or {}
    _result_line("demo/data/clean", summary.get("verdict", "?"),
                 f"risk={summary.get('risk_score', 0):.3f}  "
                 f"findings={summary.get('total_findings', 0)}")
    counts = summary.get("by_severity") or {}
    console.print(f"  [dim]CRITICAL={counts.get('CRITICAL', 0)} "
                  f"HIGH={counts.get('HIGH', 0)} "
                  f"MEDIUM={counts.get('MEDIUM', 0)} "
                  f"LOW={counts.get('LOW', 0)}[/dim]")
    console.print("\n  [green]Expected: no CRITICAL or HIGH findings. Low-severity "
                  "observations are normal — real imagery always contains some "
                  "near-duplicates and unusual samples.[/green]")
    return result


def stage_poisoned_scan(args: argparse.Namespace) -> Dict[str, Any]:
    """Scan the poisoned corpus and score against ground truth.

    Args:
        args: Parsed arguments.

    Returns:
        Scan result dictionary.
    """
    _stage(3, "POISONED DATASET (the attack)",
           "The same corpus with three attacks injected at known locations: "
           "trigger patches, flipped labels and duplicate flooding. Because "
           "ground truth is known, precision and recall are measured rather "
           "than asserted.")

    from src.scanners.data_scanner import DataIntegrityEngine

    cap = 90 if args.quick else None
    with console.status("[cyan]Scanning poisoned corpus…"):
        result = DataIntegrityEngine().scan(ROOT / "demo" / "data" / "poisoned",
                                            max_images=cap)
    summary = result.get("summary") or {}
    _result_line("demo/data/poisoned", summary.get("verdict", "?"),
                 f"risk={summary.get('risk_score', 0):.3f}  "
                 f"findings={summary.get('total_findings', 0)}")

    risks = result.get("contributor_risk") or {}
    if risks:
        table = Table(title="Contributor risk", header_style="bold")
        table.add_column("Contributor")
        table.add_column("Risk", justify="right")
        table.add_column("Findings", justify="right")
        for name, entry in sorted(risks.items(),
                                  key=lambda kv: -float(kv[1].get("risk", 0.0))):
            risk = float(entry.get("risk", 0.0))
            style = "bold red" if risk >= 0.75 else "yellow" if risk >= 0.45 else "green"
            table.add_row(name, Text(f"{risk:.3f}", style=style),
                          str(entry.get("findings", 0)))
        console.print(table)

    if not args.quick:
        try:
            from demo.benchmark import benchmark
            with console.status("[cyan]Scoring against ground truth…"):
                scored = benchmark(str(ROOT / "demo" / "data" / "poisoned"))
            metrics = Table(title="Detection accuracy vs ground truth",
                            header_style="bold")
            for column in ("Attack", "Precision", "Recall", "F1"):
                metrics.add_column(column, justify="right" if column != "Attack" else "left")
            for attack, values in (scored.get("metrics") or {}).items():
                metrics.add_row(attack,
                                f"{values.get('precision', 0):.2f}",
                                f"{values.get('recall', 0):.2f}",
                                f"{values.get('f1', 0):.2f}")
            console.print(metrics)
        except Exception as exc:
            console.print(f"[yellow]Benchmark unavailable: {exc}[/yellow]")

    return result


def stage_model_audit(args: argparse.Namespace) -> Dict[str, Any]:
    """Audit the backdoored model against the clean reference.

    Args:
        args: Parsed arguments.

    Returns:
        Audit result dictionary.
    """
    _stage(4, "MODEL AUDIT — clean vs backdoored",
           "Two models are audited without retraining either. The backdoored "
           "model contains an implanted trigger; the clean one does not. The "
           "audit must separate them.")

    from src.analysis import embeddings as _embeddings
    from src.scanners.model_auditor import ModelIntegrityEngine

    # The dataset scan leaves a ResNet-18 backbone resident in the module-level
    # extractor singleton. The model audit then loads its own torch graphs on
    # top of that. Neither stage needs the other's memory, so the backbone is
    # released here; the next stage that needs it reloads lazily from the local
    # checkpoint. This keeps peak RSS at one stage's cost rather than the sum.
    _embeddings._EXTRACTOR_SINGLETON = None
    gc.collect()

    models = ROOT / "demo" / "models"
    engine = ModelIntegrityEngine()
    results: Dict[str, Any] = {}
    for label, filename in (("clean model", "clean_model_scripted.pt"),
                            ("backdoored model", "backdoored_model_scripted.pt")):
        path = models / filename
        if not path.exists():
            console.print(f"[yellow]  {label}: {path} not found, skipped[/yellow]")
            continue
        with console.status(f"[cyan]Auditing {label}…"):
            outcome = engine.audit(path, run_neural_cleanse=not args.quick)
        summary = outcome.get("summary") or {}
        _result_line(label, summary.get("verdict", "?"),
                     f"access={outcome.get('access_level')}  "
                     f"findings={len(outcome.get('findings') or [])}")
        results[filename] = _slim(outcome)
        del outcome
        gc.collect()

    console.print("\n  [dim]Neural Cleanse reverse-engineers a minimal trigger per "
                  "class and flags classes needing an abnormally small patch. "
                  "Use --quick to skip it.[/dim]")
    return results


def stage_provenance(_: argparse.Namespace) -> Dict[str, Any]:
    """Verify intact and tampered provenance chains.

    Args:
        _: Parsed arguments (unused).

    Returns:
        Mapping of fixture name to verification result.
    """
    _stage(5, "INFERENCE PROVENANCE — tamper evidence",
           "Five chains are verified: one intact, four attacked in different "
           "ways. The intact chain must pass and every attack must be caught, "
           "including one where the attacker recomputes the hash they broke.")

    from src.scanners.crypto_chain import InferenceProvenanceEngine

    engine = InferenceProvenanceEngine()
    fixtures = [
        ("clean_chain.json", "intact chain", "must PASS"),
        ("tampered_edit.json", "record edited", "must FAIL"),
        ("tampered_rehash.json", "edited + re-hashed", "must FAIL"),
        ("tampered_delete.json", "record deleted", "must FAIL"),
        ("tampered_replay.json", "record replayed", "must FAIL"),
    ]
    results: Dict[str, Any] = {}
    table = Table(header_style="bold", expand=False)
    table.add_column("Fixture")
    table.add_column("Scenario")
    table.add_column("Verdict")
    table.add_column("Expected", style="dim")

    for filename, scenario, expectation in fixtures:
        path = ROOT / "demo" / "records" / filename
        if not path.exists():
            table.add_row(filename, scenario, "[yellow]missing[/yellow]", expectation)
            continue
        outcome = engine.verify(path)
        verdict = (outcome.get("summary") or {}).get("verdict", "?")
        results[filename] = outcome
        table.add_row(filename, scenario, _verdict(verdict), expectation)

    console.print(table)
    console.print("\n  [dim]Re-hashing repairs the chain link but cannot forge the "
                  "Ed25519 signature, so tampering still surfaces.[/dim]")
    return results


def stage_drift(args: argparse.Namespace) -> Dict[str, Any]:
    """Compare natural and adversarial distribution shift.

    Args:
        args: Parsed arguments.

    Returns:
        Mapping of corpus name to drift result.
    """
    _stage(6, "DISTRIBUTION SHIFT — weather vs attack",
           "The hard part is not detecting change, it is telling a dusk "
           "shift from injected content. Natural drift must NOT be escalated "
           "as an attack, or operators learn to ignore the alarm.")

    from src.scanners.drift_detector import DistributionShiftDetector

    detector = DistributionShiftDetector()
    baseline = ROOT / "demo" / "data" / "clean"
    cap = 60 if args.quick else 120
    corpora = [("dusk", "natural: low light"), ("monsoon", "natural: rain"),
               ("sensor", "natural: sensor change"),
               ("injected", "ATTACK: injected content"),
               ("duplicated", "ATTACK: duplicate flood")]

    table = Table(header_style="bold", expand=False)
    table.add_column("Corpus")
    table.add_column("Nature", style="dim")
    table.add_column("Shift type")
    table.add_column("Risk", justify="right")
    results: Dict[str, Any] = {}

    for name, nature in corpora:
        path = ROOT / "demo" / "data" / "drift" / name
        if not path.exists():
            continue
        with console.status(f"[cyan]Comparing {name}…"):
            outcome = detector.compare(baseline, path, max_images=cap)
        shift = str(outcome.get("shift_type", "?"))
        style = ("bold red" if "SUSPICIOUS" in shift
                 else "yellow" if "MIXED" in shift else "green")
        table.add_row(name, nature, Text(shift, style=style),
                      f"{float(outcome.get('risk_score', 0.0)):.2f}")
        results[name] = outcome

    console.print(table)
    console.print("\n  [green]Expected: the three natural corpora classify as "
                  "NATURAL_OPERATIONAL_DRIFT; the two attacks as "
                  "SUSPICIOUS_MANIPULATION.[/green]")
    return results


def stage_office(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the multi-agent office and cross-contributor meeting.

    Args:
        args: Parsed arguments.

    Returns:
        Dictionary with office and meeting results.
    """
    _stage(7, "MULTI-AGENT OFFICE — per-contributor isolation",
           "Each contributor is scanned by its own agent in a separate "
           "process. Isolation matters: in a single pooled scan, a contributor "
           "who poisons a large share of the data drags the baseline toward "
           "itself and looks normal.")

    from src.agents.manager import OfficeManager
    from src.agents.meeting import MeetingRoom
    from src.utils.image_utils import list_images

    manager = OfficeManager()
    console.print(f"  [dim]workers={manager.max_workers} "
                  f"(capped by CPU cores and available RAM)[/dim]\n")
    with console.status("[cyan]Running one agent per contributor…"):
        office = manager.run(ROOT / "demo" / "data" / "poisoned",
                             max_images=60 if args.quick else None)

    table = Table(header_style="bold", expand=False)
    for column in ("Agent", "Contributor", "Trust", "Risk", "Findings", "Verdict"):
        table.add_column(column)
    for report in office.get("reports") or []:
        risk = float(report.get("risk_score", 0.0))
        style = "bold red" if risk >= 0.75 else "yellow" if risk >= 0.45 else "green"
        table.add_row(str(report.get("agent_id")), str(report.get("contributor")),
                      str(report.get("trust_level")), Text(f"{risk:.3f}", style=style),
                      str(report.get("num_findings")),
                      _verdict(str(report.get("verdict"))))
    console.print(table)

    probes = [str(p) for p in list_images(
        ROOT / "demo" / "data" / "clean", recursive=True)[:40]]
    models = [ROOT / "demo" / "models" / "clean_model_scripted.pt",
              ROOT / "demo" / "models" / "backdoored_model_scripted.pt"]
    models = [m for m in models if m.exists()]

    with console.status("[cyan]Holding the cross-contributor meeting…"):
        meeting = MeetingRoom().convene(office, models=models or None,
                                        reference_images=probes)

    console.print(Panel(str(meeting.get("narrative", "")),
                        title="Meeting room", border_style="cyan", expand=False))
    for finding in meeting.get("findings") or []:
        console.print(f"  [bold]{finding.get('finding_id')}[/bold] "
                      f"{finding.get('attack_class')} "
                      f"({float(finding.get('confidence', 0)):.2f}) — "
                      f"{finding.get('affected_asset')}")
    for recommendation in meeting.get("recommendations") or []:
        console.print(Text(f"  -> {recommendation}", style="bold"))

    return {"office": office, "meeting": meeting}


def stage_report(args: argparse.Namespace, collected: Dict[str, Any]) -> Dict[str, Any]:
    """Correlate every module result and emit the final signed report.

    Args:
        args: Parsed arguments.
        collected: Results gathered from earlier stages.

    Returns:
        The generated report dictionary.
    """
    _stage(8, "CORRELATION, IMMUNITY SCORE & BRIEFING",
           "Individual findings become an intelligence picture. Modules that "
           "did not run are recorded as NOT ASSESSED rather than counted as "
           "clean — absence of evidence is never evidence of integrity.")

    from src.intelligence.briefing import generate_briefing
    from src.intelligence.immunity import compute_immunity_score
    from src.intelligence.threat_story import build_threat_story
    from src.reporting.report_generator import generate_report

    modules: Dict[str, Any] = {}
    if collected.get("poisoned"):
        modules["data_scanner"] = collected["poisoned"]
    audits = collected.get("models") or {}
    if audits.get("backdoored_model_scripted.pt"):
        modules["model_auditor"] = audits["backdoored_model_scripted.pt"]
    chains = collected.get("chains") or {}
    if chains.get("tampered_replay.json"):
        modules["crypto_chain"] = chains["tampered_replay.json"]
    drifts = collected.get("drift") or {}
    if drifts.get("injected"):
        modules["drift_detector"] = drifts["injected"]

    if not modules:
        console.print("[yellow]No module results available to correlate.[/yellow]")
        return {}

    console.print(f"  Correlating: {', '.join(modules)}\n")
    story = build_threat_story(modules)
    immunity = compute_immunity_score(modules, story)
    briefing = generate_briefing(modules, story, immunity,
                                 use_llm=False if args.no_llm else None)
    report = generate_report(modules, story, immunity, briefing,
                             target={"demo": "SHIELD-CV guided demonstration"},
                             persist=True)

    score = float(immunity.get("score", 0.0))
    band = str(immunity.get("band", "?"))
    colour = {"HARDENED": "green", "ADEQUATE": "green",
              "MARGINAL": "yellow", "COMPROMISED": "red"}.get(band, "white")
    console.print(Panel(
        Text.assemble(("Adversarial Immunity Score\n", "bold"),
                      (f"{score:.1f} / 100  ", colour), (band, f"bold {colour}"),
                      (f"\n{immunity.get('band_description', '')}", "dim")),
        border_style=colour, expand=False))

    pillars = Table(title="Immunity pillars", header_style="bold")
    for column in ("Pillar", "Score", "Assessed"):
        pillars.add_column(column)
    for key, component in (immunity.get("components") or {}).items():
        value = float(component.get("score", 0.0))
        pillars.add_row(str(component.get("label", key)), f"{value:.1f}",
                        "[green]yes[/green]" if component.get("assessed")
                        else "[yellow]NO[/yellow]")
    console.print(pillars)

    for entry in story.get("stories") or []:
        console.print(f"  [bold dark_orange][{entry.get('pattern')}][/bold dark_orange] "
                      f"{entry.get('title')} "
                      f"({float(entry.get('confidence', 0)):.2f})")

    console.print(Panel(str(briefing.get("briefing", "")),
                        title=f"Commander's briefing ({briefing.get('source')})",
                        border_style="cyan"))

    console.print(f"  schema_valid = [bold]{report.get('schema_valid')}[/bold]")
    console.print(f"  report_hash  = {str(report.get('report_hash', ''))[:48]}…")
    if report.get("report_path"):
        console.print(f"  [green]Report written to {report['report_path']}[/green]")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    """Run the guided demonstration.

    Args:
        argv: Optional argument vector.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="SHIELD-CV guided end-to-end demonstration.")
    parser.add_argument("--quick", action="store_true",
                        help="smaller sample and skip the slow trigger search")
    parser.add_argument("--stage", type=int, default=None, choices=range(1, 9),
                        help="run a single stage (1-8)")
    parser.add_argument("--no-llm", action="store_true",
                        help="force the deterministic template briefing")
    parser.add_argument("--isolate", action="store_true",
                        help="run every stage in its own process (lower peak "
                             "memory; enabled automatically under 3 GB free)")
    args = parser.parse_args(argv)

    console.print(Panel(
        Text.assemble(
            ("SHIELD-CV\n", "bold cyan"),
            ("Secure Holistic Integrity Evaluation Layer for Defence "
             "Computer Vision\n", "white"),
            ("Guided demonstration — 100% offline, no data leaves this host",
             "dim")),
        border_style="cyan"))

    started = time.time()
    collected: Dict[str, Any] = {}
    failures: List[str] = []

    handoff = Path(os.environ.get("SHIELD_DEMO_HANDOFF")
                   or (ROOT / "output" / "demo_handoff"))
    is_child = os.environ.get("SHIELD_DEMO_CHILD") == "1"
    # A single stage peaks near 0.9 GB. When the machine cannot comfortably
    # hold the whole run in one interpreter, each stage is executed in its own
    # child process and results are exchanged through small JSON files.
    available = _available_memory_mb()
    isolate = (args.stage is None and not is_child
               and (args.isolate or (available is not None and available < 3000)))
    if isolate:
        handoff.mkdir(parents=True, exist_ok=True)
        for stale in handoff.glob("stage_*.json"):
            stale.unlink()
        console.print(f"[dim]Low memory ({available} MB available): running "
                      f"each stage in its own process.[/dim]\n")
        argv_base: List[str] = []
        if args.quick:
            argv_base.append("--quick")
        if args.no_llm:
            argv_base.append("--no-llm")
        for number in range(1, 9):
            if not _run_isolated(number, argv_base, handoff):
                failures.append(f"stage {number}")
        elapsed = time.time() - started
        console.print()
        console.rule("[bold cyan]DEMONSTRATION COMPLETE[/bold cyan]", style="cyan")
        if failures:
            console.print(f"[red]{len(failures)} stage(s) failed:[/red]")
            for item in failures:
                console.print(f"  [red]! {item}[/red]")
        else:
            console.print("[bold green]All stages completed successfully."
                          "[/bold green]")
        console.print(f"[dim]Elapsed {elapsed:.1f}s. "
                      f"Next: python run.py --help  ·  "
                      f"streamlit run src/dashboard.py[/dim]")
        return 1 if failures else 0

    if is_child:
        for saved in sorted(handoff.glob("stage_*.json")):
            try:
                with open(saved, "r", encoding="utf-8") as handle:
                    collected.update(json.load(handle))
            except Exception:
                continue

    stages: List[tuple] = [
        (1, "environment", lambda: stage_environment(args), None),
        (2, "clean scan", lambda: stage_clean_scan(args), "clean"),
        (3, "poisoned scan", lambda: stage_poisoned_scan(args), "poisoned"),
        (4, "model audit", lambda: stage_model_audit(args), "models"),
        (5, "provenance", lambda: stage_provenance(args), "chains"),
        (6, "drift", lambda: stage_drift(args), "drift"),
        (7, "office", lambda: stage_office(args), "office"),
    ]

    for number, name, runner, key in stages:
        if args.stage is not None and args.stage != number:
            continue
        try:
            outcome = runner()
            if key:
                # Full scan results carry embedding matrices that can run to
                # hundreds of megabytes. Stage 8 only needs findings, summaries
                # and limitations, so the heavy arrays are dropped as soon as
                # each stage finishes. Without this the demo peaks at the sum
                # of every stage instead of the largest one, and is OOM-killed
                # on modest hardware.
                collected[key] = _slim(outcome)
                if is_child:
                    try:
                        handoff.mkdir(parents=True, exist_ok=True)
                        target = handoff / f"stage_{number}.json"
                        with open(target, "w", encoding="utf-8") as handle:
                            json.dump(_json_safe({key: collected[key]}), handle)
                    except Exception as exc:
                        console.print(f"[yellow]Could not persist stage "
                                      f"{number} result: {exc}[/yellow]")
            del outcome
            gc.collect()
        except Exception as exc:
            failures.append(f"stage {number} ({name}): {type(exc).__name__}: {exc}")
            console.print(f"[red]Stage {number} failed: {exc}[/red]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")

    if args.stage is None or args.stage == 8:
        try:
            stage_report(args, collected)
        except Exception as exc:
            failures.append(f"stage 8 (report): {type(exc).__name__}: {exc}")
            console.print(f"[red]Stage 8 failed: {exc}[/red]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")

    elapsed = time.time() - started
    if is_child:
        # The parent prints the single closing banner for the whole run.
        return 1 if failures else 0
    console.print()
    console.rule("[bold cyan]DEMONSTRATION COMPLETE[/bold cyan]", style="cyan")
    if failures:
        console.print(f"[red]{len(failures)} stage(s) failed:[/red]")
        for item in failures:
            console.print(f"  [red]! {item}[/red]")
    else:
        console.print("[bold green]All stages completed successfully.[/bold green]")
    console.print(f"[dim]Elapsed {elapsed:.1f}s. "
                  f"Next: python run.py --help  ·  streamlit run src/dashboard.py[/dim]")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
