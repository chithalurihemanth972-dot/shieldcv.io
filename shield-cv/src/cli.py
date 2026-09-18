"""SHIELD-CV command-line interface.

Six commands cover the full assurance workflow:

* ``scan``   — data integrity scan of a contributed dataset
* ``audit``  — model integrity audit of a contributed model
* ``verify`` — inference provenance chain verification
* ``drift``  — operational distribution shift between two corpora
* ``report`` — run every applicable module and emit a full assurance report
* ``office`` — multi-agent per-contributor scan plus the cross-contributor meeting

Presentation rules are deliberate and consistent across commands. Severity
drives colour (red CRITICAL, orange HIGH, yellow MEDIUM, cyan LOW, green
clean). Every command prints the coverage and limitations of what it just did,
because a result without its blind spots invites over-trust. Exit codes are
meaningful so the tool can gate a pipeline: ``0`` clean, ``1`` findings
requiring review, ``2`` compromised, ``3`` execution error.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from src.config import get_config
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)
console = Console()

# Exit codes, so `shield scan ... && deploy` behaves sensibly in a pipeline.
EXIT_CLEAN = 0
EXIT_FINDINGS = 1
EXIT_COMPROMISED = 2
EXIT_ERROR = 3

SEVERITY_STYLE = {
    "CRITICAL": "bold red",
    "HIGH": "bold dark_orange",
    "MEDIUM": "yellow",
    "LOW": "cyan",
    "INFO": "dim",
}

VERDICT_STYLE = {
    "COMPROMISED": "bold white on red",
    "SUSPICIOUS": "bold dark_orange",
    "CAUTION": "bold yellow",
    "MINOR_ANOMALIES": "yellow",
    "CLEAN": "bold green",
    "NOT_ASSESSED": "dim",
    "UNKNOWN": "dim",
    "ERROR": "bold red",
}

BAND_STYLE = {
    "HARDENED": "bold green",
    "ADEQUATE": "green",
    "MARGINAL": "yellow",
    "COMPROMISED": "bold red",
}


# ---------------------------------------------------------------------------
# presentation helpers
# ---------------------------------------------------------------------------
BANNER_ART = """
[bold cyan]
   _____ __  __ ____    _____           _   _ 
  / ____|  \\/  |  _ \\  |_   _|__   ___ | |_| |
  \\___ \\|      | |_) |   | |/ _ \\ / _ \\| __| |
   ___) | |\\/| |  _ <    | | (_) | (_) | |_| |
  |____/|_|  |_|_| \\_\\   |_|\\___/ \\___/ \\__|_|
[/bold cyan]
[bold white]   Secure Holistic Integrity Evaluation Layer[/bold white]
[bold white]   for Defence Computer Vision[/bold white]
"""

def _banner(subtitle: str) -> None:
    """Print the tool banner.

    Args:
        subtitle: Short description of the command being run.
    """
    try:
        cfg = get_config()
        version = cfg.get("project.version", "1.0.0")
        classification = cfg.get("project.classification", "RESTRICTED")
    except Exception:
        version, classification = "1.0.0", "RESTRICTED"

    console.print(BANNER_ART)
    console.print(Panel(
        Text.assemble(
            (f"  v{version}  ", "bold cyan"),
            (f" |  {subtitle}\n", "white"),
            (f"  {classification} | 100% OFFLINE | AIR-GAPPED READY", "dim bold cyan"),
        ),
        border_style="bright_cyan", expand=False))


def _step(label: str, detail: str = "") -> None:
    """Print a visual step indicator."""
    console.print(f"\n[bold bright_cyan]>> {label}[/bold bright_cyan]"
                  + (f" [dim]{detail}[/dim]" if detail else ""))


def _check(label: str, status: str, color: str = "green") -> None:
    """Print a check result with status icon."""
    console.print(f"  [{color}][{status}][/{color}]  {label}")


def _verdict_text(verdict: str) -> Text:
    """Render a verdict string in its severity colour.

    Args:
        verdict: Verdict label.

    Returns:
        Styled :class:`rich.text.Text`.
    """
    return Text(f" {verdict} ", style=VERDICT_STYLE.get(str(verdict).upper(), "white"))


def _findings_table(findings: Sequence[Dict[str, Any]], limit: int = 20,
                    title: str = "Findings") -> Optional[Table]:
    """Build a table of the most serious findings.

    Args:
        findings: Finding dictionaries.
        limit: Maximum rows to show.
        title: Table title.

    Returns:
        A populated :class:`rich.table.Table`, or ``None`` when there are no
        findings.
    """
    if not findings:
        return None

    table = Table(title=f"{title} (showing {min(limit, len(findings))} of {len(findings)})",
                  header_style="bold", expand=False)
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Severity", no_wrap=True)
    table.add_column("Conf", justify="right", no_wrap=True)
    table.add_column("Attack class", no_wrap=True)
    table.add_column("Asset", overflow="fold")
    table.add_column("Disposition", no_wrap=True)

    for finding in list(findings)[:limit]:
        severity = str(finding.get("severity", "INFO")).upper()
        disposition = str(finding.get("recommended_disposition")
                          or finding.get("disposition", ""))
        table.add_row(
            str(finding.get("finding_id", "")),
            Text(severity, style=SEVERITY_STYLE.get(severity, "white")),
            f"{float(finding.get('confidence', 0.0)):.2f}",
            str(finding.get("attack_class", "")),
            str(finding.get("affected_asset", ""))[:48],
            Text(disposition,
                 style="bold red" if disposition == "QUARANTINE"
                 else ("yellow" if disposition == "REVIEW" else "green")),
        )
    return table


def _print_summary(summary: Dict[str, Any]) -> None:
    """Print a severity breakdown and the overall verdict.

    Args:
        summary: Summary dictionary from ``summarize_findings``.
    """
    if not summary:
        return
    counts = summary.get("by_severity") or {}
    parts: List[Text] = []
    for severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"):
        count = int(counts.get(severity, 0) or 0)
        if count:
            parts.append(Text(f"{count} {severity}",
                              style=SEVERITY_STYLE.get(severity, "white")))
    console.print()
    verdict = summary.get("verdict", "UNKNOWN")
    console.print(Text.assemble(
        "  VERDICT: ",
        _verdict_text(verdict),
        f"   risk={float(summary.get('risk_score', 0.0)):.3f}"
        f"   findings={int(summary.get('total_findings', 0))}"))
    if parts:
        line = Text("  ")
        for index, part in enumerate(parts):
            if index:
                line.append("  ·  ", style="dim")
            line.append_text(part)
        console.print(line)


def _print_limitations(limitations: Sequence[str], coverage: Optional[Dict[str, Any]] = None,
                       limit: int = 12) -> None:
    """Print the coverage statement and limitations.

    Args:
        limitations: Limitation strings.
        coverage: Optional coverage statement.
        limit: Maximum limitations to print.
    """
    if isinstance(coverage, str):
        if coverage.strip():
            console.print(Text("\n  COVERAGE:", style="bold green"))
            console.print(Text(f"    {coverage.strip()}", style="green"))
        coverage = None

    if coverage:
        supported = coverage.get("supported") or []
        unsupported = coverage.get("unsupported") or []
        if supported:
            console.print(Text("\n  DETECTED:", style="bold green"))
            for item in list(supported)[:10]:
                console.print(Text(f"    + {item}", style="green"))
        if unsupported:
            console.print(Text("\n  BLIND SPOTS:", style="bold yellow"))
            for item in list(unsupported)[:10]:
                console.print(Text(f"    - {item}", style="yellow"))

    if limitations:
        console.print(Text(f"\n  LIMITATIONS ({len(limitations)}):", style="bold yellow"))
        for item in list(limitations)[:limit]:
            console.print(Text(f"    ! {item}", style="yellow"))
        if len(limitations) > limit:
            console.print(Text(f"    ... {len(limitations) - limit} more", style="dim"))


def _assessment_failed(result: Dict[str, Any]) -> bool:
    """Detect a run that produced no assessment at all.

    A module that could not read its target emits zero findings, which renders
    as "CLEAN". Reporting that as success would let a mistyped path or an
    unreadable dataset pass a CI gate as though it had been checked. Absence of
    evidence is not evidence of integrity, so these runs are surfaced as
    errors instead.

    Args:
        result: Full module result dictionary.

    Returns:
        ``True`` when nothing was actually assessed.
    """
    try:
        dataset = result.get("dataset") or {}
        if dataset and int(dataset.get("num_samples", 0) or 0) == 0:
            return True
        if "num_records" in result and int(result.get("num_records", 0) or 0) == 0:
            return True
        if str(result.get("access_level", "")).upper() == "UNAVAILABLE":
            return True
        detectors = result.get("detectors") or {}
        if detectors and not any(d.get("available") for d in detectors.values()
                                 if isinstance(d, dict)):
            return True
        if result.get("assessment_failed"):
            return True
        if str((result.get("summary") or {}).get("verdict", "")).upper() == "NOT_ASSESSED":
            return True
        return False
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("_assessment_failed check failed: %s", exc)
        return False


def _exit_code(summary: Dict[str, Any],
               result: Optional[Dict[str, Any]] = None) -> int:
    """Map a verdict to a process exit code.

    Args:
        summary: Summary dictionary.
        result: Optional full result, checked for a failed assessment.

    Returns:
        One of the module-level ``EXIT_*`` codes.
    """
    if result is not None and _assessment_failed(result):
        console.print("\n[bold red]NOT ASSESSED — the target could not be read. "
                      "This is not a clean result.[/bold red]")
        return EXIT_ERROR

    verdict = str(summary.get("verdict", "UNKNOWN")).upper()
    if verdict == "COMPROMISED":
        return EXIT_COMPROMISED
    if verdict in ("CLEAN", "NOT_ASSESSED", "UNKNOWN"):
        return EXIT_CLEAN
    return EXIT_FINDINGS


def _emit_json(payload: Dict[str, Any], destination: Optional[str]) -> None:
    """Write a result payload as JSON to a file or stdout.

    Args:
        payload: Result dictionary.
        destination: Output path, or ``None`` for stdout.
    """
    try:
        text = json.dumps(payload, indent=2, default=str)
        if destination:
            target = Path(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            console.print(f"[green]JSON written to[/green] {target}")
        else:
            console.print_json(text)
    except Exception as exc:
        LOGGER.error("JSON emit failed: %s", exc)
        console.print(f"[red]Could not write JSON: {exc}[/red]")


def _progress() -> Progress:
    """Create a consistently-styled progress display.

    Returns:
        A configured :class:`rich.progress.Progress`.
    """
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="cyan", finished_style="green"),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console, transient=True)


def _run_with_progress(label: str, work: Callable[[Callable[[str, int, int], None]], Any]
                       ) -> Any:
    """Run a callable that reports ``(stage, done, total)`` progress.

    Args:
        label: Initial progress description.
        work: Callable accepting a progress callback.

    Returns:
        Whatever ``work`` returns.
    """
    with _progress() as progress:
        task = progress.add_task(label, total=100)

        def callback(stage: str, done: int, total: int) -> None:
            """Forward detector progress into the rich progress bar."""
            try:
                progress.update(task, description=f"{label} · {stage}",
                                completed=int(done), total=max(int(total), 1))
            except Exception:  # pragma: no cover - display only
                pass

        return work(callback)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def _detect_target(target: str) -> str:
    """Auto-detect target type: 'dataset', 'model', or 'records'."""
    path = Path(target)
    if not path.exists():
        return "unknown"
    if path.is_dir():
        return "dataset"
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth", ".onnx"}:
        return "model"
    if suffix in {".json", ".jsonl"}:
        return "records"
    return "unknown"


def cmd_scan(args: argparse.Namespace) -> int:
    """Smart scan: auto-detects dataset, model, or records and runs the right check.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    target_type = _detect_target(args.target)

    if target_type == "dataset":
        from src.scanners.data_scanner import DataIntegrityEngine
        _banner(f"DATA INTEGRITY SCAN · {args.target}")
        _step("Detecting dataset format")
        engine = DataIntegrityEngine()
        result = _run_with_progress(
            "Scanning dataset",
            lambda cb: engine.scan(args.target, max_images=args.max_images, progress_callback=cb))

        _step("RESULTS")
        _print_dataset_summary(result)
        _print_findings(result.get("findings") or [], args.limit)
        _print_summary(result.get("summary") or {})
        _print_limitations(result.get("limitations") or [])
        if args.json is not None:
            _emit_json(result, args.json)
        return _exit_code(result.get("summary") or {}, result)

    if target_type == "model":
        from src.scanners.model_auditor import ModelIntegrityEngine
        _banner(f"MODEL INTEGRITY AUDIT · {args.target}")
        _step("Loading model weights")
        engine = ModelIntegrityEngine()
        result = _run_with_progress(
            "Auditing model",
            lambda cb: engine.audit(args.target, reference_path=args.reference,
                                    reference_images=args.images,
                                    run_neural_cleanse=not getattr(args, 'skip_neural_cleanse', False),
                                    progress_callback=cb))

        _step("RESULTS")
        _print_model_summary(result)
        _print_findings(result.get("findings") or [], args.limit)
        _print_summary(result.get("summary") or {})
        _print_limitations(result.get("limitations") or [], result.get("coverage_statement"))
        if args.json is not None:
            _emit_json(result, args.json)
        return _exit_code(result.get("summary") or {}, result)

    if target_type == "records":
        from src.scanners.crypto_chain import InferenceProvenanceEngine
        _banner(f"INFERENCE PROVENANCE VERIFICATION · {args.target}")
        _step("Loading inference records")
        engine = InferenceProvenanceEngine()
        result = engine.verify(args.target)

        _step("RESULTS")
        _print_chain_summary(result)
        _print_findings(result.get("findings") or [], args.limit)
        _print_summary(result.get("summary") or {})
        _print_limitations(result.get("limitations") or [])
        if args.json is not None:
            _emit_json(result, args.json)
        return _exit_code(result.get("summary") or {}, result)

    console.print(f"\n[bold red]>> Could not determine target type for: {args.target}[/bold red]")
    console.print("[dim]Expected: dataset directory, model file (.pt/.pth/.onnx), or records JSON[/dim]")
    return EXIT_ERROR


def _print_dataset_summary(result: Dict[str, Any]) -> None:
    dataset = result.get("dataset") or {}
    console.print(f"\n[bold]Dataset:[/bold] {dataset.get('num_samples', 0)} samples · "
                  f"{dataset.get('num_classes', 0)} classes · "
                  f"format {dataset.get('format', 'unknown')}")
    risks = result.get("contributor_risk") or {}
    if risks:
        risk_table = Table(title="Contributor risk", header_style="bold")
        risk_table.add_column("Contributor")
        risk_table.add_column("Risk", justify="right")
        risk_table.add_column("Findings", justify="right")
        risk_table.add_column("Assessment", overflow="fold")
        for name, entry in sorted(risks.items(), key=lambda kv: -float(kv[1].get("risk", 0.0))):
            risk = float(entry.get("risk", 0.0))
            style = ("bold red" if risk >= 0.75 else "yellow" if risk >= 0.45 else "green")
            risk_table.add_row(name, Text(f"{risk:.3f}", style=style),
                               str(entry.get("findings", 0)),
                               str(entry.get("assessment", ""))[:70])
        console.print()
        console.print(risk_table)


def _print_model_summary(result: Dict[str, Any]) -> None:
    access = str(result.get("access_level", "UNKNOWN"))
    console.print(f"\n  ACCESS LEVEL: "
                  f"[{'green' if 'WHITE' in access else 'yellow'}]{access}[/] | "
                  f"confidence {float(result.get('confidence', 0.0)):.2f}")
    cleanse = result.get("neural_cleanse") or {}
    if cleanse.get("available"):
        flagged = cleanse.get("flagged_classes") or []
        console.print(f"  NEURAL CLEANSE: scanned "
                      f"{cleanse.get('classes_scanned', 0)} class(es); "
                      + (f"[red]FLAGGED {flagged}[/red]" if flagged
                         else "[green]no class flagged[/green]"))


def _print_chain_summary(result: Dict[str, Any]) -> None:
    valid = bool(result.get("chain_valid", False))
    if valid:
        console.print(f"\n  [bold white on green] VALID [/bold white on green]   records={result.get('num_records', 0)}")
    else:
        console.print(f"\n  [bold white on red] BROKEN [/bold white on red]   records={result.get('num_records', 0)}")
    if result.get("first_break_index") is not None:
        console.print(f"  [red]First break at record index {result['first_break_index']}[/red]")


def _print_findings(findings: Sequence[Dict[str, Any]], limit: int) -> None:
    table = _findings_table(findings, limit=limit)
    if table:
        console.print()
        console.print(table)
    else:
        console.print("\n[bold green]No findings — target appears clean.[/bold green]")


def cmd_audit(args: argparse.Namespace) -> int:
    """Audit a contributed model for backdoors and weight anomalies.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    from src.scanners.model_auditor import ModelIntegrityEngine

    _banner(f"MODEL INTEGRITY AUDIT · {args.model}")
    _step("Loading model", args.model)
    engine = ModelIntegrityEngine()
    result = _run_with_progress(
        "Auditing model",
        lambda cb: engine.audit(args.model, reference_path=args.reference,
                                reference_images=args.images,
                                run_neural_cleanse=not args.skip_neural_cleanse,
                                progress_callback=cb))

    _step("AUDIT RESULTS")
    _print_model_summary(result)
    _print_findings(result.get("findings") or [], args.limit)
    _print_summary(result.get("summary") or {})
    _print_limitations(result.get("limitations") or [], result.get("coverage_statement"))

    if args.json is not None:
        _emit_json(result, args.json)
    return _exit_code(result.get("summary") or {}, result)


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify an inference provenance chain for tampering.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    from src.scanners.crypto_chain import InferenceProvenanceEngine

    _banner(f"Inference provenance verification · {args.records}")
    engine = InferenceProvenanceEngine()
    result = engine.verify(args.records)

    valid = bool(result.get("chain_valid", False))
    console.print(Text.assemble(
        "\nChain status: ",
        Text(" VALID " if valid else " BROKEN ",
             style="bold white on green" if valid else "bold white on red"),
        f"   records={result.get('num_records', 0)}"))

    if result.get("first_break_index") is not None:
        console.print(f"[red]First break at record index "
                      f"{result['first_break_index']}[/red]")

    table = _findings_table(result.get("findings") or [], limit=args.limit)
    if table:
        console.print()
        console.print(table)
    else:
        console.print("[bold green]No provenance findings — chain intact.[/bold green]")

    _print_summary(result.get("summary") or {})
    _print_limitations(result.get("limitations") or [])

    if args.json is not None:
        _emit_json(result, args.json)
    return _exit_code(result.get("summary") or {}, result)


def cmd_drift(args: argparse.Namespace) -> int:
    """Compare two corpora for operational or adversarial distribution shift.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    from src.scanners.drift_detector import DistributionShiftDetector

    _banner(f"Distribution shift · {args.baseline} → {args.current}")
    detector = DistributionShiftDetector()
    result = _run_with_progress(
        "Comparing distributions",
        lambda cb: detector.compare(args.baseline, args.current,
                                    max_images=args.max_images,
                                    progress_callback=cb))

    shift_type = str(result.get("shift_type", "UNKNOWN"))
    style = ("bold red" if "SUSPICIOUS" in shift_type else
             "yellow" if "MIXED" in shift_type else "green")
    console.print(Text.assemble(
        "\nShift type: ", Text(shift_type, style=style),
        f"   risk={float(result.get('risk_score', 0.0)):.3f}"
        f"   severity={result.get('severity', 'NONE')}"
        f"   confidence={float(result.get('confidence', 0.0)):.2f}"))
    console.print(f"\n[bold]Characterization:[/bold] {result.get('characterization', '')}")
    console.print(f"[bold]Recommendation:[/bold] {result.get('recommendation', '')}")

    table = _findings_table(result.get("findings") or [], limit=args.limit)
    if table:
        console.print()
        console.print(table)

    _print_summary(result.get("summary") or {})
    _print_limitations(result.get("limitations") or [])

    if args.json is not None:
        _emit_json(result, args.json)
    return _exit_code(result.get("summary") or {}, result)


def cmd_office(args: argparse.Namespace) -> int:
    """Run the multi-agent office scan and the cross-contributor meeting.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    from src.agents.manager import OfficeManager
    from src.agents.meeting import MeetingRoom

    _banner(f"Multi-agent office · {args.path}")
    manager = OfficeManager(max_workers=args.workers)
    tasks = manager.discover(args.path)
    console.print(f"\n[bold]Discovered {len(tasks)} contributor(s):[/bold] "
                  + ", ".join(t["contributor"] for t in tasks))

    office = _run_with_progress(
        "Running agents",
        lambda cb: manager.run(args.path, contributors=args.contributors,
                               max_images=args.max_images,
                               parallel=not args.serial, progress_callback=cb))

    agent_table = Table(title="Agent reports", header_style="bold")
    agent_table.add_column("Agent", no_wrap=True)
    agent_table.add_column("Contributor")
    agent_table.add_column("Trust", no_wrap=True)
    agent_table.add_column("Risk", justify="right")
    agent_table.add_column("Samples", justify="right")
    agent_table.add_column("Findings", justify="right")
    agent_table.add_column("Verdict", no_wrap=True)
    for report in office.get("reports") or []:
        risk = float(report.get("risk_score", 0.0))
        style = ("bold red" if risk >= 0.75 else
                 "yellow" if risk >= 0.45 else "green")
        agent_table.add_row(
            str(report.get("agent_id", "")), str(report.get("contributor", "")),
            str(report.get("trust_level", "")), Text(f"{risk:.3f}", style=style),
            str(report.get("num_samples", 0)), str(report.get("num_findings", 0)),
            _verdict_text(str(report.get("verdict", "UNKNOWN"))))
    console.print()
    console.print(agent_table)
    console.print(f"[dim]{office.get('agents', 0)} agent(s), "
                  f"{office.get('workers', 1)} worker(s), "
                  f"{'parallel' if office.get('parallel') else 'sequential'}, "
                  f"{office.get('duration_seconds', 0.0)}s[/dim]")

    meeting = MeetingRoom().convene(office, models=args.models,
                                    reference_images=_sample_images(args.path, 40))

    console.print(Panel(str(meeting.get("narrative", "")),
                        title="Meeting room", border_style="cyan", expand=False))

    table = _findings_table(meeting.get("findings") or [], limit=args.limit,
                            title="Cross-contributor findings")
    if table:
        console.print(table)
    else:
        console.print("[green]No cross-contributor anomaly found.[/green]")

    for recommendation in meeting.get("recommendations") or []:
        console.print(Text(f"  → {recommendation}", style="bold"))

    console.print(Text.assemble("\nOffice verdict: ",
                                _verdict_text(meeting.get("verdict", "UNKNOWN"))))
    _print_limitations(meeting.get("limitations") or [])

    payload = {"office": office, "meeting": meeting}
    if args.json is not None:
        _emit_json(payload, args.json)

    verdict = str(meeting.get("verdict", "UNKNOWN")).upper()
    if verdict == "COMPROMISED":
        return EXIT_COMPROMISED
    return EXIT_FINDINGS if meeting.get("findings") else EXIT_CLEAN


def _sample_images(root: str | Path, count: int) -> List[str]:
    """Collect a few real images from a dataset for model probing.

    Args:
        root: Dataset root.
        count: Maximum number of image paths to return.

    Returns:
        List of image paths, empty when none can be found.
    """
    try:
        from src.utils.image_utils import list_images
        return [str(p) for p in list_images(root, recursive=True)[:count]]
    except Exception as exc:
        LOGGER.error("image sampling failed for %s: %s", root, exc)
        return []


def cmd_report(args: argparse.Namespace) -> int:
    """Run every applicable module and emit a full assurance report.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    from src.intelligence.briefing import generate_briefing
    from src.intelligence.immunity import compute_immunity_score
    from src.intelligence.threat_story import build_threat_story
    from src.reporting.report_generator import generate_report

    # Auto-discover demo data when "all" is passed
    if getattr(args, "target", None) == "all":
        root = Path(__file__).resolve().parent.parent
        demo = root / "demo"
        args.data = str(demo / "data" / "clean")
        args.model = str(demo / "models" / "clean_model.pt")
        args.reference = str(demo / "models" / "backdoored_model.pt")
        args.records = str(demo / "records" / "clean_chain.json")
        args.baseline = str(demo / "data" / "clean")
        args.current = str(demo / "data" / "poisoned")
        if args.max_images is None:
            args.max_images = 10

    _banner("FULL ASSURANCE REPORT - ALL MODULES")
    modules: Dict[str, Dict[str, Any]] = {}
    target: Dict[str, Any] = {}

    if args.data:
        _step("PHASE 1/4", "DATA INTEGRITY SCAN")
        from src.scanners.data_scanner import DataIntegrityEngine
        engine = DataIntegrityEngine()
        modules["data_scanner"] = _run_with_progress(
            "Data integrity",
            lambda cb: engine.scan(args.data, max_images=args.max_images,
                                   progress_callback=cb))
        target["dataset"] = str(args.data)

    if args.model:
        _step("PHASE 2/4", "MODEL INTEGRITY AUDIT")
        from src.scanners.model_auditor import ModelIntegrityEngine
        auditor = ModelIntegrityEngine()
        modules["model_auditor"] = _run_with_progress(
            "Model integrity",
            lambda cb: auditor.audit(args.model, reference_path=args.reference,
                                     run_neural_cleanse=not args.skip_neural_cleanse,
                                     progress_callback=cb))
        target["model"] = str(args.model)

    if args.records:
        _step("PHASE 3/4", "INFERENCE PROVENANCE VERIFICATION")
        from src.scanners.crypto_chain import InferenceProvenanceEngine
        modules["crypto_chain"] = InferenceProvenanceEngine().verify(args.records)
        target["records"] = str(args.records)

    if args.baseline and args.current:
        _step("PHASE 4/4", "DISTRIBUTION SHIFT DETECTION")
        from src.scanners.drift_detector import DistributionShiftDetector
        detector = DistributionShiftDetector()
        modules["drift_detector"] = _run_with_progress(
            "Distribution shift",
            lambda cb: detector.compare(args.baseline, args.current,
                                        max_images=args.max_images,
                                        progress_callback=cb))
        target["drift"] = {"baseline": str(args.baseline), "current": str(args.current)}

    if not modules:
        console.print("\n[bold red]>> Nothing to report on.[/bold red]")
        console.print("[dim]Supply at least one of: --data, --model, --records, --baseline/--current[/dim]")
        return EXIT_ERROR

    _step("CORRELATING FINDINGS")
    story = build_threat_story(modules)
    immunity = compute_immunity_score(modules, story)
    briefing = generate_briefing(modules, story, immunity,
                                 use_llm=None if not args.no_llm else False)

    report = generate_report(modules, story, immunity, briefing,
                             target=target or str(args.data or ""),
                             persist=not args.no_save)

    score = float(immunity.get("score", 0.0))
    band = str(immunity.get("band", "UNKNOWN"))

    _step("FINAL REPORT")
    console.print(Panel(
        Text.assemble(
            ("  ADVERSARIAL IMMUNITY SCORE\n\n", "bold white"),
            (f"  {score:.1f} / 100", BAND_STYLE.get(band, "white")),
            (f"  {band}\n\n", BAND_STYLE.get(band, "white")),
            (f"  {immunity.get('band_description', '')}", "dim"),
        ), border_style="bright_cyan", expand=False))

    pillar_table = Table(title="  Immunity Pillars", header_style="bold", border_style="cyan")
    pillar_table.add_column("Pillar", style="cyan")
    pillar_table.add_column("Score", justify="right")
    pillar_table.add_column("Assessed", justify="center")
    pillar_table.add_column("Findings", justify="right")
    for key, component in (immunity.get("components") or {}).items():
        value = float(component.get("score", 0.0))
        style = ("bold red" if value < 50 else "yellow" if value < 70 else "green")
        pillar_table.add_row(
            str(component.get("label", key)), Text(f"{value:.1f}", style=style),
            "[green]Y[/green]" if component.get("assessed") else "[yellow]N[/yellow]",
            str(component.get("findings", 0)))
    console.print(pillar_table)

    stories = story.get("stories") or []
    if stories:
        console.print(f"\n[bold]  THREAT NARRATIVES ({len(stories)}):[/bold]")
        for entry in stories:
            console.print(Text(f"  -> [{entry.get('pattern')}] {entry.get('title')} "
                               f"(confidence {float(entry.get('confidence', 0.0)):.2f})",
                               style="bold dark_orange"))

    console.print(Panel(str(briefing.get("briefing", "")),
                        title=f"  COMMANDER'S BRIEFING ({briefing.get('source', 'template')})",
                        border_style="bright_cyan"))

    table = _findings_table(report.get("findings") or [], limit=args.limit)
    if table:
        console.print(table)

    _print_summary(report.get("summary") or {})
    _print_limitations(report.get("limitations") or [], report.get("coverage"))

    if report.get("report_path"):
        console.print(f"\n[green]  Report saved:[/green] {report['report_path']}")
    console.print(f"[dim]  report_hash={str(report.get('report_hash', ''))[:32]}... "
                  f"schema_valid={report.get('schema_valid')}[/dim]")

    if args.json is not None:
        _emit_json(report, args.json)
    return _exit_code(report.get("summary") or {})


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="shield",
        description="SHIELD-CV — offline integrity assurance for multi-contributor "
                    "computer-vision pipelines.",
        epilog="Exit codes: 0 clean · 1 findings need review · 2 compromised · 3 error")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    subparsers = parser.add_subparsers(dest="command")

    def add_common(sub: argparse.ArgumentParser) -> None:
        """Attach options shared by every subcommand."""
        sub.add_argument("--json", nargs="?", const="", default=None,
                         metavar="PATH",
                         help="emit JSON (to PATH, or stdout when given no value)")
        sub.add_argument("--limit", type=int, default=20,
                         help="maximum findings to display (default 20)")

    # Smart scan - auto-detects dataset / model / records
    scan = subparsers.add_parser("scan", help="scan dataset, model, or inference records")
    scan.add_argument("target", help="path to dataset root, model file, or records JSON")
    scan.add_argument("--reference", default=None, help="trusted reference model for comparison")
    scan.add_argument("--images", default=None, help="reference image directory")
    scan.add_argument("--max-images", type=int, default=None, help="cap images scanned")
    add_common(scan)
    scan.set_defaults(func=cmd_scan)

    # Model integrity audit
    audit = subparsers.add_parser("audit", help="model integrity audit")
    audit.add_argument("model", help="path to .pt/.pth/.onnx model")
    audit.add_argument("--reference", default=None, help="trusted reference model")
    audit.add_argument("--images", default=None, help="reference images for activation clustering")
    audit.add_argument("--skip-neural-cleanse", action="store_true", help="skip Neural Cleanse (faster)")
    add_common(audit)
    audit.set_defaults(func=cmd_audit)

    # Full system assurance report
    report = subparsers.add_parser("report", help="full assurance report across all modules")
    report.add_argument("target", nargs="?", default=None,
                        help="'all' to auto-run every module, or omit for manual flags")
    report.add_argument("--data", default=None, help="dataset root to scan")
    report.add_argument("--model", default=None, help="model to audit")
    report.add_argument("--reference", default=None, help="trusted reference model")
    report.add_argument("--records", default=None, help="inference records to verify")
    report.add_argument("--baseline", default=None, help="drift baseline corpus")
    report.add_argument("--current", default=None, help="drift current corpus")
    report.add_argument("--max-images", type=int, default=None, help="cap images per module")
    report.add_argument("--skip-neural-cleanse", action="store_true", help="skip Neural Cleanse")
    report.add_argument("--no-llm", action="store_true", help="force template briefing")
    report.add_argument("--no-save", action="store_true", help="do not persist report")
    add_common(report)
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if getattr(args, "version", False):
        try:
            cfg = get_config()
            console.print(f"{cfg.get('project.name', 'SHIELD-CV')} "
                          f"v{cfg.get('project.version', '1.0.0')}")
        except Exception:
            console.print("SHIELD-CV v1.0.0")
        return EXIT_CLEAN

    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_CLEAN

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        return EXIT_ERROR
    except Exception as exc:
        console.print(f"\n[bold red]Command failed:[/bold red] {type(exc).__name__}: {exc}")
        LOGGER.error("CLI command failed: %s\n%s", exc, traceback.format_exc())
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
