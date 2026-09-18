"""
Detection benchmark for SHIELD-CV.

Runs the data integrity engine against a poisoned dataset and scores its output
against the attack scripts' ``ground_truth.json`` manifest. This is what turns
"the tool produced findings" into "the tool produced *correct* findings", and it
is the number that belongs in an acceptance report.

Usage:
    python demo/benchmark.py --clean demo/data/clean --poisoned demo/data/poisoned
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scanners.data_scanner import DataIntegrityEngine  # noqa: E402
from src.utils.logger import get_console, get_logger  # noqa: E402

LOGGER = get_logger(__name__)
CONSOLE = get_console()


def _score(truth: Set[str], flagged: Set[str], universe: int) -> Dict[str, Any]:
    """Compute precision/recall/F1 for one attack class.

    Args:
        truth: Asset names that were genuinely attacked.
        flagged: Asset names the scanner flagged for this class.
        universe: Total number of assets scanned.

    Returns:
        Dictionary of confusion-matrix counts and derived metrics.
    """
    try:
        true_positive = len(truth & flagged)
        false_positive = len(flagged - truth)
        false_negative = len(truth - flagged)
        true_negative = max(universe - true_positive - false_positive - false_negative, 0)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        return {
            "ground_truth": len(truth),
            "flagged": len(flagged),
            "true_positives": true_positive,
            "false_positives": false_positive,
            "false_negatives": false_negative,
            "true_negatives": true_negative,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_positive_rate": round(false_positive / max(universe - len(truth), 1), 4),
        }
    except Exception as exc:
        LOGGER.error("_score failed: %s", exc)
        return {"error": str(exc)}


def _baseline_duplicates(poisoned_root: Path, injected: Set[str]) -> Set[str]:
    """Identify near-duplicates that exist in the dataset independently of the attack.

    Any image that is *not* part of the injected flood but still has a perceptual
    twin is a pre-existing duplicate. Counting those against the detector would
    understate its precision, since flagging them is correct behaviour.

    Args:
        poisoned_root: Root of the poisoned dataset.
        injected: File names known to be attack-generated duplicates.

    Returns:
        Set of file names that are naturally-occurring near-duplicates.
    """
    try:
        from src.analysis.outlier import _hamming, compute_phashes
        from src.loaders.yolo_loader import load_dataset

        dataset = load_dataset(poisoned_root)
        samples = [s for s in dataset.valid_samples if s.name not in injected]
        if len(samples) < 2:
            return set()
        hashes = compute_phashes([str(s.image_path) for s in samples])
        indices = sorted(hashes)
        natural: Set[str] = set()
        for position, i in enumerate(indices):
            for j in indices[position + 1:]:
                if _hamming(hashes[i], hashes[j]) <= 5:
                    natural.add(samples[i].name)
                    natural.add(samples[j].name)
        return natural
    except Exception as exc:
        LOGGER.warning("Baseline duplicate survey failed: %s", exc)
        return set()


def benchmark(poisoned_dir: str | Path,
              min_confidence: float = 0.0,
              fmt: Optional[str] = None) -> Dict[str, Any]:
    """Scan a poisoned dataset and score the findings against ground truth.

    Args:
        poisoned_dir: Directory containing the poisoned data and ``ground_truth.json``.
        min_confidence: Ignore findings below this confidence when scoring.
        fmt: Dataset format override.

    Returns:
        Benchmark report with per-attack metrics and the raw scan summary.
    """
    report: Dict[str, Any] = {"target": str(poisoned_dir), "metrics": {}, "errors": []}
    try:
        root = Path(poisoned_dir)
        truth_file = root / "ground_truth.json"
        if not truth_file.is_file():
            report["errors"].append(
                f"No ground_truth.json in {root} — run the attack scripts first")
            return report

        truth = json.loads(truth_file.read_text(encoding="utf-8"))
        result = DataIntegrityEngine().scan(root, fmt=fmt)
        findings = [f for f in result.get("findings", [])
                    if float(f.get("confidence", 0.0)) >= float(min_confidence)]
        universe = int(result.get("dataset", {}).get("num_valid_images", 0)) or 1

        def flagged_for(*classes: str) -> Set[str]:
            """Collect asset names flagged under any of the given attack classes."""
            return {f["affected_asset"] for f in findings
                    if f.get("attack_class") in classes}

        if "trigger_inject" in truth:
            expected = {p["file"] for p in truth["trigger_inject"].get("poisoned_files", [])}
            report["metrics"]["TRIGGER_INJECTION"] = _score(
                expected, flagged_for("TRIGGER_INJECTION"), universe)

        if "label_flip" in truth or "trigger_inject" in truth:
            # Ground truth for "mislabeled" is the union of every attack that
            # changed a label. trigger_inject relabels its poisoned images to the
            # target class, so those samples are genuinely mislabeled too and a
            # label-flip detector is correct to flag them.
            expected = {p["file"] for p in truth.get("label_flip", {}).get("flipped_files", [])}
            relabelled = {
                p["file"] for p in truth.get("trigger_inject", {}).get("poisoned_files", [])
                if p.get("original_labels") and p["original_labels"][0] != p.get("new_label")
            }
            expected |= relabelled
            report["metrics"]["LABEL_FLIPPING"] = _score(
                expected, flagged_for("LABEL_FLIPPING", "SYSTEMATIC_MISLABELING"), universe)
            report["metrics"]["LABEL_FLIPPING"]["explicit_flips"] = len(
                truth.get("label_flip", {}).get("flipped_files", []))
            report["metrics"]["LABEL_FLIPPING"]["trigger_relabels"] = len(relabelled)

        if "duplicate_flood" in truth:
            expected: Set[str] = set()
            for cluster in truth["duplicate_flood"].get("clusters", []):
                expected.add(cluster["source_file"])
                expected.update(d["file"] for d in cluster.get("duplicates", []))
            # A duplicate finding is emitted once per cluster, with the whole
            # cluster listed in evidence.member_files. Scoring only the
            # representative name would under-count every member the tool did
            # in fact identify, so expand each finding to its members.
            flagged: Set[str] = set()
            for finding in findings:
                if finding.get("attack_class") != "NEAR_DUPLICATE_FLOODING":
                    continue
                flagged.add(finding["affected_asset"])
                flagged.update(finding.get("evidence", {}).get("member_files", []))
            # The clean baseline may itself contain naturally-similar images
            # (procedural generators repeat; real archives contain burst frames).
            # Those are true near-duplicates, not scanner errors, so they are
            # excluded from the false-positive count and reported separately.
            baseline = _baseline_duplicates(root, expected)
            report["metrics"]["NEAR_DUPLICATE_FLOODING"] = _score(
                expected, flagged - baseline, universe)
            report["metrics"]["NEAR_DUPLICATE_FLOODING"]["pre_existing_duplicates_excluded"] = len(
                baseline & flagged)
            report["metrics"]["NEAR_DUPLICATE_FLOODING"]["clusters_flagged"] = sum(
                1 for f in findings if f.get("attack_class") == "NEAR_DUPLICATE_FLOODING")
            report["metrics"]["NEAR_DUPLICATE_FLOODING"]["flood_clusters_expected"] = len(
                truth["duplicate_flood"].get("clusters", []))

        report["scan_summary"] = result.get("summary", {})
        report["contributor_risk"] = {
            k: v.get("risk") for k, v in result.get("contributor_risk", {}).items()}
        report["duration_seconds"] = result.get("duration_seconds", 0.0)
        report["universe"] = universe
        report["min_confidence"] = min_confidence
        return report
    except Exception as exc:
        LOGGER.error("benchmark failed: %s", exc)
        report["errors"].append(str(exc))
        return report


def print_report(report: Dict[str, Any]) -> None:
    """Render a benchmark report as a rich table.

    Args:
        report: Output of :func:`benchmark`.
    """
    try:
        from rich.table import Table
        table = Table(title="SHIELD-CV Detection Benchmark", header_style="bold cyan")
        for column in ("Attack", "Truth", "Flagged", "TP", "FP", "FN",
                       "Precision", "Recall", "F1"):
            table.add_column(column, justify="right" if column != "Attack" else "left")

        for name, metrics in report.get("metrics", {}).items():
            if "error" in metrics:
                continue
            recall = metrics["recall"]
            colour = "green" if recall >= 0.75 else ("yellow" if recall >= 0.5 else "red")
            table.add_row(
                name, str(metrics["ground_truth"]), str(metrics["flagged"]),
                str(metrics["true_positives"]), str(metrics["false_positives"]),
                str(metrics["false_negatives"]),
                f"{metrics['precision']:.2f}",
                f"[{colour}]{recall:.2f}[/{colour}]",
                f"{metrics['f1']:.2f}",
            )
        CONSOLE.print(table)
        summary = report.get("scan_summary", {})
        CONSOLE.print(f"Verdict: [bold]{summary.get('verdict', '?')}[/bold]  "
                      f"risk={summary.get('risk_score', 0)}  "
                      f"scanned {report.get('universe', 0)} images in "
                      f"{report.get('duration_seconds', 0)}s")
    except Exception as exc:
        LOGGER.error("print_report failed: %s", exc)
        print(json.dumps(report.get("metrics", {}), indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point for the benchmark.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code (0 when every attack was detected at all).
    """
    parser = argparse.ArgumentParser(description="SHIELD-CV detection benchmark")
    parser.add_argument("--poisoned", default="demo/data/poisoned")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--format", default=None, choices=["coco", "yolo"])
    parser.add_argument("--json", action="store_true", help="Emit raw JSON")
    args = parser.parse_args(argv)

    report = benchmark(args.poisoned, min_confidence=args.min_confidence, fmt=args.format)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    if report.get("errors"):
        for error in report["errors"]:
            CONSOLE.print(f"[red]{error}[/red]")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
