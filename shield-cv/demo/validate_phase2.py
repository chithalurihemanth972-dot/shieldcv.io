"""
Phase 2 validation harness — scores all four scanners against known ground truth.

Every check below is a pass/fail assertion with an expected outcome, so a
regression in any engine surfaces immediately rather than hiding behind a
plausible-looking report:

* Module 1 Data Scanner        — precision/recall per attack class.
* Module 2 Model Auditor       — backdoored model detected, clean twin not.
* Module 3 Provenance Engine   — all four tamper modes caught, clean chain clean.
* Module 4 Drift Detector      — natural drift separated from manipulation.

Usage:
    python demo/validate_phase2.py
    python demo/validate_phase2.py --modules data model crypto drift --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_console, get_logger  # noqa: E402

LOGGER = get_logger(__name__)
CONSOLE = get_console()

ROOT = Path(__file__).resolve().parent.parent


class Check:
    """A single named assertion with its expected and observed values.

    Attributes:
        name: Human-readable description.
        passed: Whether the assertion held.
        expected: Expected value.
        observed: Observed value.
        detail: Optional supporting text.
    """

    def __init__(self, name: str, passed: bool, expected: Any,
                 observed: Any, detail: str = "") -> None:
        """Record one check result.

        Args:
            name: Description of the assertion.
            passed: Whether it held.
            expected: Expected value.
            observed: Observed value.
            detail: Optional supporting text.
        """
        self.name = name
        self.passed = bool(passed)
        self.expected = expected
        self.observed = observed
        self.detail = detail

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the check.

        Returns:
            JSON-safe dictionary.
        """
        return {"check": self.name, "passed": self.passed,
                "expected": self.expected, "observed": self.observed,
                "detail": self.detail}


def validate_data_scanner() -> List[Check]:
    """Score the data integrity engine against the poisoned demo dataset.

    Returns:
        List of checks.
    """
    checks: List[Check] = []
    try:
        from demo.benchmark import benchmark

        poisoned = ROOT / "demo" / "data" / "poisoned"
        if not poisoned.is_dir():
            return [Check("data: poisoned dataset present", False, "exists", "missing",
                          "Run the attack pipeline first")]

        report = benchmark(str(poisoned))
        scores = report.get("metrics", {})
        expectations = {
            "TRIGGER_INJECTION": 0.70,
            "LABEL_FLIPPING": 0.70,
            "NEAR_DUPLICATE_FLOODING": 0.90,
        }
        for attack, minimum in expectations.items():
            entry = scores.get(attack, {})
            f1 = float(entry.get("f1", 0.0))
            checks.append(Check(
                f"data: {attack} F1 >= {minimum}", f1 >= minimum, f">={minimum}",
                round(f1, 3),
                f"precision {entry.get('precision', 0):.2f} recall {entry.get('recall', 0):.2f}"))

        verdict = report.get("scan_summary", {}).get("verdict", "")
        checks.append(Check("data: poisoned dataset verdict is COMPROMISED",
                            verdict == "COMPROMISED", "COMPROMISED", verdict))
        return checks
    except Exception as exc:
        LOGGER.error("validate_data_scanner failed: %s", exc)
        return checks + [Check("data: scanner ran without error", False, "no exception", str(exc))]


def validate_model_auditor() -> List[Check]:
    """Verify the model auditor separates the backdoored model from its clean twin.

    Returns:
        List of checks.
    """
    checks: List[Check] = []
    try:
        from src.scanners.model_auditor import ModelIntegrityEngine

        models = ROOT / "demo" / "models"
        backdoored = models / "backdoored_model_scripted.pt"
        clean = models / "clean_model_scripted.pt"
        if not backdoored.is_file() or not clean.is_file():
            return [Check("model: demo models present", False, "exists", "missing",
                          "Run attacks/model_backdoor.py first")]

        truth_path = models / "ground_truth_model_backdoor.json"
        target_class = 2
        if truth_path.is_file():
            target_class = int(json.loads(truth_path.read_text(
                encoding="utf-8")).get("parameters", {}).get("target_class", 2))

        backdoor_result = ModelIntegrityEngine().audit(str(backdoored),
                                                       reference_path=str(clean))
        classes = [f["attack_class"] for f in backdoor_result["findings"]]

        checks.append(Check("model: backdoored model flagged MODEL_BACKDOOR",
                            "MODEL_BACKDOOR" in classes, "present",
                            "present" if "MODEL_BACKDOOR" in classes else "absent"))

        flagged_targets = [c["target_class"]
                           for c in backdoor_result["neural_cleanse"].get(
                               "flagged_classes", [])]
        checks.append(Check(f"model: Neural Cleanse identifies true target class "
                            f"{target_class}",
                            target_class in flagged_targets, target_class,
                            flagged_targets))

        checks.append(Check("model: weight substitution detected vs reference",
                            "MODEL_SUBSTITUTION" in classes, "present",
                            "present" if "MODEL_SUBSTITUTION" in classes else "absent"))
        checks.append(Check("model: backdoored verdict is COMPROMISED",
                            backdoor_result["summary"].get("verdict") == "COMPROMISED",
                            "COMPROMISED", backdoor_result["summary"].get("verdict")))

        clean_result = ModelIntegrityEngine().audit(str(clean), reference_path=str(clean))
        clean_classes = [f["attack_class"] for f in clean_result["findings"]]
        checks.append(Check("model: clean twin NOT flagged as backdoored",
                            "MODEL_BACKDOOR" not in clean_classes, "absent",
                            "present" if "MODEL_BACKDOOR" in clean_classes else "absent"))
        checks.append(Check("model: clean twin matches its own weight hash",
                            "MODEL_SUBSTITUTION" not in clean_classes, "absent",
                            "present" if "MODEL_SUBSTITUTION" in clean_classes else "absent"))
        checks.append(Check("model: audit states an access level",
                            bool(backdoor_result.get("access_level")), "set",
                            backdoor_result.get("access_level")))
        return checks
    except Exception as exc:
        LOGGER.error("validate_model_auditor failed: %s", exc)
        return checks + [Check("model: auditor ran without error", False,
                               "no exception", str(exc))]


def validate_crypto_chain() -> List[Check]:
    """Verify every tamper mode is caught and the clean chain stays clean.

    Returns:
        List of checks.
    """
    checks: List[Check] = []
    try:
        from attacks.inference_tamper import generate_clean_chain, tamper_records
        from src.scanners.crypto_chain import InferenceProvenanceEngine

        records_dir = ROOT / "demo" / "records"
        clean_path = records_dir / "clean_chain.json"
        if not clean_path.is_file():
            generate_clean_chain(clean_path, count=30)

        clean_result = InferenceProvenanceEngine().verify(str(clean_path))
        checks.append(Check("crypto: clean chain verifies with zero findings",
                            len(clean_result["findings"]) == 0, 0,
                            len(clean_result["findings"])))
        checks.append(Check("crypto: clean chain hash links intact",
                            bool(clean_result["chain"]["valid"]), True,
                            clean_result["chain"]["valid"]))

        expectations = {
            "edit": "INFERENCE_TAMPER",
            "rehash": "SIGNATURE_INVALID",
            "delete": "CHAIN_BREAK",
            "replay": "INFERENCE_REPLAY",
        }
        for mode, expected_class in expectations.items():
            target = records_dir / f"validate_{mode}.json"
            tamper_records(clean_path, target, mode=mode, count=2)
            result = InferenceProvenanceEngine().verify(str(target))
            classes = {f["attack_class"] for f in result["findings"]}
            checks.append(Check(
                f"crypto: '{mode}' attack detected as {expected_class}",
                expected_class in classes, expected_class,
                sorted(classes) or "nothing detected"))
        return checks
    except Exception as exc:
        LOGGER.error("validate_crypto_chain failed: %s", exc)
        return checks + [Check("crypto: provenance engine ran without error", False,
                               "no exception", str(exc))]


def validate_drift_detector() -> List[Check]:
    """Verify natural drift is distinguished from deliberate manipulation.

    Returns:
        List of checks.
    """
    checks: List[Check] = []
    try:
        from src.scanners.drift_detector import DistributionShiftDetector

        baseline = ROOT / "demo" / "data" / "clean"
        drift_root = ROOT / "demo" / "data" / "drift"
        truth_path = drift_root / "ground_truth_drift.json"
        if not truth_path.is_file():
            return [Check("drift: scenarios present", False, "exists", "missing",
                          "Run demo/generate_drift_data.py first")]

        scenarios = json.loads(truth_path.read_text(encoding="utf-8")).get("scenarios", {})
        for name, meta in scenarios.items():
            result = DistributionShiftDetector().compare(
                str(baseline), meta["path"], max_images=140)
            expected = meta["expected_shift_type"]
            observed = result["shift_type"]
            checks.append(Check(
                f"drift: '{name}' classified {expected}", observed == expected,
                expected, observed,
                f"risk {result['risk_score']:.2f} severity {result['severity']}"))

        self_result = DistributionShiftDetector().compare(
            str(baseline), str(baseline), max_images=140)
        checks.append(Check("drift: baseline vs itself shows no drift",
                            self_result["shift_type"] == "NO_SIGNIFICANT_DRIFT",
                            "NO_SIGNIFICANT_DRIFT", self_result["shift_type"]))
        return checks
    except Exception as exc:
        LOGGER.error("validate_drift_detector failed: %s", exc)
        return checks + [Check("drift: detector ran without error", False,
                               "no exception", str(exc))]


VALIDATORS: Dict[str, Callable[[], List[Check]]] = {
    "data": validate_data_scanner,
    "model": validate_model_auditor,
    "crypto": validate_crypto_chain,
    "drift": validate_drift_detector,
}


def main(argv: Optional[List[str]] = None) -> int:
    """Run the selected validators and print a pass/fail report.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code (0 when every check passes).
    """
    parser = argparse.ArgumentParser(description="SHIELD-CV Phase 2 validation")
    parser.add_argument("--modules", nargs="*", default=list(VALIDATORS),
                        choices=list(VALIDATORS))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    results: Dict[str, List[Check]] = {}
    for module in args.modules:
        CONSOLE.print(f"[cyan]Validating {module}...[/cyan]")
        results[module] = VALIDATORS[module]()

    total = sum(len(v) for v in results.values())
    passed = sum(1 for v in results.values() for c in v if c.passed)

    if args.json:
        print(json.dumps({
            "passed": passed, "total": total,
            "duration_seconds": round(time.time() - started, 2),
            "modules": {k: [c.to_dict() for c in v] for k, v in results.items()},
        }, indent=2))
    else:
        from rich.table import Table

        for module, checks in results.items():
            table = Table(title=f"Module: {module}", header_style="bold cyan")
            table.add_column("Result", width=6)
            table.add_column("Check", overflow="fold")
            table.add_column("Expected", overflow="fold")
            table.add_column("Observed", overflow="fold")
            for check in checks:
                table.add_row(
                    "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]",
                    check.name, str(check.expected), str(check.observed))
            CONSOLE.print(table)

        colour = "green" if passed == total else "red"
        CONSOLE.print(f"[bold {colour}]{passed}/{total} checks passed[/bold {colour}] "
                      f"in {time.time() - started:.1f}s")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
