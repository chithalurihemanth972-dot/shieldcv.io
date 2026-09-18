"""Per-contributor scanning agent.

Each contributor folder in a multi-contributor dataset is assigned its own
agent. An agent runs the full data integrity scan over *only* that
contributor's samples and reduces the result to an :class:`AgentReport` — a
compact, picklable summary that can cross a process boundary cheaply.

The isolation is deliberate. A dataset-wide scan lets a heavy contributor
dominate the population statistics that the detectors score against, so a
contributor who poisons a large share of the corpus can pull the baseline
towards themselves and appear normal. Scanning contributors independently and
comparing the resulting reports afterwards (see :mod:`src.agents.meeting`)
keeps that manipulation visible.

Agents never raise into the caller: any failure is captured on the report as
``status='ERROR'`` with the reason recorded, so one bad contributor folder can
never abort an office-wide run.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.config import get_config
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class AgentReport:
    """Result of one agent scanning one contributor.

    The report holds only primitives and plain containers so that it can be
    returned from a :class:`concurrent.futures.ProcessPoolExecutor` worker.

    Attributes:
        agent_id: Identifier of the agent that produced this report.
        contributor: Contributor name (normally the folder name).
        path: Filesystem path that was scanned.
        status: ``OK``, ``EMPTY`` or ``ERROR``.
        trust_level: Declared trust level from ``config/contributors.yaml``.
        registered: Whether the contributor is declared in the config.
        risk_score: Aggregate contributor risk in ``0..1``.
        verdict: Scan verdict reported by the data integrity engine.
        num_samples: Number of samples attributed to this contributor.
        num_findings: Total number of findings raised.
        findings: Full finding dictionaries for this contributor.
        by_attack_class: Count of findings per attack class.
        by_severity: Count of findings per severity band.
        attack_classes: Sorted list of distinct attack classes observed.
        density: Fraction of this contributor's samples that were flagged.
        affected_samples: Number of distinct flagged samples.
        assessment: Human-readable assessment sentence.
        class_distribution: Label distribution for this contributor.
        limitations: Coverage limitations encountered during the scan.
        duration_seconds: Wall-clock scan time.
        error: Error text when ``status='ERROR'``.
    """

    agent_id: str = ""
    contributor: str = ""
    path: str = ""
    status: str = "OK"
    trust_level: str = "UNVERIFIED"
    registered: bool = False
    risk_score: float = 0.0
    verdict: str = "UNKNOWN"
    num_samples: int = 0
    num_findings: int = 0
    findings: List[Dict[str, Any]] = field(default_factory=list)
    by_attack_class: Dict[str, int] = field(default_factory=dict)
    by_severity: Dict[str, int] = field(default_factory=dict)
    attack_classes: List[str] = field(default_factory=list)
    density: float = 0.0
    affected_samples: int = 0
    assessment: str = ""
    class_distribution: Dict[str, int] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return the report as a plain dictionary.

        Returns:
            Dictionary form of the report, safe for JSON serialisation.
        """
        try:
            return asdict(self)
        except Exception as exc:  # pragma: no cover - dataclass failure
            LOGGER.error("AgentReport.to_dict failed: %s", exc)
            return {"agent_id": self.agent_id, "contributor": self.contributor,
                    "status": "ERROR", "error": str(exc)}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AgentReport":
        """Rebuild a report from its dictionary form.

        Unknown keys are ignored so that reports written by an older version of
        the tool can still be loaded.

        Args:
            payload: Dictionary previously produced by :meth:`to_dict`.

        Returns:
            Reconstructed :class:`AgentReport`.
        """
        try:
            allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
            return cls(**{k: v for k, v in payload.items() if k in allowed})
        except Exception as exc:
            LOGGER.error("AgentReport.from_dict failed: %s", exc)
            return cls(status="ERROR", error=str(exc))


def _contributor_profile(contributor: str) -> Dict[str, Any]:
    """Look up a contributor's declared trust level.

    Args:
        contributor: Contributor name.

    Returns:
        Dictionary with ``trust_level`` and ``registered`` keys. An unknown
        contributor is reported as unregistered and ``UNVERIFIED`` — an
        undeclared data source is never treated as trusted.
    """
    try:
        profile = get_config().contributor(contributor) or {}
        return {
            "trust_level": str(profile.get("trust_level", "UNVERIFIED")),
            "registered": bool(profile.get("registered", False)),
        }
    except Exception as exc:
        LOGGER.error("contributor profile lookup failed for %s: %s", contributor, exc)
        return {"trust_level": "UNVERIFIED", "registered": False}


def _select_contributor(result: Dict[str, Any], contributor: str) -> Dict[str, Any]:
    """Pick this contributor's entry from a scan's contributor risk table.

    The engine keys the table by the contributor label it inferred from the
    folder layout, which may not match the folder name exactly (for example
    when a single folder holds one unnamed contributor). When there is exactly
    one entry it is used regardless of its key.

    Args:
        result: Raw data scanner result.
        contributor: Expected contributor name.

    Returns:
        The contributor risk entry, or an empty dictionary when absent.
    """
    try:
        table = result.get("contributor_risk") or {}
        if contributor in table:
            return dict(table[contributor])
        if len(table) == 1:
            return dict(next(iter(table.values())))
        return {}
    except Exception as exc:
        LOGGER.error("contributor selection failed: %s", exc)
        return {}


def agent_scan_contributor(path: str | Path,
                           contributor: Optional[str] = None,
                           agent_id: Optional[str] = None,
                           max_images: Optional[int] = None,
                           progress_callback: Optional[Callable[[str, int, int], None]] = None
                           ) -> AgentReport:
    """Scan a single contributor's folder and summarise the outcome.

    This is the unit of work dispatched to the process pool by
    :class:`src.agents.manager.OfficeManager`. It is a module-level function
    (not a method) precisely so that it is importable and picklable on Windows,
    where the spawn start method re-imports the worker target.

    Args:
        path: Contributor folder to scan.
        contributor: Contributor name. Defaults to the folder name.
        agent_id: Identifier for this agent. Defaults to ``AGENT-<contributor>``.
        max_images: Optional cap on images scanned, for fast demo runs.
        progress_callback: Optional ``(stage, done, total)`` progress callback.

    Returns:
        An :class:`AgentReport`. Failures are reported via ``status='ERROR'``
        rather than raised.
    """
    started = time.time()
    folder = Path(path)
    name = contributor or folder.name
    ident = agent_id or f"AGENT-{name}"
    report = AgentReport(agent_id=ident, contributor=name, path=str(folder))

    try:
        profile = _contributor_profile(name)
        report.trust_level = profile["trust_level"]
        report.registered = profile["registered"]

        if not folder.exists():
            report.status = "ERROR"
            report.error = f"path does not exist: {folder}"
            report.duration_seconds = round(time.time() - started, 3)
            return report

        # Imported lazily: in a spawned worker this keeps the parent's import
        # cost off the fork path, and it keeps a torch import failure inside
        # the try-block where it becomes a report error rather than a crash.
        from src.scanners.data_scanner import DataIntegrityEngine

        engine = DataIntegrityEngine()
        result = engine.scan(folder, max_images=max_images,
                             progress_callback=progress_callback)

        dataset = result.get("dataset") or {}
        summary = result.get("summary") or {}
        entry = _select_contributor(result, name)

        report.findings = list(result.get("findings") or [])
        report.num_findings = len(report.findings)
        report.num_samples = int(entry.get("samples", dataset.get("num_samples", 0)) or 0)
        report.verdict = str(summary.get("verdict", "UNKNOWN"))
        report.by_attack_class = dict(summary.get("by_attack_class") or {})
        report.by_severity = dict(summary.get("by_severity") or {})
        report.attack_classes = sorted(
            k for k, v in report.by_attack_class.items() if v)
        report.class_distribution = dict(dataset.get("class_distribution") or {})
        report.limitations = list(result.get("limitations") or [])

        report.risk_score = float(entry.get("risk", summary.get("risk_score", 0.0)) or 0.0)
        report.density = float(entry.get("density", 0.0) or 0.0)
        report.affected_samples = int(entry.get("affected_samples", 0) or 0)
        report.assessment = str(entry.get("assessment", "") or "")

        if report.num_samples == 0:
            # No samples is not a clean result — it is an absence of evidence,
            # and the meeting room must not average it in as a zero-risk peer.
            report.status = "EMPTY"
            report.limitations.append(
                f"No readable samples found for '{name}'; no integrity statement is made.")

        LOGGER.info("%s scanned %s: risk=%.3f, %d finding(s), status=%s",
                    ident, name, report.risk_score, report.num_findings, report.status)

    except Exception as exc:
        report.status = "ERROR"
        report.error = f"{type(exc).__name__}: {exc}"
        report.limitations.append(f"Agent scan failed for '{name}': {report.error}")
        LOGGER.error("%s failed on %s: %s\n%s", ident, folder, exc,
                     traceback.format_exc())

    report.duration_seconds = round(time.time() - started, 3)
    return report


def agent_scan_dict(task: Dict[str, Any]) -> Dict[str, Any]:
    """Dictionary in, dictionary out wrapper around :func:`agent_scan_contributor`.

    Process pools pickle both arguments and return values; passing plain
    dictionaries avoids requiring the dataclass definition to resolve
    identically in parent and child interpreters.

    Args:
        task: Mapping with ``path`` and optional ``contributor``, ``agent_id``
            and ``max_images`` keys.

    Returns:
        The resulting report in dictionary form.
    """
    try:
        return agent_scan_contributor(
            path=task.get("path", ""),
            contributor=task.get("contributor"),
            agent_id=task.get("agent_id"),
            max_images=task.get("max_images"),
        ).to_dict()
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("agent_scan_dict failed: %s", exc)
        return AgentReport(
            agent_id=str(task.get("agent_id", "AGENT-UNKNOWN")),
            contributor=str(task.get("contributor", "unknown")),
            path=str(task.get("path", "")),
            status="ERROR", error=str(exc),
        ).to_dict()


__all__ = ["AgentReport", "agent_scan_contributor", "agent_scan_dict"]
