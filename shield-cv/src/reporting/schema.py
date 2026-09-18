"""
Canonical Finding schema and report JSON Schema for SHIELD-CV.

Every detector in every module emits the *same* finding structure. This module
owns that contract: the :class:`Finding` dataclass, the severity/disposition
ladders derived from confidence, and a JSON Schema used to validate reports
before they leave the tool.
"""

from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

SCHEMA_VERSION = "1.0.0"


class Severity(str, Enum):
    """Operational severity ladder used across all modules."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

    @property
    def rank(self) -> int:
        """Numeric rank for sorting (higher is more severe)."""
        return {"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}[self.value]


class Disposition(str, Enum):
    """Recommended handling of an affected asset."""

    QUARANTINE = "QUARANTINE"
    REVIEW = "REVIEW"
    ACCEPT = "ACCEPT"


class AttackClass(str, Enum):
    """Every attack class SHIELD-CV can name in a finding."""

    TRIGGER_INJECTION = "TRIGGER_INJECTION"
    LABEL_FLIPPING = "LABEL_FLIPPING"
    NEAR_DUPLICATE_FLOODING = "NEAR_DUPLICATE_FLOODING"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
    SYSTEMATIC_MISLABELING = "SYSTEMATIC_MISLABELING"
    MODEL_BACKDOOR = "MODEL_BACKDOOR"
    MODEL_SUBSTITUTION = "MODEL_SUBSTITUTION"
    WEIGHT_ANOMALY = "WEIGHT_ANOMALY"
    ACTIVATION_ANOMALY = "ACTIVATION_ANOMALY"
    BEHAVIORAL_DEVIATION = "BEHAVIORAL_DEVIATION"
    INFERENCE_TAMPER = "INFERENCE_TAMPER"
    INFERENCE_REPLAY = "INFERENCE_REPLAY"
    CHAIN_BREAK = "CHAIN_BREAK"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"
    DISTRIBUTION_SHIFT = "DISTRIBUTION_SHIFT"
    COORDINATED_ATTACK = "COORDINATED_ATTACK"
    COVER_UP_ATTEMPT = "COVER_UP_ATTEMPT"
    CONTRIBUTOR_RISK = "CONTRIBUTOR_RISK"
    CROSS_CONTRIBUTOR_DISPARITY = "CROSS_CONTRIBUTOR_DISPARITY"
    MODEL_DISAGREEMENT = "MODEL_DISAGREEMENT"
    UNKNOWN = "UNKNOWN"


_COUNTER_LOCK = threading.Lock()
_COUNTERS: Dict[str, int] = {}


def next_finding_id(prefix: str = "FIND") -> str:
    """Generate the next sequential finding identifier.

    Args:
        prefix: Identifier prefix, e.g. ``"FIND"`` or ``"MDL"``.

    Returns:
        Zero-padded identifier such as ``FIND-001``.
    """
    with _COUNTER_LOCK:
        _COUNTERS[prefix] = _COUNTERS.get(prefix, 0) + 1
        return f"{prefix}-{_COUNTERS[prefix]:03d}"


def reset_finding_ids(prefix: Optional[str] = None) -> None:
    """Reset finding-ID counters so each scan starts at 001.

    Args:
        prefix: Reset only this prefix; all prefixes when ``None``.
    """
    with _COUNTER_LOCK:
        if prefix is None:
            _COUNTERS.clear()
        else:
            _COUNTERS.pop(prefix, None)


def severity_from_confidence(confidence: float,
                             critical: float = 0.85,
                             high: float = 0.65,
                             medium: float = 0.40) -> str:
    """Map a confidence score onto the severity ladder.

    Args:
        confidence: Detector confidence in ``[0, 1]``.
        critical: Threshold for CRITICAL.
        high: Threshold for HIGH.
        medium: Threshold for MEDIUM.

    Returns:
        Severity name.
    """
    try:
        value = float(confidence)
        if value >= critical:
            return Severity.CRITICAL.value
        if value >= high:
            return Severity.HIGH.value
        if value >= medium:
            return Severity.MEDIUM.value
        return Severity.LOW.value
    except Exception:
        return Severity.LOW.value


def disposition_from_confidence(confidence: float,
                                quarantine: float = 0.80,
                                review: float = 0.50) -> str:
    """Map a confidence score onto a recommended disposition.

    Args:
        confidence: Detector confidence in ``[0, 1]``.
        quarantine: Threshold at or above which the asset is quarantined.
        review: Threshold at or above which analyst review is required.

    Returns:
        Disposition name.
    """
    try:
        value = float(confidence)
        if value >= quarantine:
            return Disposition.QUARANTINE.value
        if value >= review:
            return Disposition.REVIEW.value
        return Disposition.ACCEPT.value
    except Exception:
        return Disposition.ACCEPT.value


@dataclass
class Finding:
    """A single integrity finding — the atomic output of every SHIELD-CV module.

    Attributes:
        finding_id: Unique identifier, e.g. ``FIND-001``.
        attack_class: One of :class:`AttackClass`.
        affected_asset: Image file, model file, record id or contributor.
        severity: One of :class:`Severity`.
        confidence: Detector confidence in ``[0, 1]``.
        reason: Human-readable explanation an officer can act on.
        evidence: Numeric evidence, always including the threshold applied.
        disposition: One of :class:`Disposition`.
        contributor: Attributed contributor, when known.
        module: Producing module name.
        detector: Specific detector within the module.
        asset_type: ``image`` | ``model`` | ``record`` | ``dataset`` | ``contributor``.
        timestamp_ns: Creation time in nanoseconds.
        related_findings: Identifiers of correlated findings.
    """

    finding_id: str
    attack_class: str
    affected_asset: str
    severity: str
    confidence: float
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    disposition: str = Disposition.REVIEW.value
    contributor: Optional[str] = None
    module: str = ""
    detector: str = ""
    asset_type: str = "unknown"
    timestamp_ns: int = 0
    related_findings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalise and clamp fields so no invalid finding can exist."""
        try:
            import time
            if not self.timestamp_ns:
                self.timestamp_ns = time.time_ns()
            self.confidence = float(max(0.0, min(1.0, float(self.confidence))))
            if self.severity not in Severity.__members__:
                self.severity = severity_from_confidence(self.confidence)
            if self.disposition not in Disposition.__members__:
                self.disposition = disposition_from_confidence(self.confidence)
            if self.attack_class not in AttackClass.__members__:
                LOGGER.debug("Unknown attack class '%s' normalised to UNKNOWN",
                             self.attack_class)
                self.attack_class = AttackClass.UNKNOWN.value
            if not isinstance(self.evidence, dict):
                self.evidence = {"value": str(self.evidence)}
        except Exception as exc:
            LOGGER.error("Finding normalisation failed: %s", exc)

    @property
    def severity_rank(self) -> int:
        """Numeric severity rank for sorting."""
        try:
            return Severity(self.severity).rank
        except Exception:
            return 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the finding to the canonical dictionary form.

        Returns:
            JSON-safe dictionary matching the SHIELD-CV finding schema.
        """
        try:
            payload = asdict(self)
            payload["confidence"] = round(float(self.confidence), 4)

            # Spec-facing aliases. The internal short names (disposition,
            # detector, asset_type) are used throughout the codebase, while the
            # published schema names are what an auditor reads in the report.
            # Emitting both keeps every existing consumer working and makes the
            # JSON self-describing, at the cost of a few duplicated fields.
            payload["recommended_disposition"] = self.disposition
            payload["detection_method"] = self.detector
            payload["affected_asset_type"] = self.asset_type
            payload["timestamp"] = self.iso_timestamp()
            return payload
        except Exception as exc:
            LOGGER.error("Finding.to_dict failed: %s", exc)
            return {"finding_id": self.finding_id, "error": str(exc)}

    def iso_timestamp(self) -> str:
        """Render the creation time as a UTC ISO-8601 string.

        Returns:
            Timestamp such as ``2025-01-15T14:23:01Z``.
        """
        try:
            from datetime import datetime, timezone
            seconds = float(self.timestamp_ns) / 1e9
            return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
        except Exception as exc:
            LOGGER.error("iso_timestamp failed: %s", exc)
            return ""

    @classmethod
    def create(cls, attack_class: str, affected_asset: str, confidence: float,
               reason: str, evidence: Dict[str, Any], module: str = "",
               detector: str = "", contributor: Optional[str] = None,
               asset_type: str = "unknown", prefix: str = "FIND",
               severity: Optional[str] = None,
               disposition: Optional[str] = None) -> "Finding":
        """Build a finding with automatic id, severity and disposition.

        Args:
            attack_class: Attack class name.
            affected_asset: Asset identifier.
            confidence: Detector confidence in ``[0, 1]``.
            reason: Human-readable explanation.
            evidence: Numeric evidence including thresholds.
            module: Producing module.
            detector: Producing detector.
            contributor: Attributed contributor.
            asset_type: Type of the affected asset.
            prefix: Finding-ID prefix.
            severity: Explicit severity override.
            disposition: Explicit disposition override.

        Returns:
            A fully-populated :class:`Finding`.
        """
        return cls(
            finding_id=next_finding_id(prefix),
            # An Enum member stringifies to "AttackClass.X", not "X", which
            # would silently produce schema-invalid findings. Normalise here so
            # callers may pass either the member or its value.
            attack_class=(attack_class.value if isinstance(attack_class, Enum)
                          else str(attack_class)),
            affected_asset=str(affected_asset),
            severity=severity or severity_from_confidence(confidence),
            confidence=float(confidence),
            reason=reason,
            evidence=evidence or {},
            disposition=disposition or disposition_from_confidence(confidence),
            contributor=contributor,
            module=module,
            detector=detector,
            asset_type=asset_type,
        )


# ---------------------------------------------------------------------------
# JSON Schema for full reports
# ---------------------------------------------------------------------------
FINDING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["finding_id", "attack_class", "affected_asset", "severity",
                 "confidence", "reason", "evidence", "disposition"],
    "properties": {
        "finding_id": {"type": "string", "minLength": 1},
        "attack_class": {"type": "string", "enum": [a.value for a in AttackClass]},
        "affected_asset": {"type": "string"},
        "severity": {"type": "string", "enum": [s.value for s in Severity]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string", "minLength": 1},
        "evidence": {"type": "object"},
        "disposition": {"type": "string", "enum": [d.value for d in Disposition]},
        "contributor": {"type": ["string", "null"]},
        "module": {"type": "string"},
        "detector": {"type": "string"},
        "asset_type": {"type": "string"},
        "timestamp_ns": {"type": "integer"},
        "related_findings": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": True,
}

REPORT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "SHIELD-CV Assurance Report",
    "type": "object",
    "required": ["report_id", "schema_version", "generated_at", "target",
                 "findings", "summary", "coverage", "limitations"],
    "properties": {
        "report_id": {"type": "string"},
        "schema_version": {"type": "string"},
        "generated_at": {"type": "string"},
        "generated_at_ns": {"type": "integer"},
        "classification": {"type": "string"},
        "tool": {"type": "object"},
        "target": {"type": "object"},
        "findings": {"type": "array", "items": FINDING_SCHEMA},
        "summary": {
            "type": "object",
            "required": ["total_findings", "risk_score", "verdict"],
            "properties": {
                "total_findings": {"type": "integer", "minimum": 0},
                "risk_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "verdict": {"type": "string"},
                "by_severity": {"type": "object"},
                "by_attack_class": {"type": "object"},
            },
        },
        "contributor_risk": {"type": "object"},
        "modules": {"type": "object"},
        "coverage": {
            "type": "object",
            "required": ["supported", "unsupported"],
            "properties": {
                "supported": {"type": "array", "items": {"type": "string"}},
                "unsupported": {"type": "array", "items": {"type": "string"}},
                "modules_run": {"type": "array", "items": {"type": "string"}},
                "modules_skipped": {"type": "array"},
            },
        },
        "limitations": {"type": "array", "items": {"type": "string"}},
        "methodology": {"type": "object"},
        "threat_story": {"type": ["object", "null"]},
        "immunity_score": {"type": ["object", "null"]},
        "briefing": {"type": ["object", "string", "null"]},
        "audit_trail_hash": {"type": "string"},
        "report_hash": {"type": "string"},
    },
    "additionalProperties": True,
}


def validate_finding(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate one finding dictionary against the finding schema.

    Args:
        payload: Finding dictionary.

    Returns:
        ``{"valid": bool, "errors": [str]}``.
    """
    return _validate(payload, FINDING_SCHEMA)


def validate_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a full report dictionary against the report schema.

    Args:
        payload: Report dictionary.

    Returns:
        ``{"valid": bool, "errors": [str]}``.
    """
    return _validate(payload, REPORT_SCHEMA)


def _validate(payload: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Run jsonschema validation, degrading to manual checks when absent.

    Args:
        payload: Object to validate.
        schema: JSON Schema to validate against.

    Returns:
        ``{"valid": bool, "errors": [str], "validator": str}``.
    """
    try:
        import jsonschema
        validator = jsonschema.Draft7Validator(schema)
        errors = [f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
                  for e in sorted(validator.iter_errors(payload), key=lambda e: list(e.path))]
        return {"valid": not errors, "errors": errors[:25], "validator": "jsonschema"}
    except ImportError:
        missing = [key for key in schema.get("required", []) if key not in payload]
        return {
            "valid": not missing,
            "errors": [f"missing required field: {k}" for k in missing],
            "validator": "builtin-fallback (jsonschema not installed)",
        }
    except Exception as exc:
        LOGGER.error("Schema validation error: %s", exc)
        return {"valid": False, "errors": [str(exc)], "validator": "error"}


def sort_findings(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort findings by severity then confidence, most serious first.

    Args:
        findings: Finding dictionaries.

    Returns:
        Newly sorted list.
    """
    try:
        ranks = {s.value: s.rank for s in Severity}
        return sorted(
            findings,
            key=lambda f: (ranks.get(f.get("severity", "LOW"), 0),
                           float(f.get("confidence", 0.0))),
            reverse=True,
        )
    except Exception as exc:
        LOGGER.error("sort_findings failed: %s", exc)
        return list(findings)


def summarize_findings(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate findings into severity/class/disposition counts and a risk score.

    Risk is **severity-dominated**: the worst single finding sets the floor, and
    the remaining findings can only add a bounded amount on top.

    A plain noisy-OR was used previously and was wrong. Because every finding
    multiplies the surviving probability, volume alone drove the score to
    certainty: 100 benign LOW findings at confidence 0.25 scored 0.978,
    *outranking* a single CRITICAL at 0.95. Real corpora always carry a long
    tail of weak findings, so that behaviour made a clean dataset look
    compromised and destroyed the score's ability to discriminate.

    The replacement mirrors the validated immunity-score curve:

    * ``worst``   — the highest severity-weighted confidence present.
    * ``breadth`` — a saturating function of how many *other* findings there
      are, ``1 - exp(-tail / 6)``, which grows quickly for the first few and
      then flattens, so volume informs the score without dominating it.
    * ``risk = clip(0.75 * worst + 0.25 * breadth * worst, 0, 1)``

    Scaling breadth by ``worst`` is the key property: many trivial findings
    cannot manufacture a high score on their own, because there is no severe
    finding for the breadth term to amplify.

    Args:
        findings: Finding dictionaries.

    Returns:
        Summary dictionary including ``risk_score`` and ``verdict``.
    """
    try:
        by_severity: Dict[str, int] = {s.value: 0 for s in Severity}
        by_class: Dict[str, int] = {}
        by_disposition: Dict[str, int] = {d.value: 0 for d in Disposition}

        weights = {"CRITICAL": 1.0, "HIGH": 0.7, "MEDIUM": 0.4, "LOW": 0.15, "INFO": 0.0}
        contributions: List[float] = []
        for finding in findings:
            severity = finding.get("severity", "LOW")
            by_severity[severity] = by_severity.get(severity, 0) + 1
            attack_class = finding.get("attack_class", "UNKNOWN")
            by_class[attack_class] = by_class.get(attack_class, 0) + 1
            disposition = finding.get("disposition", "ACCEPT")
            by_disposition[disposition] = by_disposition.get(disposition, 0) + 1
            contribution = weights.get(severity, 0.1) * float(finding.get("confidence", 0.0))
            contributions.append(min(max(contribution, 0.0), 1.0))

        if contributions:
            worst = max(contributions)
            tail = len(contributions) - 1
            breadth = 1.0 - math.exp(-tail / 6.0)
            risk = round(min(1.0, max(0.0, 0.75 * worst + 0.25 * breadth * worst)), 4)
        else:
            risk = 0.0
        if by_severity.get("CRITICAL", 0) > 0:
            verdict = "COMPROMISED"
        elif by_severity.get("HIGH", 0) > 0:
            verdict = "SUSPICIOUS"
        elif by_severity.get("MEDIUM", 0) > 0:
            verdict = "CAUTION"
        elif findings:
            verdict = "MINOR_ANOMALIES"
        else:
            verdict = "CLEAN"

        return {
            "total_findings": len(findings),
            "risk_score": risk,
            "verdict": verdict,
            "by_severity": by_severity,
            "by_attack_class": by_class,
            "by_disposition": by_disposition,
        }
    except Exception as exc:
        LOGGER.error("summarize_findings failed: %s", exc)
        return {"total_findings": len(findings), "risk_score": 0.0, "verdict": "ERROR",
                "by_severity": {}, "by_attack_class": {}, "error": str(exc)}


__all__ = [
    "Finding", "Severity", "Disposition", "AttackClass", "SCHEMA_VERSION",
    "next_finding_id", "reset_finding_ids", "severity_from_confidence",
    "disposition_from_confidence", "validate_finding", "validate_report",
    "sort_findings", "summarize_findings", "FINDING_SCHEMA", "REPORT_SCHEMA",
]
