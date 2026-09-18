"""
REPORT GENERATOR — assembles the single authoritative assurance report.

The report is the deliverable an accreditation authority signs against, so it is
built to be defensible rather than merely informative:

* **Every finding** carries its reason, numeric evidence, the threshold applied,
  a confidence and a recommended disposition.
* **A coverage statement** records what was assessed AND what was not. A report
  that lists only findings invites the reader to assume everything unmentioned
  is sound; that misreading is the most dangerous failure mode of a tool like
  this, so the gaps are stated as prominently as the results.
* **A limitations section** aggregates every degradation encountered — missing
  weights, unavailable backbones, underpowered statistics — from all modules.
* **An audit trail hash** binds the report to the tamper-evident database chain,
  so a report cannot be altered after issue without detection.

The report is written atomically and its own content hash is embedded, allowing
any recipient to verify the file has not been modified in transit.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import get_config
from src.crypto.hashing import sha256_json
from src.reporting.schema import (
    SCHEMA_VERSION,
    sort_findings,
    summarize_findings,
    validate_report,
)
from src.utils.helpers import (
    environment_summary,
    new_scan_id,
    utc_now_iso,
    write_json,
)
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

MODULE_COVERAGE = {
    "data_scanner": {
        "area": "Training dataset integrity",
        "detects": [
            "Backdoor trigger injection (FFT + spatial + SVD spectral consensus)",
            "Label flipping (KNN neighbourhood disagreement)",
            "Near-duplicate flooding (perceptual hash clustering)",
            "Out-of-distribution samples (Mahalanobis + energy score)",
            "Systematic mislabeling (confident learning)",
            "Per-contributor risk aggregation",
        ],
        "cannot_detect": [
            "Clean-label poisoning that leaves no spatial, spectral or "
            "neighbourhood signature",
            "Semantically plausible mislabels that agree with their visual neighbours",
            "Poisoning present in EVERY sample, which leaves no in-distribution "
            "baseline to compare against",
        ],
    },
    "model_auditor": {
        "area": "Model integrity",
        "detects": [
            "Model substitution (SHA-256 weight hash vs accredited reference)",
            "Anomalous weight distributions (std, kurtosis, sparsity per layer)",
            "Implanted backdoors (Neural Cleanse trigger reverse-engineering)",
            "Poisoned activation pathways (KMeans activation clustering)",
            "Behavioural deviation from a reference model (black-box fingerprinting)",
        ],
        "cannot_detect": [
            "Backdoors targeting classes beyond the scanned class cap",
            "Distributed backdoors spread across all classes, which leave no "
            "outlier for the MAD test to find",
            "Any weight-level property when only a prediction API is available",
        ],
    },
    "crypto_chain": {
        "area": "Inference provenance",
        "detects": [
            "Post-hoc edits to recorded outputs (payload re-hashing)",
            "Records altered and re-hashed by a capable attacker (Ed25519 signature)",
            "Deleted or reordered records (chain link + sequence gap)",
            "Replayed records (nonce reuse)",
        ],
        "cannot_detect": [
            "Falsification at the moment of record creation — a compromised "
            "recorder signs a lie correctly, and the chain will verify",
            "Wholesale substitution of the entire chain together with its signing key",
        ],
    },
    "drift_detector": {
        "area": "Operational distribution shift",
        "detects": [
            "Pixel-level shift across brightness, contrast, colour warmth, edge "
            "density and sharpness",
            "Feature-level shift (MMD, centroid displacement, variance ratio)",
            "Separation of natural operational drift from suspicious manipulation",
        ],
        "cannot_detect": [
            "Manipulation that mimics a physically coherent environmental change "
            "in both pixel and feature space",
            "Gradual drift slower than the comparison window",
        ],
    },
}


class ReportGenerator:
    """Builds and persists the consolidated SHIELD-CV assurance report.

    Attributes:
        cfg: Effective configuration.
    """

    def __init__(self, config: Optional[Any] = None,
                 database: Optional[Any] = None) -> None:
        """Initialise the generator.

        Args:
            config: Optional configuration override.
            database: Optional pre-initialised database handle.  When ``None``
                the database is lazily imported on first use, which is safe but
                means a misconfigured path surfaces late.  Passing the database
                explicitly avoids this surprise.
        """
        self.cfg = config or get_config()
        self._database = database

    # ------------------------------------------------------------------
    def build(self, module_results: Dict[str, Dict[str, Any]],
              threat_story: Optional[Dict[str, Any]] = None,
              immunity: Optional[Dict[str, Any]] = None,
              briefing: Optional[Dict[str, Any]] = None,
              scan_id: Optional[str] = None,
              target: Any = "",
              persist: bool = True) -> Dict[str, Any]:
        """Assemble the full assurance report.

        Args:
            module_results: Mapping of module name to result dictionary.
            threat_story: Optional threat-story result.
            immunity: Optional immunity score result.
            briefing: Optional briefing result.
            scan_id: Optional scan identifier; generated when omitted.
            target: Human-readable description of what was assessed.
            persist: Whether to write the report to disk and the database.

        Returns:
            The complete report dictionary, including ``report_path`` when
            persisted.
        """
        started = time.time()
        identifier = scan_id or new_scan_id("SCAN")

        report: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "report_id": identifier,
            "generated_at": utc_now_iso(),
            "framework": {
                "name": self.cfg.get("project.name", "SHIELD-CV"),
                "full_name": self.cfg.get("project.full_name", ""),
                "version": self.cfg.get("project.version", "1.0.0"),
                "classification": self.cfg.get("project.classification", "RESTRICTED"),
                "offline_mode": bool(self.cfg.get("project.offline_mode", True)),
            },
            # REPORT_SCHEMA requires `target` to be an object, so a plain
            # string description is normalised into one rather than failing
            # validation for a cosmetic reason.
            "target": (target if isinstance(target, dict)
                       else {"description": str(target)}),
            "classification": self.cfg.get("project.classification", "RESTRICTED"),
            "tool": {
                "name": self.cfg.get("project.name", "SHIELD-CV"),
                "version": self.cfg.get("project.version", "1.0.0"),
            },
            "generated_at_ns": time.time_ns(),
            "findings": [], "summary": {}, "modules": {},
            "coverage_statement": {}, "limitations": [],
            "threat_story": {}, "immunity": {}, "briefing": {},
            "environment": environment_summary(),
        }

        try:
            findings: List[Dict[str, Any]] = []
            limitations: List[str] = []

            for module_name, payload in (module_results or {}).items():
                if not isinstance(payload, dict):
                    continue
                module_findings = payload.get("findings", []) or []
                findings.extend(module_findings)
                for statement in payload.get("limitations", []) or []:
                    limitations.append(f"[{module_name}] {statement}")

                report["modules"][module_name] = {
                    key: value for key, value in payload.items()
                    if key not in ("findings",)
                }

            if isinstance(threat_story, dict) and threat_story:
                findings.extend(threat_story.get("findings", []) or [])
                for statement in threat_story.get("limitations", []) or []:
                    limitations.append(f"[threat_story] {statement}")
                report["threat_story"] = {
                    key: value for key, value in threat_story.items()
                    if key != "findings"
                }

            if isinstance(immunity, dict) and immunity:
                report["immunity"] = immunity
                for statement in immunity.get("limitations", []) or []:
                    limitations.append(f"[immunity] {statement}")

            if isinstance(briefing, dict) and briefing:
                report["briefing"] = {
                    "text": briefing.get("briefing", ""),
                    "source": briefing.get("source", "template"),
                    "model": briefing.get("model"),
                    "disclaimer": briefing.get("disclaimer", ""),
                }
                for statement in briefing.get("limitations", []) or []:
                    limitations.append(f"[briefing] {statement}")

            report["findings"] = sort_findings(findings)
            report["summary"] = summarize_findings(findings)
            coverage = self._coverage_statement(module_results)
            # `coverage` is the schema-required key; `coverage_statement` is
            # kept as the human-facing alias so existing consumers and the
            # dashboard continue to work.
            report["coverage"] = coverage
            report["coverage_statement"] = coverage
            report["limitations"] = list(dict.fromkeys(limitations))

            report["audit_trail"] = self._audit_trail(identifier, report, persist)

            # The content hash is computed over the report WITHOUT the hash
            # field itself, so a recipient can recompute and compare.
            report["report_hash"] = sha256_json(
                {k: v for k, v in report.items() if k != "report_hash"})

            validation = validate_report(report)
            report["schema_valid"] = bool(validation.get("valid", True))
            if not validation.get("valid", True):
                report["schema_errors"] = validation.get("errors", [])[:10]
                LOGGER.warning("Report failed schema validation: %s",
                               validation.get("errors", [])[:3])

            if persist:
                report["report_path"] = self._write(identifier, report)

            report["duration_seconds"] = round(time.time() - started, 2)
            LOGGER.info("Report %s assembled: %d finding(s), verdict %s",
                        identifier, len(report["findings"]),
                        report["summary"].get("verdict"))
            return report
        except Exception as exc:
            LOGGER.error("Report generation failed: %s", exc, exc_info=True)
            report["limitations"].append(f"Report assembly error: {exc}")
            report["summary"] = report.get("summary") or summarize_findings([])
            report["duration_seconds"] = round(time.time() - started, 2)
            return report

    # ------------------------------------------------------------------
    def _coverage_statement(self,
                            module_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Build the explicit statement of what was and was not assessed.

        Args:
            module_results: Per-module results.

        Returns:
            Coverage statement dictionary.
        """
        statement: Dict[str, Any] = {
            # `supported` / `unsupported` are the schema-mandated keys: the flat
            # list of attack classes this run could and could not detect.
            # `assessed` / `not_assessed` carry the same information grouped by
            # module for human readers.
            "supported": [], "unsupported": [],
            "assessed": [], "not_assessed": [], "known_blind_spots": [],
            "narrative": "",
        }
        try:
            for module_name, definition in MODULE_COVERAGE.items():
                payload = (module_results or {}).get(module_name)
                if isinstance(payload, dict) and payload:
                    entry = {
                        "module": module_name,
                        "area": definition["area"],
                        "detects": definition["detects"],
                        "findings": len(payload.get("findings", []) or []),
                    }
                    if payload.get("access_level"):
                        entry["access_level"] = payload["access_level"]
                    if payload.get("backbone"):
                        entry["backbone"] = payload["backbone"]
                    statement["assessed"].append(entry)
                    statement["supported"].extend(definition["detects"])
                    statement["unsupported"].extend(definition["cannot_detect"])
                    statement["known_blind_spots"].extend(
                        f"{definition['area']}: {item}"
                        for item in definition["cannot_detect"])
                else:
                    # An unassessed module's whole detection surface is
                    # unsupported for this run — not silently omitted.
                    statement["unsupported"].extend(
                        f"{item} (module not run)" for item in definition["detects"])
                    statement["not_assessed"].append({
                        "module": module_name, "area": definition["area"],
                        "consequence": (
                            f"No conclusion can be drawn about {definition['area'].lower()}. "
                            "Absence of findings here reflects absence of assessment."),
                    })

            assessed = [e["area"] for e in statement["assessed"]]
            missing = [e["area"] for e in statement["not_assessed"]]
            parts = []
            if assessed:
                parts.append("This assessment covered: " + "; ".join(assessed) + ".")
            if missing:
                parts.append(
                    "It did NOT cover: " + "; ".join(missing)
                    + ". No statement of integrity is made about these areas.")
            parts.append(
                "All detection methods have known blind spots, listed under "
                "known_blind_spots. A clean result within an assessed area means no "
                "evidence of compromise was found by the methods listed — it is not "
                "proof that no compromise exists.")
            statement["narrative"] = " ".join(parts)
            return statement
        except Exception as exc:
            LOGGER.error("_coverage_statement failed: %s", exc)
            statement["narrative"] = f"Coverage statement unavailable: {exc}"
            return statement

    def _audit_trail(self, scan_id: str, report: Dict[str, Any],
                     persist: bool) -> Dict[str, Any]:
        """Record the report in the audit database and capture the chain hash.

        Args:
            scan_id: Scan identifier.
            report: The report being assembled.
            persist: Whether to write to the database.

        Returns:
            Audit trail metadata.
        """
        trail: Dict[str, Any] = {"recorded": False}
        try:
            database = self._database
            if database is None:
                from src.database import get_database
                database = get_database()
            if persist:
                database.append_audit(
                    actor="report_generator", action="REPORT_GENERATED",
                    target=scan_id,
                    details={
                        "findings": len(report.get("findings", [])),
                        "verdict": report.get("summary", {}).get("verdict", "UNKNOWN"),
                    })
                trail["recorded"] = True

            trail["audit_trail_hash"] = database.audit_trail_hash()
            verification = database.verify_audit_chain()
            trail["chain_valid"] = bool(verification.get("valid", False))
            trail["chain_entries"] = verification.get("entries", 0)
            if not trail["chain_valid"]:
                trail["chain_warning"] = (
                    "The local audit chain did not verify. The database may have been "
                    "modified outside SHIELD-CV; treat prior reports as unproven.")
            return trail
        except Exception as exc:
            LOGGER.error("_audit_trail failed: %s", exc)
            trail["error"] = str(exc)
            return trail

    def _write(self, scan_id: str, report: Dict[str, Any]) -> str:
        """Write the report JSON to the configured reports directory.

        Args:
            scan_id: Scan identifier used in the filename.
            report: The report dictionary.

        Returns:
            Path written, or an empty string on failure.
        """
        try:
            directory = self.cfg.path("reports")
            destination = Path(directory) / f"{scan_id}.json"
            if write_json(destination, report):
                LOGGER.info("Report written to %s", destination)
                return str(destination)
            return ""
        except Exception as exc:
            LOGGER.error("_write failed: %s", exc)
            return ""


def generate_report(module_results: Dict[str, Dict[str, Any]],
                    threat_story: Optional[Dict[str, Any]] = None,
                    immunity: Optional[Dict[str, Any]] = None,
                    briefing: Optional[Dict[str, Any]] = None,
                    scan_id: Optional[str] = None,
                    target: Any = "",
                    persist: bool = True) -> Dict[str, Any]:
    """Convenience wrapper building the consolidated assurance report.

    Args:
        module_results: Mapping of module name to result dictionary.
        threat_story: Optional threat-story result.
        immunity: Optional immunity result.
        briefing: Optional briefing result.
        scan_id: Optional scan identifier.
        target: Description of what was assessed.
        persist: Whether to write the report to disk.

    Returns:
        The report dictionary.
    """
    try:
        return ReportGenerator().build(module_results, threat_story=threat_story,
                                       immunity=immunity, briefing=briefing,
                                       scan_id=scan_id, target=target, persist=persist)
    except Exception as exc:
        LOGGER.error("generate_report failed: %s", exc)
        return {"report_id": scan_id or "", "findings": [], "limitations": [str(exc)]}


__all__ = ["ReportGenerator", "generate_report", "MODULE_COVERAGE"]
