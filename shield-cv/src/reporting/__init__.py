"""Reporting layer: the Finding/report schema and the assurance report builder.

`schema` defines the canonical Finding structure that every detector emits;
`report_generator` aggregates module results into a hash-stamped, schema-valid
assurance report carrying an explicit coverage statement.
"""

from src.reporting.schema import (
    Severity, Disposition, AttackClass, Finding,
    next_finding_id, reset_finding_ids,
    severity_from_confidence, disposition_from_confidence,
    validate_finding, validate_report, sort_findings, summarize_findings,
    SCHEMA_VERSION, FINDING_SCHEMA, REPORT_SCHEMA,
)
from src.reporting.report_generator import ReportGenerator, generate_report

__all__ = [
    "Severity", "Disposition", "AttackClass", "Finding",
    "next_finding_id", "reset_finding_ids",
    "severity_from_confidence", "disposition_from_confidence",
    "validate_finding", "validate_report", "sort_findings", "summarize_findings",
    "SCHEMA_VERSION", "FINDING_SCHEMA", "REPORT_SCHEMA",
    "ReportGenerator", "generate_report",
]
