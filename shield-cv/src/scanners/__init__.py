"""Scanner modules: data integrity, model integrity, provenance, distribution shift.

Each scanner exposes an engine class for repeated use (the config and any
embedding backbone are loaded once) and a module-level convenience function for
one-shot calls from the CLI or the dashboard.
"""

from src.scanners.data_scanner import DataIntegrityEngine, scan_dataset
from src.scanners.crypto_chain import InferenceProvenanceEngine, verify_records
from src.scanners.model_auditor import ModelIntegrityEngine, audit_model
from src.scanners.drift_detector import DistributionShiftDetector, detect_drift
from src.scanners.trigger_detector import TriggerDetector, detect_triggers

__all__ = [
    "DataIntegrityEngine", "scan_dataset",
    "InferenceProvenanceEngine", "verify_records",
    "ModelIntegrityEngine", "audit_model",
    "DistributionShiftDetector", "detect_drift",
    "TriggerDetector", "detect_triggers",
]
