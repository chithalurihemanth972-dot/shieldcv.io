"""
TRIGGER DETECTION ENGINE — the three-method consensus detector.

This module exposes trigger detection as a standalone, reusable engine so it can
be driven directly (CLI ``shield trigger``, dashboard, unit tests) independently
of a full dataset scan. :mod:`src.scanners.data_scanner` performs the same
analysis inline as part of its wider pipeline; this engine is the focused,
single-purpose entry point onto the same validated primitives.

Three independent methods vote:

**Method 1 — FFT frequency analysis.** A pasted patch has hard edges, which add
broadband high-frequency energy. Measures the high/low frequency energy ratio
and an isolated-periodic-peak score.

**Method 2 — Spatial block scanning.** A trigger is a small region that is
internally uniform yet far from the image mean. Scored with a 1-pixel-stride
sliding window rather than a fixed lattice.

**Method 3 — SVD spectral signatures.** Poisoned samples in a class share a
common direction in embedding space and project onto the top singular vector as
a group.

Consensus: agreement of 2+ methods yields HIGH confidence, via a noisy-OR fusion
that rewards independent agreement without letting any single method saturate.

A critical correctness note carried over from validation: **absolute spatial
thresholds are unusable**. Per-image block statistics self-normalise, so a clean
image can score higher than a poisoned one. Every spatial and frequency score
here is converted to a *dataset-relative* modified Z-score before any threshold
is applied, which is why this engine requires a population of images rather than
scoring one image in isolation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.analysis.embeddings import get_extractor
from src.analysis.frequency import (
    analyze_image_frequency,
    population_frequency_outliers,
)
from src.analysis.spectral import analyze_classes
from src.analysis.statistics import confidence_from_zscore, modified_zscore
from src.config import get_config
from src.reporting.schema import AttackClass, Finding, reset_finding_ids, summarize_findings
from src.utils.image_utils import load_image
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

METHOD_FFT = "FFT_FREQUENCY_ANALYSIS"
METHOD_BLOCK = "SPATIAL_BLOCK_SCANNING"
METHOD_SPECTRAL = "SVD_SPECTRAL_SIGNATURES"


class TriggerDetector:
    """Detects backdoor trigger patches by three-method consensus.

    Attributes:
        cfg: Effective configuration.
        findings: Findings produced by the most recent run.
        limitations: Statements of what could not be assessed.
    """

    MODULE_NAME = "trigger_detector"

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialise the detector and read its thresholds.

        Args:
            config: Optional configuration override.
        """
        self.cfg = config or get_config()
        # Same config section as the inline detector in data_scanner, so the two
        # entry points can never drift apart in their thresholds.
        self.trigger_cfg = self.cfg.section("data_scanner").get("trigger", {}) or {}
        self.block_size = int(self.trigger_cfg.get("block_size", 16))
        self.hf_ratio_threshold = float(
            self.trigger_cfg.get("fft_high_freq_ratio_threshold", 0.32))
        self.hf_zscore_threshold = float(self.trigger_cfg.get("fft_zscore_threshold", 2.5))
        self.block_zscore_threshold = float(
            self.trigger_cfg.get("block_variance_zscore", 4.0))
        self.spectral_zscore_threshold = float(
            self.trigger_cfg.get("spectral_zscore_threshold", 2.0))
        self.spectral_min_class_size = int(self.trigger_cfg.get("spectral_min_class_size", 12))
        self.consensus_bonus = float(self.trigger_cfg.get("consensus_bonus", 0.12))
        self.findings: List[Finding] = []
        self.limitations: List[str] = []

    # ------------------------------------------------------------------
    def detect(self, image_paths: Sequence[str | Path],
               labels: Optional[Sequence[int]] = None,
               contributors: Optional[Sequence[Optional[str]]] = None,
               progress_callback: Optional[Callable[[str, int, int], None]] = None
               ) -> Dict[str, Any]:
        """Run three-method trigger detection across a population of images.

        Args:
            image_paths: Images to analyse. A population is required because all
                scores are dataset-relative.
            labels: Optional per-image class labels enabling Method 3.
            contributors: Optional per-image contributor attribution.
            progress_callback: Optional ``callable(stage, done, total)``.

        Returns:
            Result with ``findings``, ``summary``, ``methods``, ``per_image``
            and ``limitations``.
        """
        started = time.time()
        reset_finding_ids("TRG")
        self.findings = []
        self.limitations = []

        paths = [str(p) for p in image_paths]
        result: Dict[str, Any] = {
            "module": self.MODULE_NAME, "num_images": len(paths),
            "findings": [], "summary": {}, "methods": {}, "per_image": [],
            "limitations": [], "duration_seconds": 0.0,
        }

        def report(stage: str, done: int, total: int) -> None:
            """Forward progress to the caller's callback."""
            if progress_callback:
                try:
                    progress_callback(stage, done, total)
                except Exception:
                    pass

        try:
            if len(paths) < 2:
                reason = ("Trigger detection requires a population of images: all spatial "
                          "and frequency scores are dataset-relative modified Z-scores, "
                          "which are undefined for a single image")
                self.limitations.append(reason)
                result["limitations"] = self.limitations
                result["summary"] = summarize_findings([])
                LOGGER.warning(reason)
                return result

            # -- Methods 1 and 2: per-image frequency and spatial scores ----
            report("frequency", 0, len(paths))
            per_image: List[Dict[str, Any]] = []
            readable: List[int] = []
            for index, path in enumerate(paths):
                image = load_image(path)
                if image is None:
                    per_image.append({"available": False, "path": path})
                    continue
                metrics = analyze_image_frequency(
                    image, block_size=self.block_size,
                    block_zscore_threshold=self.block_zscore_threshold)
                metrics["path"] = path
                metrics["available"] = True
                per_image.append(metrics)
                readable.append(index)
                report("frequency", index + 1, len(paths))

            if not readable:
                reason = "No images could be read: trigger detection NOT AVAILABLE"
                self.limitations.append(reason)
                result["limitations"] = self.limitations
                result["summary"] = summarize_findings([])
                return result

            unreadable = len(paths) - len(readable)
            if unreadable:
                self.limitations.append(
                    f"{unreadable} image(s) could not be read and were not analysed")

            fft_flags = self._method_fft(per_image, readable)
            block_flags = self._method_blocks(per_image, readable)

            # -- Method 3: SVD spectral signatures --------------------------
            report("spectral", 0, 1)
            spectral_flags = self._method_spectral(paths, labels, readable)
            report("spectral", 1, 1)

            result["methods"] = {
                METHOD_FFT: {"flagged": len(fft_flags),
                             "threshold": self.hf_zscore_threshold,
                             "ratio_threshold": self.hf_ratio_threshold},
                METHOD_BLOCK: {"flagged": len(block_flags),
                               "threshold": self.block_zscore_threshold},
                METHOD_SPECTRAL: {"flagged": len(spectral_flags),
                                  "threshold": self.spectral_zscore_threshold,
                                  "available": bool(labels is not None)},
            }

            # -- Consensus fusion -------------------------------------------
            report("fusion", 0, 1)
            self._fuse(paths, per_image, fft_flags, block_flags, spectral_flags,
                       contributors)
            report("fusion", 1, 1)

            payloads = [f.to_dict() for f in self.findings]
            result["findings"] = payloads
            result["summary"] = summarize_findings(payloads)
            result["per_image"] = per_image
            result["limitations"] = list(dict.fromkeys(self.limitations))
            result["duration_seconds"] = round(time.time() - started, 2)

            LOGGER.info("Trigger detection complete: %d finding(s) over %d image(s) (%.1fs)",
                        len(payloads), len(paths), result["duration_seconds"])
            return result
        except Exception as exc:
            LOGGER.error("Trigger detection failed: %s", exc, exc_info=True)
            self.limitations.append(f"Trigger detection aborted: {exc}")
            result["findings"] = [f.to_dict() for f in self.findings]
            result["summary"] = summarize_findings(result["findings"])
            result["limitations"] = self.limitations
            result["duration_seconds"] = round(time.time() - started, 2)
            return result

    # ------------------------------------------------------------------
    def _method_fft(self, per_image: List[Dict[str, Any]],
                    readable: Sequence[int]) -> Dict[int, Dict[str, Any]]:
        """Method 1 — flag images with anomalous high-frequency energy.

        Args:
            per_image: Per-image frequency metrics.
            readable: Indices of successfully analysed images.

        Returns:
            Mapping of image index to FFT evidence.
        """
        flags: Dict[int, Dict[str, Any]] = {}
        try:
            ratios = [float(per_image[i].get("high_freq_ratio", 0.0)) for i in readable]
            population = population_frequency_outliers(ratios, self.hf_zscore_threshold)
            zscores = population.get("zscores", [])

            peaks = [float(per_image[i].get("spectral_peak_ratio", 0.0)) for i in readable]
            peak_z = modified_zscore(np.asarray(peaks, dtype=np.float64))

            for position, index in enumerate(readable):
                ratio = ratios[position]
                z = float(zscores[position]) if position < len(zscores) else 0.0
                pz = float(peak_z[position]) if position < len(peak_z) else 0.0

                # Either a population-relative excess of broadband high-frequency
                # energy, or an isolated periodic peak (a regular pasted pattern).
                by_zscore = z > self.hf_zscore_threshold
                by_peak = pz > self.hf_zscore_threshold
                if not (by_zscore or by_peak):
                    continue

                flags[index] = {
                    "high_freq_ratio": round(ratio, 5),
                    "high_freq_zscore": round(z, 3),
                    "spectral_peak_ratio": round(float(peaks[position]), 2),
                    "spectral_peak_zscore": round(pz, 3),
                    "threshold": self.hf_zscore_threshold,
                    "rule": ("population-relative high-frequency energy"
                             if by_zscore else "isolated periodic spectral peak"),
                }
            return flags
        except Exception as exc:
            LOGGER.error("_method_fft failed: %s", exc)
            self.limitations.append(f"FFT frequency analysis error: {exc}")
            return flags

    def _method_blocks(self, per_image: List[Dict[str, Any]],
                       readable: Sequence[int]) -> Dict[int, Dict[str, Any]]:
        """Method 2 — flag images containing an anomalous uniform block.

        Args:
            per_image: Per-image spatial metrics.
            readable: Indices of successfully analysed images.

        Returns:
            Mapping of image index to spatial evidence.
        """
        flags: Dict[int, Dict[str, Any]] = {}
        try:
            scores = [float(per_image[i].get("patch_score", 0.0)) for i in readable]
            zscores = modified_zscore(np.asarray(scores, dtype=np.float64))

            for position, index in enumerate(readable):
                z = float(zscores[position]) if position < len(zscores) else 0.0
                if z <= self.block_zscore_threshold:
                    continue
                metrics = per_image[index]
                flags[index] = {
                    "patch_score": round(scores[position], 4),
                    "patch_zscore": round(z, 3),
                    "threshold": self.block_zscore_threshold,
                    "block_row": metrics.get("patch_row"),
                    "block_col": metrics.get("patch_col"),
                    "block_size": self.block_size,
                    "corner": metrics.get("patch_corner"),
                    "rule": ("uniform region far from the image mean, scored "
                             "relative to the dataset population"),
                }
            return flags
        except Exception as exc:
            LOGGER.error("_method_blocks failed: %s", exc)
            self.limitations.append(f"Spatial block scanning error: {exc}")
            return flags

    def _method_spectral(self, paths: Sequence[str],
                         labels: Optional[Sequence[int]],
                         readable: Sequence[int]) -> Dict[int, Dict[str, Any]]:
        """Method 3 — flag samples lying on a class's poisoned spectral signature.

        Args:
            paths: All image paths.
            labels: Per-image class labels.
            readable: Indices of successfully analysed images.

        Returns:
            Mapping of image index to spectral evidence.
        """
        flags: Dict[int, Dict[str, Any]] = {}
        try:
            if labels is None:
                self.limitations.append(
                    "No labels supplied: SVD spectral signature analysis NOT AVAILABLE "
                    "(it compares samples within each class)")
                return flags
            if len(labels) != len(paths):
                self.limitations.append(
                    f"Label count ({len(labels)}) does not match image count ({len(paths)}): "
                    "spectral analysis NOT AVAILABLE")
                return flags

            extractor = get_extractor()
            embedded = extractor.embed_paths([paths[i] for i in readable])
            if not embedded.is_reliable:
                self.limitations.append(
                    "Embedding backbone unavailable: SVD spectral signature analysis "
                    "NOT AVAILABLE")
                self.limitations.extend(embedded.warnings[:2])
                return flags

            features = np.asarray(embedded.features, dtype=np.float64)
            # embed_paths reports which of the inputs it actually embedded, so
            # map back through both index layers to reach original positions.
            original = [readable[i] for i in embedded.indices]
            embedded_labels = [int(labels[i]) for i in original]

            spectral = analyze_classes(
                features, embedded_labels,
                zscore_threshold=self.spectral_zscore_threshold,
                min_class_size=self.spectral_min_class_size)

            for class_id, data in spectral.items():
                if not data.get("available"):
                    continue
                zscores = data.get("zscores", [])
                members = data.get("global_indices", [])
                for local_index in data.get("outlier_indices", []):
                    if local_index >= len(members):
                        continue
                    position = members[local_index]
                    if position >= len(original):
                        continue
                    flags[original[position]] = {
                        "spectral_score": round(float(
                            zscores[local_index]) if local_index < len(zscores) else 0.0, 3),
                        "threshold": self.spectral_zscore_threshold,
                        "class": int(class_id),
                        "rule": ("projection onto the class's top singular vector, "
                                 "modified Z-score using MAD"),
                    }
            return flags
        except Exception as exc:
            LOGGER.error("_method_spectral failed: %s", exc)
            self.limitations.append(f"SVD spectral signature error: {exc}")
            return flags

    # ------------------------------------------------------------------
    def _fuse(self, paths: Sequence[str], per_image: List[Dict[str, Any]],
              fft_flags: Dict[int, Dict[str, Any]],
              block_flags: Dict[int, Dict[str, Any]],
              spectral_flags: Dict[int, Dict[str, Any]],
              contributors: Optional[Sequence[Optional[str]]]) -> None:
        """Combine the three methods into findings via noisy-OR consensus.

        Args:
            paths: All image paths.
            per_image: Per-image metrics.
            fft_flags: Method 1 hits.
            block_flags: Method 2 hits.
            spectral_flags: Method 3 hits.
            contributors: Optional per-image contributor attribution.
        """
        try:
            flagged = set(fft_flags) | set(block_flags) | set(spectral_flags)
            for index in sorted(flagged):
                evidence: Dict[str, Any] = {}
                methods: List[str] = []
                confidences: List[float] = []
                reasons: List[str] = []

                if index in fft_flags:
                    data = fft_flags[index]
                    evidence.update(data)
                    methods.append(METHOD_FFT)
                    confidences.append(confidence_from_zscore(
                        max(data["high_freq_zscore"], data["spectral_peak_zscore"]),
                        self.hf_zscore_threshold))
                    reasons.append(
                        f"FFT analysis shows {data['rule']} "
                        f"(high-frequency ratio {data['high_freq_ratio']:.4f}, "
                        f"modified Z={data['high_freq_zscore']:.2f} against a threshold of "
                        f"{self.hf_zscore_threshold})")

                if index in block_flags:
                    data = block_flags[index]
                    evidence.update(data)
                    methods.append(METHOD_BLOCK)
                    confidences.append(confidence_from_zscore(
                        data["patch_zscore"], self.block_zscore_threshold))
                    reasons.append(
                        f"spatial scanning located a {data['block_size']}x{data['block_size']} "
                        f"uniform region at row {data['block_row']}, column {data['block_col']} "
                        f"({data.get('corner', 'unknown')} of frame) with patch score "
                        f"{data['patch_score']:.3f} (modified Z={data['patch_zscore']:.2f} "
                        f"against a threshold of {self.block_zscore_threshold})")

                if index in spectral_flags:
                    data = spectral_flags[index]
                    evidence.update(data)
                    methods.append(METHOD_SPECTRAL)
                    confidences.append(confidence_from_zscore(
                        data["spectral_score"], self.spectral_zscore_threshold))
                    reasons.append(
                        f"the sample's embedding projects onto the poisoned spectral "
                        f"signature of class {data['class']} at modified Z="
                        f"{data['spectral_score']:.2f} (threshold "
                        f"{self.spectral_zscore_threshold}), indicating it shares a common "
                        "hidden direction with other suspected samples in that class")

                if not confidences:
                    continue

                # Noisy-OR: independent methods reinforce one another without
                # any single one saturating the score.
                combined = 1.0
                for value in confidences:
                    combined *= (1.0 - float(np.clip(value, 0.0, 0.99)))
                confidence = 1.0 - combined
                if len(methods) >= 2:
                    confidence = min(0.99, confidence + self.consensus_bonus)

                evidence["detection_methods_agreed"] = len(methods)
                evidence["methods"] = methods
                evidence["consensus"] = "HIGH" if len(methods) >= 2 else "SINGLE_METHOD"

                prefix = (f"{len(methods)} independent methods agree" if len(methods) >= 2
                          else "One method flagged this sample")
                reason = (f"{prefix} that '{Path(paths[index]).name}' carries an injected "
                          f"trigger patch: " + "; ".join(reasons) + ".")
                if len(methods) == 1:
                    reason += (" Single-method detections are weaker evidence and are "
                               "reported for review rather than automatic quarantine.")

                contributor = None
                if contributors is not None and index < len(contributors):
                    contributor = contributors[index]

                self.findings.append(Finding.create(
                    attack_class=AttackClass.TRIGGER_INJECTION.value,
                    affected_asset=Path(paths[index]).name,
                    confidence=float(confidence),
                    reason=reason,
                    evidence=evidence,
                    module=self.MODULE_NAME,
                    detector="+".join(methods),
                    contributor=contributor,
                    asset_type="data_sample",
                    prefix="TRG",
                ))
        except Exception as exc:
            LOGGER.error("_fuse failed: %s", exc)
            self.limitations.append(f"Consensus fusion error: {exc}")


def detect_triggers(image_paths: Sequence[str | Path],
                    labels: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Convenience wrapper running three-method trigger detection.

    Args:
        image_paths: Images to analyse.
        labels: Optional class labels enabling spectral signature analysis.

    Returns:
        Detection result dictionary.
    """
    try:
        return TriggerDetector().detect(image_paths, labels=labels)
    except Exception as exc:
        LOGGER.error("detect_triggers failed: %s", exc)
        return {"module": "trigger_detector", "findings": [], "limitations": [str(exc)]}


__all__ = ["TriggerDetector", "detect_triggers",
           "METHOD_FFT", "METHOD_BLOCK", "METHOD_SPECTRAL"]
