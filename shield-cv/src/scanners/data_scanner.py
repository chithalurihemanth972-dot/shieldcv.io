"""
MODULE 1 — Data Integrity Engine.

Scans a COCO or YOLO dataset for five families of poisoning, using a FROZEN
ResNet-18 for all feature extraction (never retrained, always ``torch.no_grad()``):

============================  =========================================================
Attack                        Method
============================  =========================================================
TRIGGER_INJECTION             FFT high-frequency ratio + 16x16 block variance/gradient
                              anomalies + SVD spectral signatures (modified Z > 2.0)
LABEL_FLIPPING                KNN + distance from class centroid (> mean + 2σ)
NEAR_DUPLICATE_FLOODING       pHash, Hamming < 5, union-find clustering
OUT_OF_DISTRIBUTION           Mahalanobis distance + energy-based OOD from logits
SYSTEMATIC_MISLABELING        Confident learning (no retraining: nearest-centroid
                              soft labels over frozen embeddings)
============================  =========================================================

Sample-level findings are then aggregated into **contributor risk scores**, which
is what actually drives procurement decisions in a multi-vendor pipeline.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.analysis.embeddings import EmbeddingResult, get_extractor
from src.analysis.frequency import analyze_image_frequency, population_frequency_outliers
from src.analysis.outlier import (
    confident_learning,
    detect_near_duplicates,
    energy_ood,
    knn_centroid_outliers,
    mahalanobis_ood,
)
from src.analysis.spectral import analyze_classes
from src.analysis.statistics import aggregate_risk, confidence_from_zscore, modified_zscore
from src.config import get_config
from src.loaders.yolo_loader import Dataset, DatasetSample, load_dataset
from src.reporting.schema import AttackClass, Finding, reset_finding_ids, summarize_findings
from src.utils.image_utils import load_image
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class DataIntegrityEngine:
    """Detects dataset poisoning and aggregates risk per contributor.

    Attributes:
        cfg: Effective :class:`~src.config.Config`.
        findings: Findings produced by the most recent scan.
        warnings: Degradation notices for the report's limitations section.
    """

    MODULE_NAME = "data_scanner"

    # Confidence ceiling for a trigger finding supported by only one of the
    # three detection methods. 0.45 keeps it in REVIEW territory (HIGH starts
    # at 0.65), so uncorroborated leads never read as confirmed detections.
    SINGLE_METHOD_CONFIDENCE_CAP: float = 0.45

    def __init__(self, config: Optional[Any] = None,
                 extractor: Optional[Any] = None) -> None:
        """Initialise the engine and its detector thresholds.

        Args:
            config: Optional :class:`~src.config.Config` override.
            extractor: Optional pre-built embedding extractor (reused by agents).
        """
        self.cfg = config or get_config()
        self.section = self.cfg.section("data_scanner")
        self.findings: List[Finding] = []
        self.warnings: List[str] = []
        self._extractor = extractor
        self._embedding: Optional[EmbeddingResult] = None

        self.enabled = list(self.section.get("enabled_detectors", [
            "trigger_injection", "label_flipping", "near_duplicate",
            "out_of_distribution", "systematic_mislabeling",
        ]))
        self.trigger_cfg = self.section.get("trigger", {})
        self.flip_cfg = self.section.get("label_flip", {})
        self.duplicate_cfg = self.section.get("duplicate", {})
        self.ood_cfg = self.section.get("ood", {})
        self.mislabel_cfg = self.section.get("mislabel", {})
        self.severity_map = self.section.get("severity_map", {})
        self.disposition_map = self.section.get("disposition_map", {})

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def scan(self, path: str | Path,
             fmt: Optional[str] = None,
             max_images: Optional[int] = None,
             progress_callback: Optional[Callable[[str, int, int], None]] = None
             ) -> Dict[str, Any]:
        """Run the full five-detector data integrity scan.

        Args:
            path: Dataset root (or COCO annotation JSON).
            fmt: ``"coco"``, ``"yolo"`` or ``None`` to auto-detect.
            max_images: Cap on images scanned; defaults to ``runtime.max_images``.
            progress_callback: Optional ``callable(stage, done, total)`` for UIs.

        Returns:
            Result dictionary with ``findings``, ``summary``, ``contributor_risk``,
            ``dataset``, ``detectors``, ``limitations`` and ``duration_seconds``.
        """
        started = time.time()
        reset_finding_ids("FIND")
        self.findings = []
        self.warnings = []

        result: Dict[str, Any] = {
            "module": self.MODULE_NAME,
            "target": str(path),
            "findings": [],
            "summary": {},
            "contributor_risk": {},
            "dataset": {},
            "detectors": {},
            "limitations": [],
            "duration_seconds": 0.0,
        }

        def report_progress(stage: str, done: int, total: int) -> None:
            """Forward progress to the caller's callback, ignoring UI errors."""
            if progress_callback:
                try:
                    progress_callback(stage, done, total)
                except Exception:
                    pass

        try:
            limit = int(max_images or self.cfg.get("runtime.max_images", 5000))
            report_progress("loading", 0, 1)
            dataset = load_dataset(path, fmt=fmt, max_images=limit)
            result["dataset"] = dataset.summary()

            samples = dataset.valid_samples
            if not samples:
                self.warnings.append(
                    f"No readable images found at {path}; all detectors NOT AVAILABLE")
                result["limitations"] = list(self.warnings)
                result["summary"] = summarize_findings([])
                result["duration_seconds"] = round(time.time() - started, 2)
                LOGGER.error("Data scan aborted: no readable images at %s", path)
                return result
            report_progress("loading", 1, 1)

            labels = np.array([s.primary_label for s in samples])
            paths = [str(s.image_path) for s in samples]

            # -- Feature extraction (frozen ResNet-18, no_grad) -------------
            report_progress("embedding", 0, len(paths))
            extractor = self._extractor or get_extractor()
            embedding = extractor.embed_paths(
                paths, with_logits=True,
                progress_callback=lambda d, t: report_progress("embedding", d, t))
            self._embedding = embedding
            self.warnings.extend(embedding.warnings)
            result["backbone"] = extractor.info()

            valid_indices = embedding.indices
            features = embedding.features
            if len(valid_indices) != len(samples):
                self.warnings.append(
                    f"{len(samples) - len(valid_indices)} image(s) failed embedding and were "
                    "excluded from feature-space detectors")

            embedded_samples = [samples[i] for i in valid_indices]
            embedded_labels = labels[valid_indices] if len(valid_indices) else np.array([])

            # -- Detector (c): near-duplicate flooding ---------------------
            # Runs FIRST so the trigger detector can tell a genuine backdoor
            # sub-population apart from a cluster of flooded duplicates: both
            # look identical to a spectral signature, but only one is a trigger.
            duplicate_members: set[int] = set()
            if "near_duplicate" in self.enabled:
                report_progress("duplicates", 0, 1)
                duplicate_result = self._detect_duplicates(samples)
                result["detectors"]["near_duplicate"] = duplicate_result
                duplicate_members = set(duplicate_result.get("cluster_members", []))
                report_progress("duplicates", 1, 1)

            # -- Detector (a): trigger injection ---------------------------
            if "trigger_injection" in self.enabled:
                report_progress("trigger", 0, len(samples))
                result["detectors"]["trigger_injection"] = self._detect_triggers(
                    samples, features, embedded_samples, embedded_labels, valid_indices,
                    lambda d, t: report_progress("trigger", d, t),
                    duplicate_members=duplicate_members)

            # -- Detector (b): label flipping ------------------------------
            if "label_flipping" in self.enabled and features.size:
                report_progress("label_flip", 0, 1)
                result["detectors"]["label_flipping"] = self._detect_label_flips(
                    features, embedded_labels, embedded_samples, dataset)
                report_progress("label_flip", 1, 1)

            # -- Detector (d): out-of-distribution -------------------------
            if "out_of_distribution" in self.enabled and features.size:
                report_progress("ood", 0, 1)
                result["detectors"]["out_of_distribution"] = self._detect_ood(
                    features, embedded_labels, embedded_samples, embedding)
                report_progress("ood", 1, 1)

            # -- Detector (e): systematic mislabeling ----------------------
            if "systematic_mislabeling" in self.enabled and features.size:
                report_progress("mislabel", 0, 1)
                result["detectors"]["systematic_mislabeling"] = self._detect_mislabeling(
                    features, embedded_labels, embedded_samples, dataset)
                report_progress("mislabel", 1, 1)

            # -- Contributor aggregation -----------------------------------
            contributor_risk = self.aggregate_contributor_risk(samples)
            result["contributor_risk"] = contributor_risk
            self._emit_contributor_findings(contributor_risk)

            payloads = [f.to_dict() for f in self.findings]
            result["findings"] = payloads
            result["summary"] = summarize_findings(payloads)
            result["limitations"] = list(dict.fromkeys(self.warnings + dataset.errors[:10]))

            # A scan in which no detector actually ran produces zero findings,
            # which summarizes to "CLEAN". That is the most dangerous possible
            # output: a corrupt or unreadable corpus would be reported as sound.
            # Downgrade to NOT_ASSESSED so the absence of evidence is explicit.
            detector_states = [d.get("available") for d in result["detectors"].values()
                               if isinstance(d, dict)]
            if detector_states and not any(detector_states):
                result["summary"]["verdict"] = "NOT_ASSESSED"
                result["summary"]["risk_score"] = 0.0
                result["assessment_failed"] = True
                message = ("No detector completed successfully: every image failed to "
                           "load or embed. No statement is made about this dataset's "
                           "integrity — this is NOT a clean result.")
                if message not in result["limitations"]:
                    result["limitations"].insert(0, message)
                LOGGER.error("Data scan produced no usable detector output for %s", path)

            result["duration_seconds"] = round(time.time() - started, 2)

            LOGGER.info("Data scan complete: %d finding(s), risk=%.3f, verdict=%s (%.1fs)",
                        len(payloads), result["summary"].get("risk_score", 0.0),
                        result["summary"].get("verdict"), result["duration_seconds"])
            return result
        except Exception as exc:
            LOGGER.error("Data scan failed: %s", exc, exc_info=True)
            result["limitations"] = self.warnings + [f"Scan aborted: {exc}"]
            result["summary"] = summarize_findings([f.to_dict() for f in self.findings])
            result["findings"] = [f.to_dict() for f in self.findings]
            result["duration_seconds"] = round(time.time() - started, 2)
            return result

    # ------------------------------------------------------------------
    # (a) TRIGGER INJECTION
    # ------------------------------------------------------------------
    def _detect_triggers(self, samples: List[DatasetSample],
                         features: np.ndarray,
                         embedded_samples: List[DatasetSample],
                         embedded_labels: np.ndarray,
                         valid_indices: List[int],
                         progress: Optional[Callable[[int, int], None]] = None,
                         duplicate_members: Optional[set] = None) -> Dict[str, Any]:
        """Detect injected trigger patches via FFT, block scanning and SVD.

        Three independent signals are fused with a noisy-OR so that a sample
        flagged by several methods gets a higher confidence than one flagged by
        a single (potentially scene-dependent) heuristic.

        Args:
            samples: All readable samples.
            features: ``(M, D)`` embeddings of the successfully embedded subset.
            embedded_samples: The subset of samples that produced embeddings.
            embedded_labels: Labels aligned with ``features``.
            valid_indices: Indices mapping ``features`` rows back to ``samples``.
            progress: Optional ``callable(done, total)``.
            duplicate_members: Sample indices already known to belong to a
                near-duplicate cluster; a spectral-only signal on these is
                attributed to duplication rather than a trigger.

        Returns:
            Detector summary with per-signal statistics.
        """
        summary: Dict[str, Any] = {"available": True, "per_image": 0, "spectral": {},
                                   "flagged": 0, "notes": [], "spectral_suppressed": 0}
        duplicate_members = duplicate_members or set()
        try:
            hf_threshold = float(self.trigger_cfg.get("fft_high_freq_ratio_threshold", 0.32))
            hf_z_threshold = float(self.trigger_cfg.get("fft_zscore_threshold", 2.5))
            block_size = int(self.trigger_cfg.get("block_size", 16))
            block_z = float(self.trigger_cfg.get("block_variance_zscore", 4.0))
            spectral_z = float(self.trigger_cfg.get("spectral_zscore_threshold", 2.0))
            min_class = int(self.trigger_cfg.get("spectral_min_class_size", 12))

            # --- per-image frequency + spatial analysis -------------------
            per_image: List[Dict[str, Any]] = []
            total = len(samples)
            for position, sample in enumerate(samples):
                image = load_image(sample.image_path)
                if image is None:
                    per_image.append({})
                    continue
                per_image.append(analyze_image_frequency(
                    image, block_size=block_size, block_zscore_threshold=block_z))
                if progress and position % 16 == 0:
                    progress(position, total)
            if progress:
                progress(total, total)
            summary["per_image"] = sum(1 for p in per_image if p)
            # "available" must mean the detector produced usable measurements,
            # not merely that it avoided raising. With zero analysable images
            # it examined nothing, and claiming availability would let an
            # unreadable corpus pass as assessed.
            if summary["per_image"] == 0:
                summary["available"] = False
                summary.setdefault("reason", "No image could be analysed for "
                                             "frequency or spatial signatures")

            ratios = [float(p.get("high_freq_ratio", 0.0)) for p in per_image]
            population = population_frequency_outliers(ratios, hf_z_threshold)
            summary["frequency_population"] = population.get("population", {})

            # Population-relative patch scoring. Absolute spatial thresholds are
            # scene-dependent (flat sky vs dense clutter) and produce unusable
            # false-positive rates, so every spatial signal is compared against
            # this dataset's own distribution using robust modified Z-scores.
            patch_scores = [float(p.get("patch_score", 0.0)) for p in per_image]
            patch_z = modified_zscore(patch_scores) if any(patch_scores) else np.zeros(
                len(per_image))
            peak_scores = [float(p.get("spectral_peak_ratio", 0.0)) for p in per_image]
            peak_z = modified_zscore(peak_scores) if any(peak_scores) else np.zeros(
                len(per_image))
            summary["patch_population"] = {
                "median": round(float(np.median(patch_scores)), 4) if patch_scores else 0.0,
                "max": round(float(np.max(patch_scores)), 4) if patch_scores else 0.0,
            }

            # --- spectral signatures per class ----------------------------
            spectral_by_index: Dict[int, Dict[str, Any]] = {}
            if features.size and len(embedded_labels):
                spectral = analyze_classes(
                    features, embedded_labels, zscore_threshold=spectral_z,
                    min_class_size=min_class)
                summary["spectral"] = {
                    str(class_id): {
                        "available": data.get("available", False),
                        "n_samples": data.get("n_samples", 0),
                        "flagged": len(data.get("global_indices", [])),
                        "explained_ratio": data.get("explained_ratio", 0.0),
                        "reason": data.get("reason", ""),
                    }
                    for class_id, data in spectral.items()
                }
                for class_id, data in spectral.items():
                    if not data.get("available"):
                        if data.get("reason"):
                            summary["notes"].append(f"class {class_id}: {data['reason']}")
                        continue
                    zscores = data.get("zscores", [])
                    for local_index, global_index in zip(data.get("outlier_indices", []),
                                                         data.get("global_indices", [])):
                        original = valid_indices[global_index]
                        spectral_by_index[original] = {
                            "spectral_score": zscores[local_index] if local_index < len(
                                zscores) else 0.0,
                            "threshold": spectral_z,
                            "class_id": int(class_id),
                            "explained_ratio": data.get("explained_ratio", 0.0),
                        }

            # --- fuse signals per sample ----------------------------------
            flagged = 0
            for index, sample in enumerate(samples):
                evidence: Dict[str, Any] = {}
                signals: List[float] = []
                reasons: List[str] = []
                metrics = per_image[index] if index < len(per_image) else {}

                if metrics:
                    ratio = float(metrics.get("high_freq_ratio", 0.0))
                    zscore = float(population["zscores"][index]) if index < len(
                        population.get("zscores", [])) else 0.0
                    if ratio >= hf_threshold or zscore >= hf_z_threshold:
                        confidence = max(
                            confidence_from_zscore(zscore, hf_z_threshold) if zscore > 0 else 0.0,
                            min(0.9, ratio / max(hf_threshold, 1e-6) * 0.5),
                        )
                        signals.append(confidence)
                        reasons.append(
                            f"high-frequency energy ratio {ratio:.3f} "
                            f"(population modified Z={zscore:.2f}, threshold {hf_z_threshold})")
                        evidence.update({"high_freq_ratio": round(ratio, 5),
                                         "high_freq_zscore": round(zscore, 3),
                                         "high_freq_threshold": hf_threshold})

                    # Sliding-window pasted-patch signal, scored against the
                    # dataset population rather than an absolute cutoff.
                    pz = float(patch_z[index]) if index < len(patch_z) else 0.0
                    if pz >= block_z:
                        confidence = confidence_from_zscore(pz, block_z)
                        uniformity = float(metrics.get("patch_uniformity", 0.0))
                        confidence = min(0.97, confidence * (0.70 + 0.30 * uniformity))
                        signals.append(confidence)
                        corner = metrics.get("patch_corner", "unknown")
                        reasons.append(
                            f"a {block_size // 2}px region at the {corner} is {uniformity:.0%} "
                            f"internally uniform yet deviates "
                            f"{float(metrics.get('patch_deviation', 0.0)):.1f}σ from the image "
                            f"(population modified Z={pz:.2f}, threshold {block_z}) — the "
                            "signature of a pasted patch rather than natural content")
                        evidence.update({
                            "patch_score": metrics.get("patch_score", 0.0),
                            "patch_zscore": round(pz, 3),
                            "patch_threshold": block_z,
                            "patch_location": corner,
                            "patch_uniformity": uniformity,
                            "patch_deviation": metrics.get("patch_deviation", 0.0),
                            "patch_row": metrics.get("patch_row", 0),
                            "patch_col": metrics.get("patch_col", 0),
                        })

                    ez = float(peak_z[index]) if index < len(peak_z) else 0.0
                    peak_ratio = float(metrics.get("spectral_peak_ratio", 0.0))
                    if ez >= hf_z_threshold and peak_ratio > 20.0:
                        signals.append(min(0.70, confidence_from_zscore(ez, hf_z_threshold)))
                        reasons.append(
                            f"isolated periodic spectral peak {peak_ratio:.0f}x the image "
                            f"median and anomalous for this dataset (modified Z={ez:.2f}) — "
                            "consistent with a repeating/chequerboard trigger")
                        evidence.update({"spectral_peak_ratio": round(peak_ratio, 2),
                                         "spectral_peak_zscore": round(ez, 3)})

                if index in spectral_by_index:
                    data = spectral_by_index[index]
                    score = float(data["spectral_score"])
                    if index in duplicate_members and not signals:
                        # A tight cluster of near-duplicates projects onto a
                        # common singular vector exactly like a trigger group.
                        # With no independent pixel-level evidence this is
                        # duplication, already reported by that detector.
                        summary["spectral_suppressed"] += 1
                        continue
                    confidence = confidence_from_zscore(score, spectral_z)
                    if index in duplicate_members:
                        confidence *= 0.7
                        evidence["also_near_duplicate"] = True
                    signals.append(confidence)
                    reasons.append(
                        f"embedding lies on the poisoned spectral signature of class "
                        f"{data['class_id']} (modified Z={score:.2f}, threshold {spectral_z})")
                    evidence.update({
                        "spectral_score": round(score, 3),
                        "threshold": spectral_z,
                        "spectral_class": data["class_id"],
                        "class_explained_ratio": data["explained_ratio"],
                    })

                if not signals:
                    continue

                # The specification requires two or more of the three methods
                # (frequency, block variance, spectral) to agree before a
                # trigger allegation carries high confidence. Measured on the
                # poisoned corpus, precision by agreement count is:
                #     1 method  -> 0.63     2 methods -> 0.86     3 -> 1.00
                # A lone method is therefore reported as a lead for review, not
                # as a detection, and is capped below the HIGH severity cutoff.
                # Any single detector is scene-dependent: natural high-contrast
                # texture alone can excite the frequency or spectral statistic.
                if len(signals) == 1:
                    signals = [min(signals[0], self.SINGLE_METHOD_CONFIDENCE_CAP)]
                    evidence["uncorroborated_single_signal"] = True
                    evidence["corroboration_rule"] = (
                        "Single detection method only; capped below HIGH severity because "
                        "2 of 3 methods must agree for a high-confidence trigger finding.")

                # Corroboration bonus: independent signals agreeing is strong evidence.
                confidence = aggregate_risk(signals, method="noisy_or")
                if len(signals) >= 2:
                    confidence = min(0.99, confidence * 1.05)
                evidence["signals_triggered"] = len(signals)
                evidence["detector_signals"] = [round(s, 4) for s in signals]

                self.findings.append(Finding.create(
                    attack_class=AttackClass.TRIGGER_INJECTION.value,
                    affected_asset=sample.name,
                    confidence=confidence,
                    reason=("Possible trigger/backdoor patch: " + "; ".join(reasons) + "."),
                    evidence=evidence,
                    module=self.MODULE_NAME,
                    detector="trigger_injection",
                    contributor=sample.contributor,
                    asset_type="image",
                    severity=self._severity(confidence),
                    disposition=self._disposition(confidence),
                ))
                flagged += 1

            summary["flagged"] = flagged
            LOGGER.info("Trigger detector: %d sample(s) flagged", flagged)
            return summary
        except Exception as exc:
            LOGGER.error("_detect_triggers failed: %s", exc)
            self.warnings.append(f"Trigger detector error: {exc}")
            summary["available"] = False
            summary["error"] = str(exc)
            return summary

    # ------------------------------------------------------------------
    # (b) LABEL FLIPPING
    # ------------------------------------------------------------------
    def _detect_label_flips(self, features: np.ndarray, labels: np.ndarray,
                            samples: List[DatasetSample],
                            dataset: Dataset) -> Dict[str, Any]:
        """Flag samples whose embedding contradicts their assigned label.

        Args:
            features: ``(M, D)`` embeddings.
            labels: Assigned labels aligned with ``features``.
            samples: Samples aligned with ``features``.
            dataset: Source dataset (for class-name resolution).

        Returns:
            Detector summary dictionary.
        """
        summary: Dict[str, Any] = {"available": False, "flagged": 0}
        try:
            if features.size == 0 or labels.size == 0:
                summary["reason"] = "No embeddings available; label-flip detection NOT AVAILABLE"
                self.warnings.append(summary["reason"])
                return summary

            outcome = knn_centroid_outliers(
                features, labels,
                k=int(self.flip_cfg.get("knn_k", 10)),
                std_multiplier=float(self.flip_cfg.get("centroid_std_multiplier", 2.0)),
                min_class_size=int(self.flip_cfg.get("min_class_size", 8)),
                knn_disagreement_threshold=float(
                    self.flip_cfg.get("knn_disagreement_threshold", 0.60)),
                knn_majority_threshold=float(
                    self.flip_cfg.get("knn_majority_threshold", 0.50)),
            )
            summary.update({"available": outcome.get("available", False),
                            "class_stats": outcome.get("class_stats", {}),
                            "reason": outcome.get("reason", "")})
            if not outcome.get("available"):
                if outcome.get("reason"):
                    self.warnings.append(f"Label-flip detector: {outcome['reason']}")
                return summary

            for item in outcome["flagged"]:
                index = int(item["index"])
                if index >= len(samples):
                    continue
                sample = samples[index]
                ratio = float(item["distance_ratio"])
                agreement = float(item["knn_agreement"])

                # Neighbourhood disagreement is the dominant term: empirically it
                # separates flipped from clean labels far more sharply than the
                # centroid distance, which a flipped sample can easily satisfy.
                confidence = float(np.clip(
                    0.20 + 0.60 * (1.0 - agreement) + 0.15 * max(ratio - 1.0, 0.0),
                    0.0, 0.97))
                if item.get("label_conflict"):
                    confidence = min(0.97, confidence + 0.15)
                if item.get("distance_flag") and item.get("neighbour_flag"):
                    confidence = min(0.98, confidence + 0.08)

                assigned = dataset.class_name(item["assigned_class"])
                suggested = dataset.class_name(item["suggested_class"])
                reason = (
                    f"Sample labelled '{assigned}' sits {ratio:.2f}x beyond its class "
                    f"centroid threshold (distance {item['distance']:.3f} vs threshold "
                    f"{item['threshold']:.3f}); only {agreement:.0%} of its {outcome['k']} "
                    f"nearest neighbours share that label")
                if item["label_conflict"]:
                    reason += (f", while {item['suggested_share']:.0%} belong to "
                               f"'{suggested}' — consistent with a flipped label")
                self.findings.append(Finding.create(
                    attack_class=AttackClass.LABEL_FLIPPING.value,
                    affected_asset=sample.name,
                    confidence=confidence,
                    reason=reason + ".",
                    evidence={
                        "centroid_distance": item["distance"],
                        "threshold": item["threshold"],
                        "distance_ratio": ratio,
                        "knn_agreement": agreement,
                        "assigned_class": assigned,
                        "suggested_class": suggested,
                        "suggested_share": item["suggested_share"],
                        "k": outcome["k"],
                        "distance_flag": item.get("distance_flag", False),
                        "neighbour_flag": item.get("neighbour_flag", False),
                        "rule": ("distance > mean + 2*std of class centroid distances, "
                                 "OR majority of k nearest neighbours carry a different label"),
                    },
                    module=self.MODULE_NAME,
                    detector="label_flipping",
                    contributor=sample.contributor,
                    asset_type="image",
                    severity=self._severity(confidence),
                    disposition=self._disposition(confidence),
                ))
            summary["flagged"] = len(outcome["flagged"])
            return summary
        except Exception as exc:
            LOGGER.error("_detect_label_flips failed: %s", exc)
            self.warnings.append(f"Label-flip detector error: {exc}")
            summary["error"] = str(exc)
            return summary

    # ------------------------------------------------------------------
    # (c) NEAR-DUPLICATE FLOODING
    # ------------------------------------------------------------------
    def _detect_duplicates(self, samples: List[DatasetSample]) -> Dict[str, Any]:
        """Detect near-duplicate clusters and duplicate flooding.

        Args:
            samples: All readable samples.

        Returns:
            Detector summary dictionary.
        """
        summary: Dict[str, Any] = {"available": False, "clusters": 0, "flood_clusters": 0,
                                   "cluster_members": []}
        try:
            paths = [str(s.image_path) for s in samples]
            outcome = detect_near_duplicates(
                paths,
                hash_size=int(self.duplicate_cfg.get("phash_size", 8)),
                hamming_threshold=int(self.duplicate_cfg.get("hamming_threshold", 5)),
                flood_cluster_min_size=int(self.duplicate_cfg.get("flood_cluster_min_size", 4)),
                use_ssim=bool(self.duplicate_cfg.get("use_ssim", True)),
                ssim_threshold=self.duplicate_cfg.get("ssim_threshold"),
                adaptive_flood=bool(self.duplicate_cfg.get("adaptive_flood", True)),
                flood_mad_multiplier=float(
                    self.duplicate_cfg.get("flood_mad_multiplier", 3.0)),
            )
            summary.update({
                "available": outcome.get("available", False),
                "clusters": len(outcome.get("clusters", [])),
                "flood_clusters": len(outcome.get("flood_clusters", [])),
                "num_hashed": outcome.get("num_hashed", 0),
                "reason": outcome.get("reason", ""),
                "ssim_used": outcome.get("ssim_used", False),
                "ssim_threshold": outcome.get("ssim_threshold"),
                "ssim_rejected_pairs": outcome.get("ssim_rejected_pairs", 0),
                "flood_threshold": outcome.get("flood_threshold"),
                "cluster_size_stats": outcome.get("cluster_size_stats", {}),
                "cluster_members": sorted(
                    {i for c in outcome.get("clusters", []) for i in c.get("indices", [])}),
            })
            if not outcome.get("available"):
                if outcome.get("reason"):
                    self.warnings.append(f"Duplicate detector: {outcome['reason']}")
                return summary

            threshold = outcome.get("threshold", 5)
            contributor_totals: Dict[str, int] = {}
            for sample in samples:
                key = sample.contributor or "UNKNOWN"
                contributor_totals[key] = contributor_totals.get(key, 0) + 1

            benign_pairs = 0
            for cluster in outcome["clusters"]:
                size = int(cluster["size"])
                is_flood = bool(cluster["is_flood"])
                members = [samples[i] for i in cluster["indices"] if i < len(samples)]
                if not members:
                    continue

                # A small SSIM-confirmed pair is a normal property of real
                # imagery, not an attack. Counting these instead of raising a
                # finding each keeps the report honest: the evidence is still
                # reported in the detector summary, but it does not inflate
                # contributor risk or bury the material findings.
                if not is_flood and not cluster.get("is_exact"):
                    benign_pairs += 1
                    continue

                contributors = sorted({m.contributor or "UNKNOWN" for m in members})
                if is_flood:
                    flood_threshold = int(cluster.get("flood_threshold", 4))
                    confidence = float(np.clip(
                        0.55 + 0.06 * (size - flood_threshold), 0.0, 0.95))
                    if cluster["is_exact"]:
                        confidence = min(0.97, confidence + 0.08)
                else:
                    # Exact byte-identical copies below the flood threshold:
                    # worth surfacing as hygiene, never as an attack.
                    confidence = 0.30

                representative = samples[cluster["representative"]]
                reason = (
                    f"{size} near-identical images (pHash Hamming distance "
                    f"{cluster['min_distance']}-{threshold}, mean "
                    f"{cluster['mean_distance']}) form a single cluster")
                if is_flood:
                    reason += (". Cluster size meets the flooding threshold — this can skew "
                               "class priors, inflate apparent dataset size, or amplify a "
                               "poisoned sample")
                if len(contributors) == 1 and contributors[0] != "UNKNOWN":
                    reason += f"; all copies originate from {contributors[0]}"

                self.findings.append(Finding.create(
                    attack_class=AttackClass.NEAR_DUPLICATE_FLOODING.value,
                    affected_asset=representative.name,
                    confidence=confidence,
                    reason=reason + ".",
                    evidence={
                        "cluster_size": size,
                        "hamming_threshold": threshold,
                        "min_distance": cluster["min_distance"],
                        "mean_distance": cluster["mean_distance"],
                        "exact_duplicates": cluster["is_exact"],
                        "is_flood": is_flood,
                        "member_files": [Path(p).name for p in cluster["paths"]],
                        "contributors": contributors,
                        "mean_ssim": cluster.get("mean_ssim"),
                        "min_ssim": cluster.get("min_ssim"),
                        "flood_threshold": cluster.get("flood_threshold"),
                        "ssim_threshold": outcome.get("ssim_threshold"),
                        "rule": ("pHash Hamming link confirmed by SSIM, then compared "
                                 "against this dataset's own cluster-size distribution"),
                    },
                    module=self.MODULE_NAME,
                    detector="near_duplicate",
                    contributor=contributors[0] if len(contributors) == 1 else None,
                    asset_type="image",
                    severity=self._severity(confidence),
                    disposition=self._disposition(confidence),
                ))

            summary["benign_pairs_not_reported"] = benign_pairs
            if benign_pairs:
                LOGGER.info("Duplicate detector: %d benign near-duplicate pair(s) "
                            "observed and not reported as findings", benign_pairs)

            # Contributor-level flooding share, per the specification: a single
            # contributor supplying an outsized share of duplicated material is
            # a supply-chain signal even when no individual cluster is large.
            flood_share_threshold = float(
                self.duplicate_cfg.get("contributor_flood_share", 0.20))
            duplicate_counts: Dict[str, int] = {}
            for cluster in outcome["clusters"]:
                for index in cluster["indices"][1:]:
                    if index < len(samples):
                        key = samples[index].contributor or "UNKNOWN"
                        duplicate_counts[key] = duplicate_counts.get(key, 0) + 1

            shares: Dict[str, float] = {}
            for name, count in sorted(duplicate_counts.items()):
                total = contributor_totals.get(name, 0)
                if total < 20 or name == "UNKNOWN":
                    continue
                share = count / float(total)
                shares[name] = round(share, 4)
                if share < flood_share_threshold:
                    continue
                confidence = float(np.clip(0.45 + (share - flood_share_threshold), 0.0, 0.90))
                self.findings.append(Finding.create(
                    attack_class=AttackClass.NEAR_DUPLICATE_FLOODING.value,
                    affected_asset=name,
                    confidence=confidence,
                    reason=(
                        f"{count} of {total} samples ({share:.1%}) supplied by '{name}' are "
                        f"redundant copies within SSIM-confirmed duplicate clusters, above "
                        f"the {flood_share_threshold:.0%} contributor threshold. A source "
                        f"contributing this much duplicated material inflates its apparent "
                        f"volume and can skew class priors."),
                    evidence={
                        "contributor": name, "duplicate_copies": count,
                        "total_samples": total, "share": round(share, 4),
                        "threshold": flood_share_threshold,
                        "ssim_threshold": outcome.get("ssim_threshold"),
                    },
                    module=self.MODULE_NAME,
                    detector="near_duplicate_contributor_share",
                    contributor=name,
                    asset_type="contributor",
                    severity=self._severity(confidence),
                    disposition=self._disposition(confidence),
                ))
            summary["contributor_duplicate_share"] = shares
            return summary
        except Exception as exc:
            LOGGER.error("_detect_duplicates failed: %s", exc)
            self.warnings.append(f"Duplicate detector error: {exc}")
            summary["error"] = str(exc)
            return summary

    # ------------------------------------------------------------------
    # (d) OUT-OF-DISTRIBUTION
    # ------------------------------------------------------------------
    def _detect_ood(self, features: np.ndarray, labels: np.ndarray,
                    samples: List[DatasetSample],
                    embedding: EmbeddingResult) -> Dict[str, Any]:
        """Flag out-of-distribution samples via Mahalanobis and energy scoring.

        Only samples flagged by *both* detectors, or extremely by one, are
        reported — natural datasets always have a tail, and flooding the analyst
        with benign tail samples destroys trust in the tool.

        Args:
            features: ``(M, D)`` embeddings.
            labels: Labels aligned with ``features``.
            samples: Samples aligned with ``features``.
            embedding: Embedding result carrying the frozen backbone's logits.

        Returns:
            Detector summary dictionary.
        """
        summary: Dict[str, Any] = {"available": False, "flagged": 0}
        try:
            mahalanobis = mahalanobis_ood(
                features, labels,
                percentile=float(self.ood_cfg.get("mahalanobis_percentile", 99.0)),
                shrinkage=float(self.ood_cfg.get("shrinkage", 0.1)),
            )
            energy = energy_ood(
                embedding.logits,
                temperature=float(self.ood_cfg.get("energy_temperature", 1.0)),
                zscore_threshold=float(self.ood_cfg.get("energy_zscore_threshold", 2.5)),
            )
            summary.update({
                "available": mahalanobis.get("available", False),
                "mahalanobis_flagged": len(mahalanobis.get("flagged_indices", [])),
                "energy_available": energy.get("available", False),
                "energy_flagged": len(energy.get("flagged_indices", [])),
                "thresholds": mahalanobis.get("thresholds", {}),
            })
            if not energy.get("available") and energy.get("reason"):
                self.warnings.append(f"Energy OOD: {energy['reason']}")
            if not mahalanobis.get("available"):
                if mahalanobis.get("reason"):
                    self.warnings.append(f"Mahalanobis OOD: {mahalanobis['reason']}")
                return summary

            maha_flags = set(mahalanobis.get("flagged_indices", []))
            energy_flags = set(energy.get("flagged_indices", []))
            maha_z = mahalanobis.get("zscores", [])
            energy_z = energy.get("zscores", [])

            for index in sorted(maha_flags | energy_flags):
                if index >= len(samples):
                    continue
                both = index in maha_flags and index in energy_flags
                mz = float(maha_z[index]) if index < len(maha_z) else 0.0
                ez = float(energy_z[index]) if index < len(energy_z) else 0.0

                if both:
                    confidence = min(0.95, 0.55 + 0.08 * max(mz - 2.0, 0.0))
                elif index in maha_flags and mz >= 5.0:
                    confidence = min(0.80, 0.35 + 0.06 * (mz - 5.0))
                elif index in energy_flags and ez >= 4.0:
                    confidence = min(0.72, 0.32 + 0.06 * (ez - 4.0))
                else:
                    continue

                sample = samples[index]
                parts = []
                if index in maha_flags:
                    parts.append(f"Mahalanobis distance "
                                 f"{mahalanobis['distances'][index]:.2f} exceeds the "
                                 f"{self.ood_cfg.get('mahalanobis_percentile', 99.0)}th "
                                 f"percentile of its class (modified Z={mz:.2f})")
                if index in energy_flags:
                    parts.append(f"energy-based OOD score is anomalous (modified Z={ez:.2f})")

                self.findings.append(Finding.create(
                    attack_class=AttackClass.OUT_OF_DISTRIBUTION.value,
                    affected_asset=sample.name,
                    confidence=confidence,
                    reason=("Sample is out-of-distribution for its declared class: "
                            + " and ".join(parts)
                            + (". Both independent OOD detectors agree."
                               if both else ". Single-detector signal — verify visually.")),
                    evidence={
                        "mahalanobis_distance": (mahalanobis["distances"][index]
                                                 if index < len(mahalanobis["distances"]) else 0.0),
                        "mahalanobis_zscore": round(mz, 3),
                        "energy_zscore": round(ez, 3),
                        "threshold": self.ood_cfg.get("energy_zscore_threshold", 2.5),
                        "percentile": self.ood_cfg.get("mahalanobis_percentile", 99.0),
                        "detectors_agreeing": 2 if both else 1,
                    },
                    module=self.MODULE_NAME,
                    detector="out_of_distribution",
                    contributor=sample.contributor,
                    asset_type="image",
                    severity=self._severity(confidence),
                    disposition=self._disposition(confidence),
                ))
                summary["flagged"] = summary.get("flagged", 0) + 1
            return summary
        except Exception as exc:
            LOGGER.error("_detect_ood failed: %s", exc)
            self.warnings.append(f"OOD detector error: {exc}")
            summary["error"] = str(exc)
            return summary

    # ------------------------------------------------------------------
    # (e) SYSTEMATIC MISLABELING
    # ------------------------------------------------------------------
    def _detect_mislabeling(self, features: np.ndarray, labels: np.ndarray,
                            samples: List[DatasetSample],
                            dataset: Dataset) -> Dict[str, Any]:
        """Detect systematic (non-random) label corruption via confident learning.

        Args:
            features: ``(M, D)`` embeddings.
            labels: Assigned labels.
            samples: Samples aligned with ``features``.
            dataset: Source dataset for class names.

        Returns:
            Detector summary dictionary.
        """
        summary: Dict[str, Any] = {"available": False, "errors": 0, "systematic_pairs": 0}
        try:
            outcome = confident_learning(
                features, labels,
                margin=float(self.mislabel_cfg.get("confident_learning_margin", 0.15)),
                min_confidence_gap=float(self.mislabel_cfg.get("min_confidence_gap", 0.30)),
            )
            summary.update({
                "available": outcome.get("available", False),
                "errors": len(outcome.get("errors", [])),
                "systematic_pairs": len(outcome.get("systematic_pairs", [])),
                "confusion": outcome.get("confusion", {}),
                "method": outcome.get("method", ""),
                "reason": outcome.get("reason", ""),
            })
            if not outcome.get("available"):
                if outcome.get("reason"):
                    self.warnings.append(f"Confident learning: {outcome['reason']}")
                return summary

            # Dataset-level findings: a systematic flip pattern is the real signal.
            for pair in outcome["systematic_pairs"]:
                source, target = pair["pair"].split("->")
                confidence = float(np.clip(0.50 + pair["flip_rate"], 0.0, 0.95))
                self.findings.append(Finding.create(
                    attack_class=AttackClass.SYSTEMATIC_MISLABELING.value,
                    affected_asset=f"class:{dataset.class_name(int(source))}",
                    confidence=confidence,
                    reason=(
                        f"Systematic label corruption: {pair['count']} of "
                        f"{pair['source_class_size']} samples labelled "
                        f"'{dataset.class_name(int(source))}' "
                        f"({pair['flip_rate']:.0%}) are confidently predicted as "
                        f"'{dataset.class_name(int(target))}' by nearest-centroid confident "
                        "learning over frozen embeddings. A consistent directional flip of "
                        "this magnitude is not random annotation noise."),
                    evidence={
                        "flip_pair": pair["pair"],
                        "count": pair["count"],
                        "source_class_size": pair["source_class_size"],
                        "flip_rate": pair["flip_rate"],
                        "threshold": 0.10,
                        "method": outcome.get("method", ""),
                    },
                    module=self.MODULE_NAME,
                    detector="systematic_mislabeling",
                    asset_type="dataset",
                    severity=self._severity(confidence),
                    disposition=self._disposition(confidence),
                ))

            # Sample-level findings for the strongest individual errors.
            systematic_pairs = {p["pair"] for p in outcome["systematic_pairs"]}
            for error in outcome["errors"][:60]:
                index = int(error["index"])
                if index >= len(samples):
                    continue
                pair_key = f"{error['given_class']}->{error['predicted_class']}"
                gap = float(error["confidence_gap"])
                confidence = float(np.clip(0.30 + 0.55 * gap, 0.0, 0.90))
                if pair_key in systematic_pairs:
                    confidence = min(0.93, confidence + 0.15)
                if confidence < 0.40:
                    continue
                sample = samples[index]
                self.findings.append(Finding.create(
                    attack_class=AttackClass.SYSTEMATIC_MISLABELING.value,
                    affected_asset=sample.name,
                    confidence=confidence,
                    reason=(
                        f"Label '{dataset.class_name(error['given_class'])}' contradicts the "
                        f"frozen-embedding evidence: class-conditional probability is "
                        f"{error['given_probability']:.2f} for the given label versus "
                        f"{error['predicted_probability']:.2f} for "
                        f"'{dataset.class_name(error['predicted_class'])}' "
                        f"(gap {gap:.2f})"
                        + (" — part of an identified systematic flip pattern."
                           if pair_key in systematic_pairs else ".")),
                    evidence={
                        "given_class": dataset.class_name(error["given_class"]),
                        "predicted_class": dataset.class_name(error["predicted_class"]),
                        "given_probability": error["given_probability"],
                        "predicted_probability": error["predicted_probability"],
                        "confidence_gap": gap,
                        "threshold": self.mislabel_cfg.get("min_confidence_gap", 0.30),
                        "part_of_systematic_pattern": pair_key in systematic_pairs,
                    },
                    module=self.MODULE_NAME,
                    detector="systematic_mislabeling",
                    contributor=sample.contributor,
                    asset_type="image",
                    severity=self._severity(confidence),
                    disposition=self._disposition(confidence),
                ))
            return summary
        except Exception as exc:
            LOGGER.error("_detect_mislabeling failed: %s", exc)
            self.warnings.append(f"Confident learning error: {exc}")
            summary["error"] = str(exc)
            return summary

    # ------------------------------------------------------------------
    # CONTRIBUTOR RISK AGGREGATION
    # ------------------------------------------------------------------
    def aggregate_contributor_risk(self,
                                   samples: Sequence[DatasetSample]) -> Dict[str, Any]:
        """Aggregate sample-level findings into per-contributor risk scores.

        Risk combines three factors, because raw counts alone mislead — a vendor
        supplying 10,000 images will always accumulate more findings than one
        supplying 50:

        * **severity-weighted noisy-OR** of that contributor's finding confidences;
        * **finding density** (findings per supplied sample);
        * **attack diversity** (distinct attack classes observed).

        Args:
            samples: All scanned samples, used for per-contributor denominators.

        Returns:
            Mapping of contributor name to
            ``{"risk", "findings", "types", "samples", "density", ...}``.
        """
        output: Dict[str, Any] = {}
        try:
            sample_counts: Dict[str, int] = defaultdict(int)
            for sample in samples:
                sample_counts[sample.contributor or "UNKNOWN"] += 1

            grouped: Dict[str, List[Finding]] = defaultdict(list)
            unattributed = 0
            for finding in self.findings:
                if finding.attack_class == AttackClass.CONTRIBUTOR_RISK.value:
                    continue
                if finding.contributor:
                    grouped[finding.contributor].append(finding)
                else:
                    unattributed += 1

            weights = {"CRITICAL": 1.0, "HIGH": 0.7, "MEDIUM": 0.4, "LOW": 0.15, "INFO": 0.0}
            for name, count in sample_counts.items():
                items = grouped.get(name, [])
                if not items and name not in grouped:
                    output[name] = {
                        "risk": 0.0, "findings": 0, "types": [], "samples": int(count),
                        "density": 0.0, "severity_counts": {}, "affected_samples": 0,
                        "assessment": "No integrity findings attributed to this contributor.",
                    }
                    continue

                product = 1.0
                severity_counts: Dict[str, int] = defaultdict(int)
                affected: set[str] = set()
                for finding in items:
                    contribution = weights.get(finding.severity, 0.1) * finding.confidence
                    product *= (1.0 - min(max(contribution, 0.0), 0.999))
                    severity_counts[finding.severity] += 1
                    affected.add(finding.affected_asset)

                confidence_risk = 1.0 - product
                density = len(items) / max(count, 1)
                density_risk = float(np.clip(density * 2.0, 0.0, 1.0))
                types = sorted({f.attack_class for f in items})
                diversity_risk = float(np.clip((len(types) - 1) / 4.0, 0.0, 1.0))

                risk = float(np.clip(
                    0.60 * confidence_risk + 0.25 * density_risk + 0.15 * diversity_risk,
                    0.0, 1.0))

                output[name] = {
                    "risk": round(risk, 4),
                    "findings": len(items),
                    "types": types,
                    "samples": int(count),
                    "affected_samples": len(affected),
                    "density": round(density, 4),
                    "severity_counts": dict(severity_counts),
                    "confidence_component": round(confidence_risk, 4),
                    "density_component": round(density_risk, 4),
                    "diversity_component": round(diversity_risk, 4),
                    "assessment": self._contributor_assessment(name, risk, len(items),
                                                               types, count),
                }

            if unattributed:
                output.setdefault("_unattributed", {
                    "risk": 0.0, "findings": unattributed, "types": [], "samples": 0,
                    "assessment": (f"{unattributed} finding(s) could not be attributed to a "
                                   "contributor — dataset metadata lacks contributor/source "
                                   "fields and directory naming gave no hint."),
                })
                self.warnings.append(
                    f"{unattributed} finding(s) unattributed: no contributor metadata present")

            LOGGER.info("Contributor aggregation: %d contributor(s)", len(sample_counts))
            return output
        except Exception as exc:
            LOGGER.error("aggregate_contributor_risk failed: %s", exc)
            self.warnings.append(f"Contributor aggregation error: {exc}")
            return output

    def _contributor_assessment(self, name: str, risk: float, count: int,
                                types: List[str], samples: int) -> str:
        """Write the plain-language verdict shown next to a contributor's bar.

        Args:
            name: Contributor name.
            risk: Aggregated risk score.
            count: Number of findings.
            types: Distinct attack classes.
            samples: Samples supplied.

        Returns:
            Human-readable assessment sentence.
        """
        try:
            readable = ", ".join(t.replace("_", " ").lower() for t in types) or "none"
            if risk >= 0.80:
                stance = (f"REJECT CONTRIBUTION. {name} supplied {samples} sample(s) with "
                          f"{count} integrity finding(s) spanning {readable}. Treat this "
                          "source as hostile or fundamentally compromised pending investigation")
            elif risk >= 0.60:
                stance = (f"QUARANTINE AND INVESTIGATE. {name} shows {count} finding(s) "
                          f"({readable}) across {samples} sample(s) — well above benign "
                          "annotation noise")
            elif risk >= 0.35:
                stance = (f"MANUAL REVIEW. {name} has {count} finding(s) ({readable}); "
                          "could be sloppy collection rather than deliberate poisoning")
            elif count:
                stance = (f"ACCEPT WITH MONITORING. {name} has {count} low-severity "
                          f"finding(s), consistent with normal data-quality variance")
            else:
                stance = f"ACCEPT. No integrity findings attributed to {name}"
            return stance + "."
        except Exception:
            return "Assessment unavailable."

    def _emit_contributor_findings(self, contributor_risk: Dict[str, Any]) -> None:
        """Raise a dataset-level finding for each high-risk contributor.

        Args:
            contributor_risk: Output of :meth:`aggregate_contributor_risk`.
        """
        try:
            for name, data in contributor_risk.items():
                if name.startswith("_"):
                    continue
                risk = float(data.get("risk", 0.0))
                if risk < 0.60:
                    continue
                self.findings.append(Finding.create(
                    attack_class=AttackClass.CONTRIBUTOR_RISK.value,
                    affected_asset=name,
                    confidence=risk,
                    reason=str(data.get("assessment", "")),
                    evidence={
                        "risk": risk,
                        "findings": data.get("findings", 0),
                        "samples": data.get("samples", 0),
                        "affected_samples": data.get("affected_samples", 0),
                        "attack_types": data.get("types", []),
                        "finding_density": data.get("density", 0.0),
                        "threshold": 0.60,
                        "severity_counts": data.get("severity_counts", {}),
                    },
                    module=self.MODULE_NAME,
                    detector="contributor_aggregation",
                    contributor=name,
                    asset_type="contributor",
                    severity=self._severity(risk),
                    disposition="QUARANTINE" if risk >= 0.80 else "REVIEW",
                ))
        except Exception as exc:
            LOGGER.error("_emit_contributor_findings failed: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _severity(self, confidence: float) -> str:
        """Map confidence to severity using this scan's configured thresholds.

        Args:
            confidence: Value in ``[0, 1]``.

        Returns:
            Severity name.
        """
        from src.reporting.schema import severity_from_confidence
        return severity_from_confidence(
            confidence,
            critical=float(self.severity_map.get("critical", 0.85)),
            high=float(self.severity_map.get("high", 0.65)),
            medium=float(self.severity_map.get("medium", 0.40)),
        )

    def _disposition(self, confidence: float) -> str:
        """Map confidence to a disposition using configured thresholds.

        Args:
            confidence: Value in ``[0, 1]``.

        Returns:
            Disposition name.
        """
        from src.reporting.schema import disposition_from_confidence
        return disposition_from_confidence(
            confidence,
            quarantine=float(self.disposition_map.get("quarantine", 0.80)),
            review=float(self.disposition_map.get("review", 0.50)),
        )


def scan_dataset(path: str | Path, fmt: Optional[str] = None,
                 max_images: Optional[int] = None) -> Dict[str, Any]:
    """Convenience wrapper running a full data integrity scan.

    Args:
        path: Dataset root or COCO annotation JSON.
        fmt: Optional explicit format.
        max_images: Cap on images scanned.

    Returns:
        Scan result dictionary.
    """
    try:
        return DataIntegrityEngine().scan(path, fmt=fmt, max_images=max_images)
    except Exception as exc:
        LOGGER.error("scan_dataset failed: %s", exc)
        return {"module": "data_scanner", "target": str(path), "findings": [],
                "summary": summarize_findings([]), "limitations": [str(exc)]}


__all__ = ["DataIntegrityEngine", "scan_dataset"]
