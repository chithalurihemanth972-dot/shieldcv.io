"""
MODULE 4 — Distribution Shift Detector.

Compares a baseline dataset against current operational data and decides not
merely *whether* the data moved, but **why** — the distinction that matters
operationally:

``NATURAL_OPERATIONAL_DRIFT``
    Coherent, physically plausible change: a new season, a different sensor, a
    change of theatre. Brightness/warmth/sharpness move together in ways the
    physical world produces. Action: recalibrate.

``SUSPICIOUS_MANIPULATION``
    Incoherent change: feature-space displacement without a matching pixel-level
    explanation, collapsed or inflated variance, or a shift concentrated in a
    minority of samples. Action: investigate as a possible attack.

``MIXED_DRIFT``
    Both signatures present — genuine environmental change that may be
    concealing deliberate manipulation.

Pixel level: brightness, contrast, colour warmth (R/B), Canny edge density and
Laplacian sharpness, each tested at ``|Z| > 2.0``.
Feature level: MMD, centroid displacement and variance ratio over frozen
ResNet-18 embeddings.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.analysis.embeddings import get_extractor
from src.analysis.statistics import (
    kolmogorov_smirnov,
    maximum_mean_discrepancy,
    robust_stats,
    safe_array,
)
from src.config import get_config
from src.loaders.yolo_loader import load_dataset
from src.reporting.schema import AttackClass, Finding, reset_finding_ids, summarize_findings
from src.utils.image_utils import compute_pixel_properties, load_image
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

PIXEL_PROPERTIES = ("brightness", "contrast", "color_warmth", "edge_density", "sharpness")

PROPERTY_LABELS = {
    "brightness": "mean luminance",
    "contrast": "luminance standard deviation",
    "color_warmth": "red/blue channel ratio",
    "edge_density": "Canny edge pixel fraction",
    "sharpness": "variance of the Laplacian",
}

# Physically coherent co-movements. When a shift matches one of these patterns
# it has a mundane explanation and should not be escalated as an attack.
# Physically coherent co-movements, expressed as REQUIRED DIRECTIONS rather than
# bare property sets. Direction is what separates a real physical process from a
# coincidence: haze raises brightness while lowering contrast and edge density,
# whereas a brightness rise WITH rising contrast is not haze and must not be
# excused as one. ``+1`` means the property must increase, ``-1`` decrease and
# ``0`` that it may move either way.
NATURAL_SIGNATURES: Tuple[Dict[str, Any], ...] = (
    {"name": "failing daylight or dusk (illumination loss)",
     "directions": {"brightness": -1, "contrast": -1, "sharpness": -1,
                    "color_warmth": 0},
     "required": ("brightness",), "min_matches": 2},
    {"name": "atmospheric attenuation (haze, dust, rain)",
     "directions": {"brightness": +1, "contrast": -1, "edge_density": -1,
                    "sharpness": -1},
     "required": ("contrast", "edge_density", "sharpness"), "min_matches": 2},
    {"name": "sensor or optics change (focus, resolution)",
     "directions": {"sharpness": +1, "edge_density": +1, "contrast": 0},
     "required": ("sharpness",), "min_matches": 1},
    {"name": "seasonal or theatre change (vegetation, terrain colour)",
     "directions": {"color_warmth": 0, "brightness": 0, "contrast": 0},
     "required": ("color_warmth",), "min_matches": 2},
)

# A property counts as materially changed when its relative change exceeds this,
# regardless of Z. Z alone is unsafe: a baseline property with near-zero spread
# (a synthetic or tightly-controlled collection) yields enormous Z values for
# physically trivial changes, while a naturally variable property can move a
# long way at modest Z.
# Default only — the effective value is read from
# ``drift_detector.material_relative_change`` in config.
MATERIAL_RELATIVE_CHANGE = 0.15

class DistributionShiftDetector:
    """Detects and characterises distribution shift between two datasets.

    Attributes:
        cfg: Effective configuration.
        findings: Findings from the most recent comparison.
    """

    MODULE_NAME = "drift_detector"

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialise the detector and read its thresholds.

        Args:
            config: Optional configuration override.
        """
        self.cfg = config or get_config()
        self.section = self.cfg.section("drift_detector")
        self.zscore_threshold = float(self.section.get("zscore_threshold", 2.0))
        self.mmd_threshold = float(self.section.get("mmd_threshold", 0.05))
        self.centroid_threshold = float(self.section.get("centroid_threshold", 0.15))
        self.variance_ratio_low = float(self.section.get("variance_ratio_low", 0.60))
        self.variance_ratio_high = float(self.section.get("variance_ratio_high", 1.70))
        self.min_samples = int(self.section.get("min_samples", 10))
        self.material_relative_change = float(
            self.section.get("material_relative_change", MATERIAL_RELATIVE_CHANGE))
        self.bimodality_separation = float(self.section.get("bimodality_separation", 5.0))
        self.bimodality_minority_fraction = float(
            self.section.get("bimodality_minority_fraction", 0.08))
        self.findings: List[Finding] = []
        self.limitations: List[str] = []

    # ------------------------------------------------------------------
    def compare(self, baseline: str | Path, current: str | Path,
                max_images: Optional[int] = None,
                progress_callback: Optional[Callable[[str, int, int], None]] = None
                ) -> Dict[str, Any]:
        """Compare a current dataset against a trusted baseline.

        Args:
            baseline: Path to the reference/baseline dataset.
            current: Path to the current operational dataset.
            max_images: Optional cap on images loaded per side.
            progress_callback: Optional ``callable(stage, done, total)``.

        Returns:
            Result with ``risk_score``, ``severity``, ``shift_type``,
            ``characterization``, ``confidence``, ``recommendation``,
            ``pixel_analysis``, ``feature_analysis``, ``findings`` and
            ``limitations``.
        """
        started = time.time()
        reset_finding_ids("DFT")
        self.findings = []
        self.limitations = []

        result: Dict[str, Any] = {
            "module": self.MODULE_NAME,
            "baseline": str(baseline), "current": str(current),
            "risk_score": 0.0, "severity": "INFO", "shift_type": "NO_SIGNIFICANT_DRIFT",
            "characterization": "", "confidence": 0.0, "recommendation": "",
            "pixel_analysis": {}, "feature_analysis": {}, "findings": [],
            "summary": {}, "limitations": [], "duration_seconds": 0.0,
        }

        def report(stage: str, done: int, total: int) -> None:
            """Forward progress to the caller's callback."""
            if progress_callback:
                try:
                    progress_callback(stage, done, total)
                except Exception:
                    pass

        try:
            report("loading", 0, 2)
            baseline_paths = self._collect_paths(baseline, max_images)
            report("loading", 1, 2)
            current_paths = self._collect_paths(current, max_images)
            report("loading", 2, 2)

            result["baseline_samples"] = len(baseline_paths)
            result["current_samples"] = len(current_paths)

            if len(baseline_paths) < self.min_samples or len(current_paths) < self.min_samples:
                reason = (f"Insufficient samples for drift analysis: baseline "
                          f"{len(baseline_paths)}, current {len(current_paths)}; "
                          f"{self.min_samples} required per side")
                self.limitations.append(reason)
                result["characterization"] = reason
                result["recommendation"] = "Collect more operational data before assessing drift."
                result["limitations"] = self.limitations
                result["summary"] = summarize_findings([])
                result["duration_seconds"] = round(time.time() - started, 2)
                LOGGER.warning(reason)
                return result

            report("pixel", 0, 1)
            pixel = self._pixel_analysis(baseline_paths, current_paths)
            result["pixel_analysis"] = pixel
            report("pixel", 1, 1)

            report("features", 0, 1)
            feature = self._feature_analysis(baseline_paths, current_paths,
                                             lambda d, t: report("features", d, t))
            result["feature_analysis"] = feature
            report("features", 1, 1)

            verdict = self._characterize(pixel, feature)
            result.update(verdict)

            self._emit_findings(result, current)

            payloads = [f.to_dict() for f in self.findings]
            result["findings"] = payloads
            result["summary"] = summarize_findings(payloads)
            result["limitations"] = list(dict.fromkeys(self.limitations))
            result["duration_seconds"] = round(time.time() - started, 2)

            LOGGER.info("Drift comparison complete: %s, risk %.2f (%.1fs)",
                        result["shift_type"], result["risk_score"],
                        result["duration_seconds"])
            return result
        except Exception as exc:
            LOGGER.error("Drift comparison failed: %s", exc, exc_info=True)
            self.limitations.append(f"Drift analysis aborted: {exc}")
            result["limitations"] = self.limitations
            result["summary"] = summarize_findings([])
            result["duration_seconds"] = round(time.time() - started, 2)
            return result

    # ------------------------------------------------------------------
    def _collect_paths(self, source: str | Path,
                       max_images: Optional[int]) -> List[str]:
        """Resolve a dataset directory to a list of image paths.

        Args:
            source: Dataset root or plain image directory.
            max_images: Optional cap.

        Returns:
            List of image file paths.
        """
        try:
            dataset = load_dataset(source, max_images=max_images)
            paths = [sample.image_path for sample in dataset.samples]
            if dataset.errors:
                self.limitations.extend(dataset.errors[:3])
            if paths:
                return paths
        except Exception as exc:
            LOGGER.debug("Dataset load failed for %s (%s); falling back to image scan",
                         source, exc)
        try:
            from src.utils.image_utils import list_images
            return list_images(source, limit=max_images)
        except Exception as exc:
            LOGGER.error("_collect_paths failed for %s: %s", source, exc)
            self.limitations.append(f"Could not enumerate images in {source}: {exc}")
            return []

    # ------------------------------------------------------------------
    def _pixel_analysis(self, baseline_paths: Sequence[str],
                        current_paths: Sequence[str]) -> Dict[str, Any]:
        """Compare the five pixel-level properties between the two datasets.

        Args:
            baseline_paths: Baseline image paths.
            current_paths: Current image paths.

        Returns:
            Per-property comparison with Z-scores and shifted-property list.
        """
        summary: Dict[str, Any] = {"available": False, "properties": {},
                                   "shifted_properties": []}
        try:
            baseline_values = self._property_matrix(baseline_paths)
            current_values = self._property_matrix(current_paths)
            if not baseline_values or not current_values:
                summary["reason"] = "Pixel properties could not be computed"
                self.limitations.append(summary["reason"])
                return summary

            shifted: List[str] = []
            for name in PIXEL_PROPERTIES:
                base = safe_array(baseline_values.get(name, []))
                curr = safe_array(current_values.get(name, []))
                if base.size < 2 or curr.size < 2:
                    summary["properties"][name] = {"available": False,
                                                   "reason": "insufficient samples"}
                    continue

                base_stats = robust_stats(base)
                current_mean = float(np.mean(curr))
                spread = float(base_stats.get("std", 0.0))
                # Fall back to MAD-derived spread when std collapses, so a
                # near-constant baseline cannot manufacture an infinite Z.
                if spread <= 1e-9:
                    spread = float(base_stats.get("mad", 0.0)) * 1.4826
                zscore = ((current_mean - float(base_stats["mean"])) / spread
                          if spread > 1e-9 else 0.0)

                ks = kolmogorov_smirnov(base, curr)
                entry = {
                    "available": True,
                    "label": PROPERTY_LABELS[name],
                    "baseline_mean": round(float(base_stats["mean"]), 5),
                    "baseline_std": round(float(base_stats.get("std", 0.0)), 5),
                    "current_mean": round(current_mean, 5),
                    "current_std": round(float(np.std(curr)), 5),
                    "delta": round(current_mean - float(base_stats["mean"]), 5),
                    "relative_change": round(
                        (current_mean - float(base_stats["mean"]))
                        / max(abs(float(base_stats["mean"])), 1e-9), 4),
                    "zscore": round(float(zscore), 4),
                    "threshold": self.zscore_threshold,
                    "ks_statistic": round(float(ks.get("statistic", 0.0)), 4),
                    "direction": "increase" if current_mean >= float(base_stats["mean"])
                                 else "decrease",
                }
                relative = abs(entry["relative_change"])
                entry["material"] = bool(relative >= self.material_relative_change)
                entry["statistically_significant"] = bool(
                    abs(zscore) > self.zscore_threshold)
                # Require BOTH statistical significance and practical magnitude.
                # Significance alone fires on trivial changes when the baseline
                # is tightly controlled; magnitude alone fires on noise when the
                # baseline is broad.
                entry["shifted"] = bool(entry["material"]
                                        and entry["statistically_significant"])
                summary["properties"][name] = entry
                if entry["shifted"]:
                    shifted.append(name)

            summary["available"] = True
            summary["shifted_properties"] = shifted
            summary["num_shifted"] = len(shifted)
            summary["max_abs_zscore"] = round(max(
                (abs(p.get("zscore", 0.0)) for p in summary["properties"].values()
                 if p.get("available")), default=0.0), 4)
            return summary
        except Exception as exc:
            LOGGER.error("_pixel_analysis failed: %s", exc)
            summary["reason"] = str(exc)
            self.limitations.append(f"Pixel-level drift analysis error: {exc}")
            return summary

    def _property_matrix(self, paths: Sequence[str]) -> Dict[str, List[float]]:
        """Compute pixel properties for every readable image.

        Args:
            paths: Image paths.

        Returns:
            Mapping of property name to the list of per-image values.
        """
        values: Dict[str, List[float]] = {name: [] for name in PIXEL_PROPERTIES}
        failures = 0
        try:
            for path in paths:
                image = load_image(path)
                if image is None:
                    failures += 1
                    continue
                properties = compute_pixel_properties(image)
                for name in PIXEL_PROPERTIES:
                    value = properties.get(name)
                    if value is not None and np.isfinite(value):
                        values[name].append(float(value))
            if failures:
                self.limitations.append(
                    f"{failures} image(s) could not be read during pixel analysis")
            return values
        except Exception as exc:
            LOGGER.error("_property_matrix failed: %s", exc)
            return values

    # ------------------------------------------------------------------
    def _feature_analysis(self, baseline_paths: Sequence[str],
                          current_paths: Sequence[str],
                          progress: Optional[Callable[[int, int], None]] = None
                          ) -> Dict[str, Any]:
        """Compare embedding distributions via MMD, centroid shift and variance ratio.

        Args:
            baseline_paths: Baseline image paths.
            current_paths: Current image paths.
            progress: Optional ``callable(done, total)``.

        Returns:
            Feature-level comparison dictionary.
        """
        summary: Dict[str, Any] = {"available": False}
        try:
            extractor = get_extractor()
            total = len(baseline_paths) + len(current_paths)
            state = {"done": 0}

            def forward(done: int, _total: int) -> None:
                """Aggregate embedding progress across both datasets."""
                state["done"] = done
                if progress:
                    try:
                        progress(min(state["done"], total), total)
                    except Exception:
                        pass

            baseline_result = extractor.embed_paths(list(baseline_paths),
                                                    progress_callback=forward)
            current_result = extractor.embed_paths(list(current_paths),
                                                   progress_callback=forward)

            summary["backbone_status"] = baseline_result.backbone_status
            if not baseline_result.is_reliable or not current_result.is_reliable:
                summary["reason"] = ("Embedding backbone unavailable or unreliable: "
                                     "feature-level drift analysis NOT AVAILABLE")
                self.limitations.append(summary["reason"])
                self.limitations.extend(baseline_result.warnings[:2])
                return summary

            base_features = np.asarray(baseline_result.features, dtype=np.float64)
            current_features = np.asarray(current_result.features, dtype=np.float64)
            if base_features.size == 0 or current_features.size == 0:
                summary["reason"] = "No embeddings produced"
                self.limitations.append(summary["reason"])
                return summary

            mmd = maximum_mean_discrepancy(base_features, current_features)
            base_centroid = base_features.mean(axis=0)
            current_centroid = current_features.mean(axis=0)
            centroid_shift = float(np.linalg.norm(current_centroid - base_centroid))
            cosine = float(np.dot(base_centroid, current_centroid) / (
                (np.linalg.norm(base_centroid) * np.linalg.norm(current_centroid)) or 1.0))

            base_variance = float(np.mean(np.var(base_features, axis=0)))
            current_variance = float(np.mean(np.var(current_features, axis=0)))
            variance_ratio = current_variance / max(base_variance, 1e-12)

            # Per-sample displacement identifies whether the shift is uniform
            # (whole population moved) or concentrated in a minority subset,
            # which distinguishes an environmental change from injected samples.
            distances = np.linalg.norm(current_features - base_centroid, axis=1)
            base_distances = np.linalg.norm(base_features - base_centroid, axis=1)
            base_mean_distance = float(np.mean(base_distances))
            base_std_distance = float(np.std(base_distances)) or 1e-9
            outlier_fraction = float(np.mean(
                distances > base_mean_distance + 2.0 * base_std_distance))

            # Distance from the BASELINE centroid cannot distinguish injection
            # from a uniform environmental shift: when the whole population
            # moves, every sample is an "outlier" and the fraction saturates at
            # 1.0. Injection is instead identified by BIMODALITY about the
            # CURRENT centroid — a genuinely mixed stream contains one cluster
            # that stayed put and another that did not.
            current_distances = np.linalg.norm(current_features - current_centroid, axis=1)
            bimodality = self._bimodality(current_distances)

            # How much of the feature shift is explained by a rigid translation
            # of the whole population (natural) versus a change in the shape of
            # the distribution (manipulation).
            base_spread = float(np.mean(base_distances))
            current_spread = float(np.mean(current_distances))
            spread_ratio = current_spread / max(base_spread, 1e-12)

            summary.update({
                "available": True,
                "mmd": round(float(mmd.get("mmd", 0.0)), 6),
                "mmd_threshold": self.mmd_threshold,
                "mmd_significant": bool(float(mmd.get("mmd", 0.0)) > self.mmd_threshold),
                "centroid_shift": round(centroid_shift, 6),
                "centroid_threshold": self.centroid_threshold,
                "centroid_cosine": round(cosine, 6),
                "baseline_variance": round(base_variance, 8),
                "current_variance": round(current_variance, 8),
                "variance_ratio": round(variance_ratio, 4),
                "variance_collapsed": bool(variance_ratio < self.variance_ratio_low),
                "variance_inflated": bool(variance_ratio > self.variance_ratio_high),
                "outlier_fraction": round(outlier_fraction, 4),
                "bimodality": bimodality,
                "spread_ratio": round(spread_ratio, 4),
                # A concentrated (injected) shift shows a bimodal current
                # distribution AND only a subset displaced from the baseline.
                "concentrated_shift": bool(bimodality.get("bimodal")
                                           and 0.02 < outlier_fraction < 0.80),
                "baseline_n": int(base_features.shape[0]),
                "current_n": int(current_features.shape[0]),
            })
            return summary
        except Exception as exc:
            LOGGER.error("_feature_analysis failed: %s", exc)
            summary["reason"] = str(exc)
            self.limitations.append(f"Feature-level drift analysis error: {exc}")
            return summary

    def _bimodality(self, distances: np.ndarray) -> Dict[str, Any]:
        """Test whether per-sample distances form two distinct groups.

        Uses a 1-D two-means split and reports the between-group separation
        relative to within-group spread. A unimodal population (everything
        shifted together) yields low separation; an injected minority yields a
        clear gap.

        Args:
            distances: Per-sample distances from the current centroid.

        Returns:
            Dictionary with ``bimodal``, ``separation`` and group sizes.
        """
        summary: Dict[str, Any] = {"bimodal": False, "separation": 0.0}
        try:
            values = np.asarray(distances, dtype=np.float64).ravel()
            if values.size < 8:
                return summary

            ordered = np.sort(values)
            best_split, best_separation = 0, 0.0
            # Evaluate every split point between the 10th and 90th percentile.
            low = max(1, int(0.10 * ordered.size))
            high = min(ordered.size - 1, int(0.90 * ordered.size))
            for position in range(low, high):
                left, right = ordered[:position], ordered[position:]
                within = (left.std() * left.size + right.std() * right.size) / ordered.size
                between = abs(right.mean() - left.mean())
                separation = between / max(within, 1e-9)
                if separation > best_separation:
                    best_separation, best_split = separation, position

            minority = min(best_split, ordered.size - best_split) / ordered.size
            summary.update({
                "separation": round(float(best_separation), 4),
                "minority_fraction": round(float(minority), 4),
                "group_sizes": [int(best_split), int(ordered.size - best_split)],
                # Threshold chosen so a single displaced cluster registers while
                # the natural spread of a homogeneous population does not.
                # Calibrated against homogeneous baselines, which reach roughly
                # 3.0 by chance from the tails of a unimodal distribution.
                "bimodal": bool(best_separation > self.bimodality_separation
                                and minority > self.bimodality_minority_fraction),
            })
            return summary
        except Exception as exc:
            LOGGER.error("_bimodality failed: %s", exc)
            return summary

    # ------------------------------------------------------------------
    def _characterize(self, pixel: Dict[str, Any],
                      feature: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the shift and produce the analyst-facing narrative.

        Args:
            pixel: Pixel-level analysis result.
            feature: Feature-level analysis result.

        Returns:
            Dictionary with ``shift_type``, ``risk_score``, ``severity``,
            ``characterization``, ``confidence`` and ``recommendation``.
        """
        try:
            shifted = list(pixel.get("shifted_properties", []))
            properties = pixel.get("properties", {})
            feature_available = bool(feature.get("available"))

            natural_matches = self._match_natural_signatures(shifted, properties)
            explained: set = set()
            for match in natural_matches:
                explained.update(match["explains"])
            unexplained = [name for name in shifted if name not in explained]

            mmd_significant = bool(feature.get("mmd_significant"))
            centroid_significant = bool(
                feature.get("centroid_shift", 0.0) > self.centroid_threshold)
            variance_anomaly = bool(feature.get("variance_collapsed")
                                    or feature.get("variance_inflated"))
            concentrated = bool(feature.get("concentrated_shift"))

            # A feature-space shift with NO pixel-level explanation is the key
            # manipulation signature: natural environmental change always leaves
            # a pixel-level trace, whereas adversarial content alters semantics
            # while keeping low-level statistics within normal bounds.
            material_pixel_change = [
                name for name, entry in properties.items()
                if entry.get("available") and entry.get("material")]
            unexplained_semantic_shift = bool(
                (mmd_significant or centroid_significant) and not material_pixel_change)

            suspicious_signals: List[str] = []
            if unexplained_semantic_shift:
                suspicious_signals.append(
                    "feature-space displacement with no corresponding pixel-level change")
            if unexplained:
                suspicious_signals.append(
                    "pixel shifts that fit no coherent physical pattern "
                    f"({', '.join(unexplained)})")
            if feature.get("variance_collapsed"):
                suspicious_signals.append(
                    f"embedding variance collapsed to {feature.get('variance_ratio')}x baseline, "
                    "indicating unnaturally homogeneous data such as duplicated or "
                    "synthetically generated frames")
            if feature.get("variance_inflated"):
                suspicious_signals.append(
                    f"embedding variance inflated to {feature.get('variance_ratio')}x baseline, "
                    "indicating injected heterogeneous content")
            if concentrated:
                suspicious_signals.append(
                    "the current data is bimodal in feature space (separation "
                    f"{feature.get('bimodality', {}).get('separation', 0):.1f}, minority group "
                    f"{feature.get('bimodality', {}).get('minority_fraction', 0):.0%}) — one "
                    "group matches the baseline and another does not, which is characteristic "
                    "of injected content rather than a uniform environmental change")

            natural_signals = [match["name"] for match in natural_matches]

            has_drift = bool(shifted or mmd_significant or centroid_significant
                             or variance_anomaly)
            if not has_drift:
                shift_type = "NO_SIGNIFICANT_DRIFT"
            elif suspicious_signals and natural_signals:
                shift_type = "MIXED_DRIFT"
            elif suspicious_signals:
                shift_type = "SUSPICIOUS_MANIPULATION"
            elif natural_signals:
                shift_type = "NATURAL_OPERATIONAL_DRIFT"
            else:
                shift_type = "MIXED_DRIFT"

            risk = self._risk_score(pixel, feature, suspicious_signals, natural_signals,
                                    shift_type)
            severity = ("CRITICAL" if risk >= 0.85 else "HIGH" if risk >= 0.65
                        else "MEDIUM" if risk >= 0.40 else "LOW" if risk >= 0.15 else "INFO")

            confidence = 0.35
            if pixel.get("available"):
                confidence += 0.25
            if feature_available:
                confidence += 0.30
            if len(shifted) >= 2 or (mmd_significant and centroid_significant):
                confidence += 0.10
            confidence = float(np.clip(confidence, 0.0, 0.95))
            if not feature_available:
                self.limitations.append(
                    "Feature-level analysis unavailable: drift classification rests on "
                    "pixel statistics alone and cannot detect semantic manipulation that "
                    "preserves low-level image statistics")

            return {
                "shift_type": shift_type,
                "risk_score": round(risk, 4),
                "severity": severity,
                "confidence": round(confidence, 3),
                "characterization": self._narrative(
                    shift_type, shifted, properties, natural_signals,
                    suspicious_signals, feature),
                "recommendation": self._recommendation(shift_type, severity),
                "natural_signatures": natural_signals,
                "suspicious_signals": suspicious_signals,
                "unexplained_properties": unexplained,
            }
        except Exception as exc:
            LOGGER.error("_characterize failed: %s", exc)
            self.limitations.append(f"Shift characterisation error: {exc}")
            return {"shift_type": "UNKNOWN", "risk_score": 0.0, "severity": "INFO",
                    "confidence": 0.0, "characterization": f"Characterisation failed: {exc}",
                    "recommendation": "Re-run drift analysis."}

    def _match_natural_signatures(self, shifted: Sequence[str],
                                  properties: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify which physically coherent patterns the observed shift matches.

        A signature matches when enough of its properties moved in the direction
        that physical process actually produces, and none moved in a direction
        that process forbids. Matching on property identity alone would excuse
        any change involving the right properties regardless of sign.

        Args:
            shifted: Names of properties considered shifted.
            properties: Full per-property analysis.

        Returns:
            List of matched natural-signature descriptors.
        """
        matches: List[Dict[str, Any]] = []
        try:
            # Consider every materially-changed property, not only those that
            # passed the strict shift test: a coherent physical process moves
            # several properties together, some of which sit just under the
            # threshold, and ignoring those would fragment the pattern.
            moved: Dict[str, int] = {}
            for name, entry in properties.items():
                if not entry.get("available"):
                    continue
                if entry.get("material") or entry.get("shifted"):
                    moved[name] = 1 if entry.get("direction") == "increase" else -1

            if not moved:
                return matches

            for signature in NATURAL_SIGNATURES:
                directions = signature["directions"]
                agreeing: List[str] = []
                contradicting: List[str] = []
                for name, observed in moved.items():
                    expected = directions.get(name)
                    if expected is None:
                        continue
                    if expected == 0 or expected == observed:
                        agreeing.append(name)
                    else:
                        contradicting.append(name)

                if contradicting:
                    continue
                if len(agreeing) < int(signature.get("min_matches", 2)):
                    continue
                if not any(name in agreeing for name in signature["required"]):
                    continue
                # A physical process must leave at least one properly shifted
                # property. Without this, sub-threshold noise in the right
                # directions manufactures a "natural" explanation for a purely
                # semantic manipulation and downgrades a real attack to
                # MIXED_DRIFT.
                if not any(name in shifted for name in agreeing):
                    continue

                matches.append({
                    "name": signature["name"],
                    "properties": agreeing,
                    "explains": tuple(agreeing),
                    "contradicting": contradicting,
                })

            # Keep only the most complete explanation when several match, so the
            # narrative names one coherent process rather than a vague list.
            matches.sort(key=lambda m: len(m["properties"]), reverse=True)
            return matches[:2]
        except Exception as exc:
            LOGGER.error("_match_natural_signatures failed: %s", exc)
            return matches

    def _risk_score(self, pixel: Dict[str, Any], feature: Dict[str, Any],
                    suspicious: Sequence[str], natural: Sequence[str],
                    shift_type: str) -> float:
        """Combine the evidence into a single risk score.

        Args:
            pixel: Pixel-level analysis.
            feature: Feature-level analysis.
            suspicious: Suspicious signal descriptions.
            natural: Matched natural signature names.
            shift_type: The assigned shift classification.

        Returns:
            Risk score in ``[0, 1]``.
        """
        try:
            magnitude = min(1.0, float(pixel.get("max_abs_zscore", 0.0)) / 8.0)
            breadth = min(1.0, float(pixel.get("num_shifted", 0)) / 3.0)
            feature_component = 0.0
            if feature.get("available"):
                feature_component = max(
                    min(1.0, float(feature.get("mmd", 0.0)) / max(self.mmd_threshold * 4, 1e-9)),
                    min(1.0, float(feature.get("centroid_shift", 0.0))
                        / max(self.centroid_threshold * 4, 1e-9)))

            base = 0.35 * magnitude + 0.20 * breadth + 0.45 * feature_component
            suspicion = min(0.45, 0.15 * len(suspicious))

            if shift_type == "SUSPICIOUS_MANIPULATION":
                score = min(1.0, 0.45 + 0.55 * base + suspicion)
            elif shift_type == "MIXED_DRIFT":
                score = min(1.0, 0.30 + 0.45 * base + suspicion * 0.8)
            elif shift_type == "NATURAL_OPERATIONAL_DRIFT":
                # Natural drift still degrades accuracy and must be actioned,
                # but it is a maintenance issue rather than a security incident.
                score = min(0.55, 0.12 + 0.40 * base)
            else:
                score = min(0.15, 0.30 * base)
            return float(np.clip(score, 0.0, 1.0))
        except Exception as exc:
            LOGGER.error("_risk_score failed: %s", exc)
            return 0.0

    def _narrative(self, shift_type: str, shifted: Sequence[str],
                   properties: Dict[str, Any], natural: Sequence[str],
                   suspicious: Sequence[str], feature: Dict[str, Any]) -> str:
        """Compose the analyst-facing explanation of the shift.

        Args:
            shift_type: Assigned classification.
            shifted: Shifted property names.
            properties: Per-property analysis.
            natural: Matched natural signatures.
            suspicious: Suspicious signal descriptions.
            feature: Feature-level analysis.

        Returns:
            Multi-sentence characterisation.
        """
        try:
            parts: List[str] = []
            if shifted:
                details = []
                for name in shifted:
                    entry = properties.get(name, {})
                    details.append(
                        f"{PROPERTY_LABELS.get(name, name)} {entry.get('direction', 'moved')}d "
                        f"by {abs(entry.get('relative_change', 0.0)):.0%} "
                        f"(Z = {entry.get('zscore', 0.0):+.2f})")
                parts.append("Pixel-level: " + "; ".join(details) + ".")
            else:
                parts.append("Pixel-level: no property moved beyond "
                             f"|Z| = {self.zscore_threshold}.")

            if feature.get("available"):
                parts.append(
                    f"Feature-level: MMD {feature.get('mmd', 0.0):.4f} "
                    f"(threshold {self.mmd_threshold}), centroid displacement "
                    f"{feature.get('centroid_shift', 0.0):.4f} "
                    f"(threshold {self.centroid_threshold}), variance ratio "
                    f"{feature.get('variance_ratio', 1.0):.2f}x.")
            else:
                parts.append("Feature-level: NOT AVAILABLE.")

            if natural:
                parts.append("The pixel changes co-move in a physically coherent way "
                             f"consistent with {', and '.join(natural)}.")
            if suspicious:
                parts.append("Indicators inconsistent with natural change: "
                             + "; ".join(suspicious) + ".")

            if shift_type == "NATURAL_OPERATIONAL_DRIFT":
                parts.append("Assessment: the data has genuinely moved, but every observed "
                             "change has a mundane physical explanation. This is a model "
                             "maintenance issue, not a security incident.")
            elif shift_type == "SUSPICIOUS_MANIPULATION":
                parts.append("Assessment: the observed pattern does not correspond to any "
                             "natural environmental change and should be treated as possible "
                             "deliberate manipulation of the operational data stream.")
            elif shift_type == "MIXED_DRIFT":
                parts.append("Assessment: genuine environmental change is present, but it does "
                             "not account for all of the observed movement. Natural drift can "
                             "be used as cover for manipulation, so the unexplained component "
                             "must be investigated on its own merits.")
            else:
                parts.append("Assessment: the current data is statistically consistent with "
                             "the baseline.")
            return " ".join(parts)
        except Exception as exc:
            LOGGER.error("_narrative failed: %s", exc)
            return f"Narrative unavailable: {exc}"

    def _recommendation(self, shift_type: str, severity: str) -> str:
        """Produce the recommended course of action.

        Args:
            shift_type: Assigned classification.
            severity: Assigned severity band.

        Returns:
            Recommendation text.
        """
        mapping = {
            "NATURAL_OPERATIONAL_DRIFT": (
                "Recalibrate or re-validate the model against current conditions. Collect a "
                "labelled sample of the new distribution and re-measure accuracy before "
                "continuing to rely on existing performance figures."),
            "SUSPICIOUS_MANIPULATION": (
                "Treat as a potential security incident. Quarantine the affected inference "
                "stream, verify the provenance chain for the same period, and confirm the "
                "sensor and ingestion path have not been altered before resuming operations."),
            "MIXED_DRIFT": (
                "Recalibrate for the environmental component AND investigate the unexplained "
                "component separately. Do not attribute the entire shift to conditions until "
                "the anomalous portion has been accounted for."),
            "NO_SIGNIFICANT_DRIFT": (
                "No action required. Continue routine monitoring at the established interval."),
        }
        text = mapping.get(shift_type, "Re-run drift analysis with a larger sample.")
        if severity in ("CRITICAL", "HIGH") and shift_type != "NO_SIGNIFICANT_DRIFT":
            text += " Escalate to the accreditation authority given the magnitude observed."
        return text

    # ------------------------------------------------------------------
    def _emit_findings(self, result: Dict[str, Any], current: str | Path) -> None:
        """Convert the characterisation into a Finding when drift is material.

        Args:
            result: The assembled result dictionary.
            current: Path to the current dataset (the affected asset).
        """
        try:
            if result["shift_type"] == "NO_SIGNIFICANT_DRIFT":
                return
            if float(result["risk_score"]) < 0.15:
                return

            pixel = result.get("pixel_analysis", {})
            feature = result.get("feature_analysis", {})
            self.findings.append(Finding.create(
                attack_class=AttackClass.DISTRIBUTION_SHIFT.value,
                affected_asset=str(current),
                confidence=float(result["confidence"]),
                reason=result["characterization"],
                evidence={
                    "shift_type": result["shift_type"],
                    "risk_score": result["risk_score"],
                    "shifted_properties": pixel.get("shifted_properties", []),
                    "max_abs_zscore": pixel.get("max_abs_zscore", 0.0),
                    "threshold": self.zscore_threshold,
                    "mmd": feature.get("mmd"),
                    "centroid_shift": feature.get("centroid_shift"),
                    "variance_ratio": feature.get("variance_ratio"),
                    "outlier_fraction": feature.get("outlier_fraction"),
                    "natural_signatures": result.get("natural_signatures", []),
                    "suspicious_signals": result.get("suspicious_signals", []),
                    "baseline_samples": result.get("baseline_samples", 0),
                    "current_samples": result.get("current_samples", 0),
                    "recommendation": result["recommendation"],
                },
                module=self.MODULE_NAME, detector="distribution_shift",
                asset_type="dataset", prefix="DFT",
                severity=result["severity"],
            ))
        except Exception as exc:
            LOGGER.error("_emit_findings failed: %s", exc)
            self.limitations.append(f"Finding emission error: {exc}")


def detect_drift(baseline: str | Path, current: str | Path,
                 max_images: Optional[int] = None) -> Dict[str, Any]:
    """Convenience wrapper comparing two datasets for distribution shift.

    Args:
        baseline: Baseline dataset path.
        current: Current dataset path.
        max_images: Optional cap on images per side.

    Returns:
        Drift analysis result dictionary.
    """
    try:
        return DistributionShiftDetector().compare(baseline, current, max_images=max_images)
    except Exception as exc:
        LOGGER.error("detect_drift failed: %s", exc)
        return {"module": "drift_detector", "findings": [], "limitations": [str(exc)],
                "shift_type": "UNKNOWN", "risk_score": 0.0}


__all__ = ["DistributionShiftDetector", "detect_drift", "PIXEL_PROPERTIES"]
