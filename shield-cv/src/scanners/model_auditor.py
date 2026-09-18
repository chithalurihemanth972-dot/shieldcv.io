"""
MODULE 2 — Model Integrity Engine.

Audits a contributed model for backdoors, substitution and weight anomalies,
**without ever retraining it**. The engine first determines how much access it
actually has and then runs only what that access supports, stating the rest as
``NOT AVAILABLE`` rather than silently skipping it:

WHITE_BOX (.pt / .onnx with readable weights)
    * weight hashing (substitution detection)
    * per-layer statistics — mean, std, sparsity, kurtosis
    * Neural Cleanse trigger reverse-engineering (MAD anomaly index > 2.0)
    * activation clustering (KMeans k=2, >70/30 imbalance ⇒ suspicious)

GREY_BOX (weights but no runnable graph)
    * everything above except behavioural probing

BLACK_BOX (prediction API only)
    * behavioural fingerprinting against a reference model (>5% mismatch)
    * output distribution shape (anomalously sharp or flat confidence)

Every report states ``access_level``, a ``confidence`` in the overall verdict,
and an explicit ``limitations`` list.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.analysis.neural_cleanse import NeuralCleanse, make_probe_batch
from src.analysis.spectral import activation_cluster_imbalance
from src.analysis.statistics import (
    kurtosis,
    normalized_entropy,
    robust_stats,
    softmax,
)
from src.config import get_config
from src.crypto.hashing import short_hash
from src.loaders.model_loader import AccessLevel, LoadedModel, load_model
from src.reporting.schema import AttackClass, Finding, reset_finding_ids, summarize_findings
from src.utils.image_utils import list_images, load_image
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class ModelIntegrityEngine:
    """Audits model integrity at the highest access level actually available.

    Attributes:
        cfg: Effective configuration.
        findings: Findings from the most recent audit.
        limitations: Explicit statements of what could not be checked.
    """

    MODULE_NAME = "model_auditor"

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialise the engine and read its thresholds.

        Args:
            config: Optional configuration override.
        """
        self.cfg = config or get_config()
        self.section = self.cfg.section("model_auditor")
        self.weight_cfg = self.section.get("weight_stats", {})
        self.nc_cfg = self.section.get("neural_cleanse", {})
        self.ac_cfg = self.section.get("activation_clustering", {})
        self.bb_cfg = self.section.get("blackbox", {})
        self.findings: List[Finding] = []
        self.limitations: List[str] = []

    # ------------------------------------------------------------------
    def audit(self, model_path: str | Path | Callable,
              reference_path: Optional[str | Path] = None,
              reference_images: Optional[str | Path] = None,
              run_neural_cleanse: Optional[bool] = None,
              progress_callback: Optional[Callable[[str, int, int], None]] = None
              ) -> Dict[str, Any]:
        """Run the full model integrity audit.

        Args:
            model_path: Model file (``.pt``/``.onnx``) or a prediction callable.
            reference_path: Trusted reference model for substitution/behaviour comparison.
            reference_images: Directory of images used as probes and for clustering.
            run_neural_cleanse: Force Neural Cleanse on/off (defaults to config).
            progress_callback: Optional ``callable(stage, done, total)``.

        Returns:
            Audit result with ``access_level``, ``findings``, ``summary``,
            ``weight_statistics``, ``neural_cleanse``, ``activation_clustering``,
            ``blackbox``, ``confidence`` and ``limitations``.
        """
        started = time.time()
        reset_finding_ids("MDL")
        self.findings = []
        self.limitations = []

        result: Dict[str, Any] = {
            "module": self.MODULE_NAME,
            "target": str(model_path) if not callable(model_path) else "<prediction-api>",
            "access_level": AccessLevel.UNAVAILABLE.value,
            "findings": [], "summary": {}, "model": {},
            "weight_statistics": {}, "neural_cleanse": {}, "activation_clustering": {},
            "blackbox": {}, "limitations": [], "confidence": 0.0,
            "duration_seconds": 0.0,
        }

        def report(stage: str, done: int, total: int) -> None:
            """Forward progress to the caller's callback."""
            if progress_callback:
                try:
                    progress_callback(stage, done, total)
                except Exception:
                    pass

        try:
            report("loading", 0, 1)
            model = load_model(model_path)
            result["model"] = model.summary()
            result["access_level"] = model.access_level.value
            self.limitations.extend(model.limitations)
            report("loading", 1, 1)

            if model.access_level == AccessLevel.UNAVAILABLE:
                self.limitations.append(
                    "Model could not be opened: ALL model-integrity checks NOT AVAILABLE")
                result["limitations"] = list(dict.fromkeys(self.limitations))
                result["summary"] = summarize_findings([])
                result["duration_seconds"] = round(time.time() - started, 2)
                LOGGER.error("Model audit aborted — model unavailable")
                return result

            reference: Optional[LoadedModel] = None
            if reference_path is not None:
                reference = load_model(reference_path)
                result["reference_model"] = reference.summary()

            probe_images = self._load_probe_images(reference_images)

            # ---- WHITE/GREY BOX ------------------------------------------
            if model.state_dict:
                report("weights", 0, 1)
                result["weight_statistics"] = self._analyze_weights(model, reference)
                report("weights", 1, 1)

                should_run = (bool(self.nc_cfg.get("enabled", True))
                              if run_neural_cleanse is None else bool(run_neural_cleanse))
                if should_run:
                    report("neural_cleanse", 0, 1)
                    result["neural_cleanse"] = self._run_neural_cleanse(
                        model, probe_images,
                        lambda d, t: report("neural_cleanse", d, t))
                else:
                    result["neural_cleanse"] = {"available": False,
                                                "reason": "Disabled by configuration"}
                    self.limitations.append("Neural Cleanse disabled by configuration")

                if bool(self.ac_cfg.get("enabled", True)):
                    report("activation_clustering", 0, 1)
                    result["activation_clustering"] = self._activation_clustering(
                        model, probe_images)
                    report("activation_clustering", 1, 1)
            else:
                self.limitations.append(
                    "No weights accessible: weight hashing, per-layer statistics, "
                    "Neural Cleanse and activation clustering are NOT AVAILABLE")

            # ---- BLACK BOX (always attempted when the model can run) ------
            if model.can_predict:
                report("blackbox", 0, 1)
                result["blackbox"] = self._blackbox_analysis(model, reference, probe_images)
                report("blackbox", 1, 1)
            else:
                self.limitations.append(
                    "Model cannot be executed: behavioural fingerprinting and output "
                    "distribution analysis are NOT AVAILABLE")

            payloads = [f.to_dict() for f in self.findings]
            result["findings"] = payloads
            result["summary"] = summarize_findings(payloads)
            result["confidence"] = self._assessment_confidence(model, result)
            result["limitations"] = list(dict.fromkeys(self.limitations))
            result["coverage_statement"] = self._coverage_statement(model, result)
            result["duration_seconds"] = round(time.time() - started, 2)

            LOGGER.info("Model audit complete: %s, %d finding(s), verdict=%s (%.1fs)",
                        result["access_level"], len(payloads),
                        result["summary"].get("verdict"), result["duration_seconds"])
            return result
        except Exception as exc:
            LOGGER.error("Model audit failed: %s", exc, exc_info=True)
            self.limitations.append(f"Audit aborted: {exc}")
            result["findings"] = [f.to_dict() for f in self.findings]
            result["summary"] = summarize_findings(result["findings"])
            result["limitations"] = list(dict.fromkeys(self.limitations))
            result["duration_seconds"] = round(time.time() - started, 2)
            return result

    # ------------------------------------------------------------------
    def _load_probe_images(self, directory: Optional[str | Path]) -> List[np.ndarray]:
        """Load reference images used as probes for clustering and behaviour tests.

        Args:
            directory: Directory of reference images.

        Returns:
            List of RGB arrays (empty when no directory was supplied).
        """
        images: List[np.ndarray] = []
        try:
            if not directory:
                return images
            limit = int(self.bb_cfg.get("canonical_samples", 100))
            for path in list_images(directory, limit=limit):
                image = load_image(path)
                if image is not None:
                    images.append(image)
            LOGGER.info("Loaded %d probe image(s) for model audit", len(images))
            return images
        except Exception as exc:
            LOGGER.error("_load_probe_images failed: %s", exc)
            return images

    # ------------------------------------------------------------------
    # (a) WEIGHT HASH + (b) WEIGHT STATISTICS
    # ------------------------------------------------------------------
    def _analyze_weights(self, model: LoadedModel,
                         reference: Optional[LoadedModel]) -> Dict[str, Any]:
        """Hash the weights and profile every layer's statistics.

        Args:
            model: The model under audit.
            reference: Optional trusted reference for substitution detection.

        Returns:
            Dictionary with ``weight_hash``, per-layer stats and flagged layers.
        """
        summary: Dict[str, Any] = {"available": True, "layers": [], "flagged_layers": [],
                                   "weight_hash": model.weight_hash}
        try:
            std_multiplier = float(self.weight_cfg.get("std_median_multiplier", 3.0))
            kurtosis_threshold = float(self.weight_cfg.get("kurtosis_threshold", 12.0))
            sparsity_threshold = float(self.weight_cfg.get("sparsity_threshold", 0.90))

            layers: List[Dict[str, Any]] = []
            for name, array in model.state_dict.items():
                try:
                    flat = np.asarray(array, dtype=np.float64).ravel()
                    if flat.size == 0:
                        continue
                    layers.append({
                        "layer": name,
                        "shape": list(np.asarray(array).shape),
                        "count": int(flat.size),
                        "mean": round(float(np.mean(flat)), 6),
                        "std": round(float(np.std(flat)), 6),
                        "min": round(float(np.min(flat)), 6),
                        "max": round(float(np.max(flat)), 6),
                        "sparsity": round(float(np.mean(np.abs(flat) < 1e-8)), 6),
                        "kurtosis": round(kurtosis(flat), 4),
                        "abs_max": round(float(np.max(np.abs(flat))), 6),
                    })
                except Exception as exc:
                    LOGGER.debug("Layer %s stats failed: %s", name, exc)

            summary["layers"] = layers
            summary["num_layers"] = len(layers)
            summary["total_parameters"] = int(sum(l["count"] for l in layers))

            if not layers:
                summary["available"] = False
                return summary

            stds = [l["std"] for l in layers]
            median_std = float(np.median(stds))
            summary["median_std"] = round(median_std, 6)
            summary["std_threshold"] = round(median_std * std_multiplier, 6)

            flagged: List[Dict[str, Any]] = []
            for layer in layers:
                reasons: List[str] = []
                evidence: Dict[str, Any] = {"layer": layer["layer"], "shape": layer["shape"]}

                if median_std > 0 and layer["std"] > median_std * std_multiplier:
                    ratio = layer["std"] / max(median_std, 1e-12)
                    reasons.append(
                        f"weight standard deviation {layer['std']:.5f} is {ratio:.1f}x the "
                        f"network median {median_std:.5f} (threshold {std_multiplier}x)")
                    evidence.update({"std": layer["std"], "median_std": median_std,
                                     "std_ratio": round(ratio, 3),
                                     "threshold": std_multiplier})

                if layer["kurtosis"] > kurtosis_threshold:
                    reasons.append(
                        f"excess kurtosis {layer['kurtosis']:.1f} exceeds {kurtosis_threshold} "
                        "— a small number of weights carry extreme magnitude, which is how a "
                        "trigger-sensitive pathway is typically implanted")
                    evidence.update({"kurtosis": layer["kurtosis"],
                                     "kurtosis_threshold": kurtosis_threshold})

                if layer["sparsity"] > sparsity_threshold and layer["count"] > 100:
                    reasons.append(
                        f"{layer['sparsity']:.1%} of weights are exactly zero "
                        f"(threshold {sparsity_threshold:.0%}) — consistent with a pruned or "
                        "substituted layer")
                    evidence.update({"sparsity": layer["sparsity"],
                                     "sparsity_threshold": sparsity_threshold})

                if not reasons:
                    continue

                confidence = min(0.88, 0.42 + 0.16 * len(reasons)
                                 + min(0.22, 0.02 * evidence.get("std_ratio", 0.0)))
                flagged.append({"layer": layer["layer"], "reasons": reasons,
                                "confidence": round(confidence, 4)})
                self.findings.append(Finding.create(
                    attack_class=AttackClass.WEIGHT_ANOMALY.value,
                    affected_asset=f"{Path(str(model.path)).name if model.path else 'model'}"
                                   f"::{layer['layer']}",
                    confidence=confidence,
                    reason=("Anomalous weight distribution in layer "
                            f"'{layer['layer']}': " + "; ".join(reasons) + "."),
                    evidence=evidence,
                    module=self.MODULE_NAME, detector="weight_statistics",
                    asset_type="model", prefix="MDL",
                ))

            summary["flagged_layers"] = flagged

            # -- substitution detection via weight hash --------------------
            if reference is not None and reference.weight_hash and model.weight_hash:
                match = reference.weight_hash == model.weight_hash
                summary["reference_match"] = match
                summary["reference_hash"] = reference.weight_hash
                if not match:
                    divergence = self._weight_divergence(model, reference)
                    summary["divergence"] = divergence
                    confidence = 0.93 if divergence.get("architecture_match") else 0.97
                    self.findings.append(Finding.create(
                        attack_class=AttackClass.MODEL_SUBSTITUTION.value,
                        affected_asset=str(model.path or "model"),
                        confidence=confidence,
                        reason=(
                            "Model weight hash does not match the trusted reference. "
                            f"Audited SHA-256 {short_hash(model.weight_hash, 16)} vs reference "
                            f"{short_hash(reference.weight_hash, 16)}. "
                            + (f"{divergence.get('changed_layers', 0)} of "
                               f"{divergence.get('total_layers', 0)} layers differ "
                               f"(max relative change {divergence.get('max_relative_change', 0):.3f}). "
                               if divergence.get("architecture_match")
                               else "The architectures also differ, so this is a different "
                                    "model entirely rather than a fine-tune. ")
                            + "The deployed model is not the one that was accredited."),
                        evidence={
                            "model_hash": model.weight_hash,
                            "reference_hash": reference.weight_hash,
                            "match": False,
                            **divergence,
                        },
                        module=self.MODULE_NAME, detector="weight_hash",
                        asset_type="model", prefix="MDL",
                        severity="CRITICAL", disposition="QUARANTINE",
                    ))
                else:
                    LOGGER.info("Weight hash matches reference — no substitution")
            elif reference is None:
                self.limitations.append(
                    "No reference model supplied: model-substitution detection NOT AVAILABLE "
                    "(the weight hash was still computed and can be registered as a baseline)")

            return summary
        except Exception as exc:
            LOGGER.error("_analyze_weights failed: %s", exc)
            summary["available"] = False
            summary["error"] = str(exc)
            self.limitations.append(f"Weight analysis error: {exc}")
            return summary

    def _weight_divergence(self, model: LoadedModel,
                           reference: LoadedModel) -> Dict[str, Any]:
        """Quantify how far a model's weights drifted from a reference.

        Args:
            model: Audited model.
            reference: Trusted reference model.

        Returns:
            Dictionary with changed-layer counts and the most-changed layers.
        """
        try:
            shared = set(model.state_dict) & set(reference.state_dict)
            if not shared:
                return {"architecture_match": False, "shared_layers": 0,
                        "total_layers": len(model.state_dict)}

            changes: List[Tuple[str, float]] = []
            for name in shared:
                a = np.asarray(model.state_dict[name], dtype=np.float64)
                b = np.asarray(reference.state_dict[name], dtype=np.float64)
                if a.shape != b.shape:
                    changes.append((name, 1.0))
                    continue
                denominator = float(np.linalg.norm(b)) or 1.0
                changes.append((name, float(np.linalg.norm(a - b) / denominator)))

            changes.sort(key=lambda c: c[1], reverse=True)
            changed = [c for c in changes if c[1] > 1e-6]
            return {
                "architecture_match": len(shared) == len(model.state_dict) == len(
                    reference.state_dict),
                "shared_layers": len(shared),
                "total_layers": len(model.state_dict),
                "changed_layers": len(changed),
                "max_relative_change": round(changes[0][1], 6) if changes else 0.0,
                "most_changed": [{"layer": n, "relative_change": round(v, 6)}
                                 for n, v in changes[:5]],
            }
        except Exception as exc:
            LOGGER.error("_weight_divergence failed: %s", exc)
            return {"architecture_match": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # (c) NEURAL CLEANSE
    # ------------------------------------------------------------------
    def _run_neural_cleanse(self, model: LoadedModel,
                            probe_images: List[np.ndarray],
                            progress: Optional[Callable[[int, int], None]] = None
                            ) -> Dict[str, Any]:
        """Reverse-engineer minimal triggers and flag anomalously small ones.

        Args:
            model: The model under audit.
            probe_images: Real images to optimise against (noise used if empty).
            progress: Optional ``callable(done, total)``.

        Returns:
            Neural Cleanse result dictionary.
        """
        try:
            input_size = int(self.nc_cfg.get("input_size", 32))
            channels = model.input_shape[0] if model.input_shape else 3
            probes = make_probe_batch(
                num_samples=16, channels=channels, size=input_size,
                images=probe_images or None, seed=int(self.cfg.get("runtime.seed", 1337)))

            # Probe validity gate. Neural Cleanse measures how large a
            # perturbation is needed to move probes into each class, so it is
            # only meaningful if the probes lie in the model's input
            # distribution. Probes the model cannot classify confidently sit in
            # a flat region of its decision surface where every class is equally
            # close, which compresses all mask norms together and destroys the
            # separation the MAD test depends on. This is silent and looks like
            # a clean result, so it must be checked and declared.
            probe_quality = self._probe_distribution_check(model, probes)
            if probe_quality.get("checked") and not probe_quality.get("in_distribution", True):
                self.limitations.append(
                    "Neural Cleanse probe set appears OUT OF DISTRIBUTION for this model "
                    f"(mean top-class probability {probe_quality['mean_max_probability']:.2f}, "
                    f"predictions spread over {probe_quality['distinct_predictions']} of "
                    f"{probe_quality['num_classes']} classes). Trigger-size comparisons across "
                    "classes are unreliable; supply reference images drawn from the model's "
                    "own operational domain.")

            cleanse = NeuralCleanse(
                model,
                epochs=int(self.nc_cfg.get("epochs", 200)),
                lr=float(self.nc_cfg.get("lr", 0.1)),
                l1_lambda=float(self.nc_cfg.get("l1_lambda", 0.01)),
                input_size=input_size,
                mad_threshold=float(self.nc_cfg.get("mad_threshold", 2.0)),
                attack_success_target=float(self.nc_cfg.get("attack_success_target", 0.90)),
                early_stop_patience=int(self.nc_cfg.get("early_stop_patience", 30)),
                size_ratio_threshold=float(self.nc_cfg.get("size_ratio_threshold", 0.34)),
                max_trigger_coverage=float(self.nc_cfg.get("max_trigger_coverage", 0.06)),
                seed=int(self.cfg.get("runtime.seed", 1337)),
            )
            outcome = cleanse.scan(
                probes, num_classes=model.num_classes,
                max_classes=int(self.nc_cfg.get("max_classes", 10)),
                progress_callback=progress)

            outcome["probe_quality"] = probe_quality
            self.limitations.extend(outcome.get("limitations", []))
            if not outcome.get("available"):
                if outcome.get("reason"):
                    self.limitations.append(f"Neural Cleanse: {outcome['reason']}")
                return outcome

            for flagged in outcome.get("flagged_classes", []):
                index = float(flagged["anomaly_index"])
                success = float(flagged["attack_success_rate"])
                confidence = float(np.clip(0.45 + 0.14 * (index - 2.0) + 0.25 * success,
                                           0.0, 0.96))
                location = flagged.get("trigger_location", {}).get("quadrant", "unknown")
                self.findings.append(Finding.create(
                    attack_class=AttackClass.MODEL_BACKDOOR.value,
                    affected_asset=f"{Path(str(model.path)).name if model.path else 'model'}"
                                   f"::class_{flagged['target_class']}",
                    confidence=confidence,
                    reason=(
                        f"Neural Cleanse reverse-engineered a trigger for class "
                        f"{flagged['target_class']} whose mask is anomalously small: L1 norm "
                        f"{flagged['mask_l1']:.2f} versus a median of "
                        f"{flagged['median_mask_l1']:.2f} across scanned classes "
                        f"({flagged['size_ratio']:.2f}x), giving a MAD anomaly index of "
                        f"{index:.2f} against a threshold of "
                        f"{outcome.get('threshold', 2.0)}. The recovered patch flips "
                        f"{success:.0%} of probe inputs to that class and is concentrated at "
                        f"the {location}. A clean class requires a large perturbation to be "
                        "universally reachable; a class reachable via a tiny fixed patch is "
                        "the signature of an implanted backdoor."),
                    evidence={
                        "target_class": flagged["target_class"],
                        "anomaly_index": index,
                        "threshold": outcome.get("threshold", 2.0),
                        "mask_l1": flagged["mask_l1"],
                        "median_mask_l1": flagged["median_mask_l1"],
                        "size_ratio": flagged["size_ratio"],
                        "attack_success_rate": success,
                        "trigger_location": flagged.get("trigger_location", {}),
                        "classes_scanned": outcome.get("classes_scanned", 0),
                        "mode": outcome.get("mode", ""),
                    },
                    module=self.MODULE_NAME, detector="neural_cleanse",
                    asset_type="model", prefix="MDL",
                ))
            return outcome
        except Exception as exc:
            LOGGER.error("_run_neural_cleanse failed: %s", exc)
            self.limitations.append(f"Neural Cleanse error: {exc}")
            return {"available": False, "reason": str(exc)}

    def _probe_distribution_check(self, model: LoadedModel,
                                  probes: np.ndarray) -> Dict[str, Any]:
        """Assess whether the probe batch lies in the model's input distribution.

        Args:
            model: The model under audit.
            probes: The probe batch about to be used for trigger recovery.

        Returns:
            Dictionary with ``checked``, ``in_distribution`` and the supporting
            statistics.
        """
        summary: Dict[str, Any] = {"checked": False, "in_distribution": True}
        try:
            if not model.can_predict:
                return summary
            logits = model.predict(probes)
            if logits is None:
                return summary
            logits = np.asarray(logits, dtype=np.float64)
            if logits.ndim != 2 or logits.shape[0] == 0:
                return summary

            probabilities = softmax(logits)
            mean_max = float(np.mean(probabilities.max(axis=1)))
            predictions = np.argmax(logits, axis=1)
            distinct = int(np.unique(predictions).size)
            num_classes = int(logits.shape[1])
            uniform = 1.0 / max(num_classes, 1)

            summary.update({
                "checked": True,
                "mean_max_probability": round(mean_max, 4),
                "distinct_predictions": distinct,
                "num_classes": num_classes,
                "uniform_baseline": round(uniform, 4),
                "in_distribution": bool(mean_max >= min(0.5, uniform * 2.0)),
            })
            return summary
        except Exception as exc:
            LOGGER.error("_probe_distribution_check failed: %s", exc)
            return summary

    # ------------------------------------------------------------------
    # (d) ACTIVATION CLUSTERING
    # ------------------------------------------------------------------
    def _activation_clustering(self, model: LoadedModel,
                               probe_images: List[np.ndarray]) -> Dict[str, Any]:
        """Cluster per-class activations and flag imbalanced splits.

        Args:
            model: The model under audit.
            probe_images: Reference images to push through the model.

        Returns:
            Clustering result per predicted class.
        """
        summary: Dict[str, Any] = {"available": False, "classes": {}, "reason": ""}
        try:
            if not model.can_predict:
                summary["reason"] = ("Model cannot be executed: activation clustering "
                                     "NOT AVAILABLE")
                self.limitations.append(summary["reason"])
                return summary
            if not probe_images:
                summary["reason"] = ("No reference images supplied: activation clustering "
                                     "NOT AVAILABLE (pass --reference-images)")
                self.limitations.append(summary["reason"])
                return summary

            from src.utils.image_utils import normalize_imagenet, resize_image
            size = model.input_shape[1] if model.input_shape else 32
            batch = np.stack([
                normalize_imagenet(resize_image(image, (size, size)))
                for image in probe_images
            ]).astype(np.float32)

            logits = model.predict(batch)
            if logits is None:
                summary["reason"] = "Model returned no output: activation clustering failed"
                self.limitations.append(summary["reason"])
                return summary

            logits = np.asarray(logits)
            if logits.ndim != 2:
                summary["reason"] = f"Unexpected output shape {logits.shape}"
                return summary

            predictions = np.argmax(logits, axis=1)
            imbalance_threshold = float(self.ac_cfg.get("imbalance_threshold", 0.70))
            min_samples = int(self.ac_cfg.get("min_samples_per_class", 10))

            for class_id in np.unique(predictions):
                positions = np.where(predictions == class_id)[0]
                if positions.size < min_samples:
                    summary["classes"][str(int(class_id))] = {
                        "available": False,
                        "reason": f"only {positions.size} sample(s); need {min_samples}",
                    }
                    continue

                outcome = activation_cluster_imbalance(
                    logits[positions],
                    n_clusters=int(self.ac_cfg.get("n_clusters", 2)),
                    pca_components=int(self.ac_cfg.get("pca_components", 10)),
                    seed=int(self.cfg.get("runtime.seed", 1337)),
                )
                summary["classes"][str(int(class_id))] = outcome
                if not outcome.get("available"):
                    continue

                ratio = float(outcome["imbalance_ratio"])
                if ratio >= imbalance_threshold:
                    silhouette = float(outcome.get("silhouette", 0.0))
                    confidence = float(np.clip(
                        0.35 + 0.9 * (ratio - imbalance_threshold) + 0.3 * max(silhouette, 0.0),
                        0.0, 0.90))
                    self.findings.append(Finding.create(
                        attack_class=AttackClass.ACTIVATION_ANOMALY.value,
                        affected_asset=f"{Path(str(model.path)).name if model.path else 'model'}"
                                       f"::class_{int(class_id)}",
                        confidence=confidence,
                        reason=(
                            f"Activations for predicted class {int(class_id)} split into two "
                            f"clusters of sizes {outcome['cluster_sizes']} — a "
                            f"{ratio:.0%}/{1 - ratio:.0%} imbalance exceeding the "
                            f"{imbalance_threshold:.0%} threshold, with silhouette "
                            f"{silhouette:.2f}. Clean classes form a single activation cloud; "
                            "a distinct minority cluster is the classic signature of poisoned "
                            "samples being routed through a separate internal pathway."),
                        evidence={
                            "predicted_class": int(class_id),
                            "cluster_sizes": outcome["cluster_sizes"],
                            "imbalance_ratio": ratio,
                            "threshold": imbalance_threshold,
                            "silhouette": silhouette,
                            "minority_fraction": outcome.get("minority_fraction", 0.0),
                            "n_samples": outcome.get("n_samples", 0),
                        },
                        module=self.MODULE_NAME, detector="activation_clustering",
                        asset_type="model", prefix="MDL",
                    ))

            summary["available"] = bool(summary["classes"])
            return summary
        except Exception as exc:
            LOGGER.error("_activation_clustering failed: %s", exc)
            summary["reason"] = str(exc)
            self.limitations.append(f"Activation clustering error: {exc}")
            return summary

    # ------------------------------------------------------------------
    # BLACK-BOX ANALYSIS
    # ------------------------------------------------------------------
    def _blackbox_analysis(self, model: LoadedModel,
                           reference: Optional[LoadedModel],
                           probe_images: List[np.ndarray]) -> Dict[str, Any]:
        """Run behavioural fingerprinting and output-distribution checks.

        Args:
            model: The model under audit.
            reference: Optional reference model to compare predictions against.
            probe_images: Canonical probe images.

        Returns:
            Black-box analysis dictionary.
        """
        summary: Dict[str, Any] = {"available": False, "fingerprint": {},
                                   "output_distribution": {}, "reason": ""}
        try:
            count = int(self.bb_cfg.get("canonical_samples", 100))
            size = model.input_shape[1] if model.input_shape else 32
            channels = model.input_shape[0] if model.input_shape else 3

            batch = make_probe_batch(
                num_samples=count, channels=channels, size=size,
                images=probe_images or None, seed=int(self.cfg.get("runtime.seed", 1337)))

            logits = model.predict(batch)
            if logits is None:
                summary["reason"] = "Model produced no output on the canonical probe set"
                self.limitations.append(summary["reason"])
                return summary
            logits = np.asarray(logits, dtype=np.float64)
            summary["available"] = True
            summary["num_probes"] = int(batch.shape[0])

            # -- behavioural fingerprinting --------------------------------
            if reference is not None and reference.can_predict:
                reference_logits = reference.predict(batch)
                if reference_logits is not None:
                    reference_logits = np.asarray(reference_logits, dtype=np.float64)
                    if reference_logits.shape == logits.shape:
                        predictions = np.argmax(logits, axis=1)
                        reference_predictions = np.argmax(reference_logits, axis=1)
                        mismatch = float(np.mean(predictions != reference_predictions))
                        threshold = float(self.bb_cfg.get("mismatch_rate_threshold", 0.05))
                        divergence = float(np.mean(np.abs(
                            softmax(logits) - softmax(reference_logits))))
                        summary["fingerprint"] = {
                            "mismatch_rate": round(mismatch, 4),
                            "threshold": threshold,
                            "mean_probability_divergence": round(divergence, 5),
                            "num_probes": int(batch.shape[0]),
                        }
                        if mismatch > threshold:
                            confidence = float(np.clip(0.45 + 2.0 * (mismatch - threshold),
                                                       0.0, 0.94))
                            self.findings.append(Finding.create(
                                attack_class=AttackClass.BEHAVIORAL_DEVIATION.value,
                                affected_asset=str(model.path or "model"),
                                confidence=confidence,
                                reason=(
                                    f"Behavioural fingerprint diverges from the trusted "
                                    f"reference: {mismatch:.1%} of {int(batch.shape[0])} "
                                    f"canonical probes receive a different predicted class "
                                    f"(threshold {threshold:.0%}), with mean probability "
                                    f"divergence {divergence:.4f}. The deployed model does not "
                                    "behave like the accredited one."),
                                evidence=summary["fingerprint"],
                                module=self.MODULE_NAME,
                                detector="behavioral_fingerprint",
                                asset_type="model", prefix="MDL",
                            ))
                    else:
                        summary["fingerprint"] = {
                            "available": False,
                            "reason": (f"Output shapes differ: {logits.shape} vs "
                                       f"{reference_logits.shape} — the models are not "
                                       "comparable"),
                        }
                        self.limitations.append(summary["fingerprint"]["reason"])
            else:
                self.limitations.append(
                    "No runnable reference model: behavioural fingerprinting NOT AVAILABLE")

            # -- output distribution shape ---------------------------------
            probabilities = softmax(logits)
            entropies = np.array([normalized_entropy(row) for row in probabilities])
            max_probabilities = probabilities.max(axis=1)
            sharp_threshold = float(self.bb_cfg.get("entropy_sharp_threshold", 0.15))
            flat_threshold = float(self.bb_cfg.get("entropy_flat_threshold", 0.90))
            mean_entropy = float(np.mean(entropies))

            distribution = {
                "mean_normalized_entropy": round(mean_entropy, 4),
                "median_max_probability": round(float(np.median(max_probabilities)), 4),
                "sharp_threshold": sharp_threshold,
                "flat_threshold": flat_threshold,
                "entropy_stats": robust_stats(entropies),
                "shape": ("SHARP" if mean_entropy < sharp_threshold
                          else "FLAT" if mean_entropy > flat_threshold else "NORMAL"),
            }
            summary["output_distribution"] = distribution

            if distribution["shape"] != "NORMAL":
                is_sharp = distribution["shape"] == "SHARP"
                confidence = 0.55 if is_sharp else 0.48
                self.findings.append(Finding.create(
                    attack_class=AttackClass.BEHAVIORAL_DEVIATION.value,
                    affected_asset=str(model.path or "model"),
                    confidence=confidence,
                    reason=(
                        f"Output confidence distribution is anomalously "
                        f"{'sharp' if is_sharp else 'flat'}: mean normalised entropy "
                        f"{mean_entropy:.3f} "
                        f"{'<' if is_sharp else '>'} threshold "
                        f"{sharp_threshold if is_sharp else flat_threshold}. "
                        + ("Near-one-hot outputs on arbitrary probes indicate an "
                           "over-confident or deliberately hardened model, and can mask a "
                           "backdoor's activation from confidence-based monitoring."
                           if is_sharp else
                           "Near-uniform outputs indicate the model is not discriminating, "
                           "which may mean a substituted, corrupted or untrained model.")),
                    evidence=distribution,
                    module=self.MODULE_NAME, detector="output_distribution",
                    asset_type="model", prefix="MDL",
                ))
            return summary
        except Exception as exc:
            LOGGER.error("_blackbox_analysis failed: %s", exc)
            summary["reason"] = str(exc)
            self.limitations.append(f"Black-box analysis error: {exc}")
            return summary

    # ------------------------------------------------------------------
    def _assessment_confidence(self, model: LoadedModel,
                               result: Dict[str, Any]) -> float:
        """Score how much confidence the analyst should place in this audit.

        A clean result from a black-box audit is far weaker evidence of safety
        than a clean result from a full white-box audit, and the report must say
        so explicitly rather than implying equivalent assurance.

        Args:
            model: The audited model.
            result: The assembled audit result.

        Returns:
            Confidence in the completeness of the audit, in ``[0, 1]``.
        """
        try:
            score = 0.0
            if model.state_dict:
                score += 0.30
            if result.get("neural_cleanse", {}).get("available"):
                score += 0.30
                if result["neural_cleanse"].get("mode") == "gradient_free":
                    score -= 0.12
            if result.get("activation_clustering", {}).get("available"):
                score += 0.15
            if result.get("blackbox", {}).get("fingerprint", {}).get("mismatch_rate") is not None:
                score += 0.15
            if result.get("weight_statistics", {}).get("available"):
                score += 0.10
            return round(float(np.clip(score, 0.05, 0.98)), 3)
        except Exception:
            return 0.0

    def _coverage_statement(self, model: LoadedModel, result: Dict[str, Any]) -> str:
        """Compose the plain-language statement of what this audit did and did not cover.

        Args:
            model: The audited model.
            result: The assembled audit result.

        Returns:
            Human-readable coverage sentence.
        """
        try:
            level = model.access_level.value
            ran: List[str] = []
            if result.get("weight_statistics", {}).get("available"):
                ran.append("weight hashing and per-layer statistics")
            if result.get("neural_cleanse", {}).get("available"):
                mode = result["neural_cleanse"].get("mode", "")
                ran.append("Neural Cleanse trigger recovery"
                           + (" (gradient-free approximation)"
                              if mode == "gradient_free" else ""))
            if result.get("activation_clustering", {}).get("available"):
                ran.append("activation clustering")
            if result.get("blackbox", {}).get("available"):
                ran.append("black-box behavioural analysis")

            confidence = result.get("confidence", 0.0)
            verdict = result.get("summary", {}).get("verdict", "UNKNOWN")
            statement = (f"Access level {level}. Checks performed: "
                         f"{', '.join(ran) if ran else 'none'}. "
                         f"Audit completeness {confidence:.0%}. ")
            if verdict == "CLEAN" and confidence < 0.6:
                statement += (
                    "NOTE: no findings were raised, but coverage was incomplete — this is "
                    "absence of evidence, not evidence of absence. Do not treat this model "
                    "as accredited on the strength of this audit alone.")
            elif verdict == "CLEAN":
                statement += ("No integrity findings at this access level.")
            else:
                statement += f"Verdict: {verdict}."
            return statement
        except Exception as exc:
            return f"Coverage statement unavailable: {exc}"


def audit_model(model_path: str | Path,
                reference_path: Optional[str | Path] = None,
                reference_images: Optional[str | Path] = None) -> Dict[str, Any]:
    """Convenience wrapper running a full model audit.

    Args:
        model_path: Model file to audit.
        reference_path: Optional trusted reference model.
        reference_images: Optional directory of probe images.

    Returns:
        Audit result dictionary.
    """
    try:
        return ModelIntegrityEngine().audit(model_path, reference_path=reference_path,
                                            reference_images=reference_images)
    except Exception as exc:
        LOGGER.error("audit_model failed: %s", exc)
        return {"module": "model_auditor", "findings": [], "summary": summarize_findings([]),
                "limitations": [str(exc)], "access_level": AccessLevel.UNAVAILABLE.value}


__all__ = ["ModelIntegrityEngine", "audit_model"]
