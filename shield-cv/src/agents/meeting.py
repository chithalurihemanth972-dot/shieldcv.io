"""Meeting room: cross-contributor comparison and the office's final verdict.

Individual agents only see their own contributor. The meeting room is where
their reports are laid side by side, because several important signals only
exist *between* contributors:

* **Disparity** — one contributor whose risk stands well clear of its peers.
* **Class targeting** — several contributors independently attacking the same
  label, which is far stronger evidence of coordination than either alone.
* **Trust inversion** — a nominally TRUSTED source behaving worse than an
  UNVERIFIED one, which inverts the procurement assumption.
* **Model disagreement** — candidate models that diverge on identical inputs,
  meaning at least one has been altered.

Two statistical hazards are handled explicitly. First, a z-score over a handful
of contributors is bounded: with ``n`` contributors the largest attainable
absolute z-score is ``(n-1)/sqrt(n)``, which is only 1.41 for three
contributors. A threshold above that bound can never fire, so the attainable
maximum is checked and a median-ratio rule is used as the operative test, with
the ceiling recorded as a limitation. Second, the mean and standard deviation
are themselves dragged by the outlier being hunted, so comparisons are made
against the *median* and a MAD-based scale.
"""

from __future__ import annotations

import math
import statistics
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.agents.agent import AgentReport
from src.config import get_config
from src.reporting.schema import AttackClass, Finding, reset_finding_ids, summarize_findings
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

# Ordering used to detect trust inversion: a higher number means the
# contributor was vetted more thoroughly and is therefore held to a
# correspondingly higher standard.
_TRUST_RANK = {"UNVERIFIED": 0, "VERIFIED": 1, "TRUSTED": 2}

# A contributor must look this much worse than the peer median before disparity
# is claimed, and must clear this absolute risk floor. The floor stops a
# cosmetic gap between three near-clean contributors (0.02 vs 0.01 is a 2x
# ratio) from being escalated as an outlier.
_DISPARITY_RATIO = 1.8
_DISPARITY_FLOOR = 0.45


def _normalise_class(label: Any) -> str:
    """Reduce a class label to a canonical bare identifier.

    Args:
        label: Raw class label from a finding's evidence.

    Returns:
        Canonical label string, e.g. both ``"class_2"`` and ``2`` yield ``"2"``.
    """
    try:
        text = str(label).strip()
        for prefix in ("class_", "class-", "label_", "label-"):
            if text.lower().startswith(prefix):
                return text[len(prefix):]
        return text
    except Exception:  # pragma: no cover - defensive
        return str(label)


class MeetingRoom:
    """Compares agent reports and issues the office's consolidated verdict.

    Attributes:
        cfg: Loaded configuration object.
        zscore_threshold: Configured disparity z-score threshold.
        disagreement_threshold: Prediction mismatch rate that flags models as
            disagreeing.
    """

    def __init__(self) -> None:
        """Initialise the meeting room from configuration."""
        self.cfg = get_config()
        try:
            section = self.cfg.section("agents") or {}
        except Exception as exc:
            LOGGER.error("agents config unavailable, using defaults: %s", exc)
            section = {}
        self.zscore_threshold = float(section.get("disparity_zscore", 1.5) or 1.5)
        self.disagreement_threshold = float(
            section.get("model_disagreement_threshold", 0.2) or 0.2)

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------
    def convene(self, office_result: Dict[str, Any],
                models: Optional[Sequence[str | Path]] = None,
                num_probes: int = 100,
                reference_images: Optional[Sequence[str | Path]] = None
                ) -> Dict[str, Any]:
        """Run the full cross-contributor meeting.

        Args:
            office_result: Result dictionary from
                :meth:`src.agents.manager.OfficeManager.run`.
            models: Optional model paths to cross-check for disagreement.
            num_probes: Number of canonical probes for the model comparison.

        Returns:
            Dictionary with ``module``, ``participants``, ``findings``,
            ``summary``, ``disparity``, ``class_targeting``,
            ``trust_inversion``, ``model_disagreement``, ``verdict``,
            ``narrative``, ``recommendations``, ``limitations`` and
            ``duration_seconds``.
        """
        started = time.time()
        reset_finding_ids("MEET")
        result: Dict[str, Any] = {
            "module": "meeting_room",
            "participants": [],
            "findings": [],
            "summary": {},
            "disparity": {},
            "class_targeting": {},
            "trust_inversion": {},
            "model_disagreement": {},
            "verdict": "UNKNOWN",
            "narrative": "",
            "recommendations": [],
            "limitations": [],
            "duration_seconds": 0.0,
        }

        try:
            reports = self._usable_reports(office_result, result)
            result["participants"] = [
                {"agent_id": r.agent_id, "contributor": r.contributor,
                 "trust_level": r.trust_level, "registered": r.registered,
                 "risk_score": r.risk_score, "verdict": r.verdict,
                 "num_samples": r.num_samples, "num_findings": r.num_findings,
                 "status": r.status}
                for r in reports
            ]

            findings: List[Dict[str, Any]] = []

            if len(reports) < 2:
                result["limitations"].append(
                    f"Cross-contributor comparison needs at least 2 usable contributors; "
                    f"{len(reports)} available. Disparity, targeting and trust-inversion "
                    f"checks were not performed.")
            else:
                for block, key in ((self._disparity, "disparity"),
                                   (self._class_targeting, "class_targeting"),
                                   (self._trust_inversion, "trust_inversion")):
                    try:
                        analysis = block(reports)
                        result[key] = analysis
                        findings.extend(analysis.pop("findings", []))
                        result["limitations"].extend(analysis.pop("limitations", []))
                    except Exception as exc:
                        message = f"{key} analysis failed: {type(exc).__name__}: {exc}"
                        result["limitations"].append(message)
                        LOGGER.error("%s\n%s", message, traceback.format_exc())

            if models:
                try:
                    analysis = self._model_disagreement(
                        list(models), num_probes, reference_images)
                    result["model_disagreement"] = analysis
                    findings.extend(analysis.pop("findings", []))
                    result["limitations"].extend(analysis.pop("limitations", []))
                except Exception as exc:
                    message = f"model disagreement analysis failed: {type(exc).__name__}: {exc}"
                    result["limitations"].append(message)
                    LOGGER.error("%s\n%s", message, traceback.format_exc())
            else:
                result["limitations"].append(
                    "No candidate models supplied; model disagreement was not assessed.")

            result["findings"] = findings
            result["summary"] = summarize_findings(findings)
            verdict = self._verdict(reports, findings)
            result.update(verdict)

            LOGGER.info("meeting: %d participant(s), %d finding(s), verdict=%s",
                        len(reports), len(findings), result["verdict"])

        except Exception as exc:
            result["limitations"].append(f"Meeting failed: {type(exc).__name__}: {exc}")
            LOGGER.error("meeting failed: %s\n%s", exc, traceback.format_exc())

        result["duration_seconds"] = round(time.time() - started, 3)
        return result

    # ------------------------------------------------------------------
    # participants
    # ------------------------------------------------------------------
    @staticmethod
    def _usable_reports(office_result: Dict[str, Any],
                        result: Dict[str, Any]) -> List[AgentReport]:
        """Convert raw office output into the reports eligible for comparison.

        Agents that errored or found no samples are excluded from the
        statistics — averaging in a contributor that was never actually scanned
        would understate the peer baseline and hide a real outlier — but their
        exclusion is recorded as a coverage limitation.

        Args:
            office_result: Office run output.
            result: Meeting result being built, mutated to add limitations.

        Returns:
            List of usable :class:`AgentReport` objects.
        """
        usable: List[AgentReport] = []
        for payload in office_result.get("reports") or []:
            report = (payload if isinstance(payload, AgentReport)
                      else AgentReport.from_dict(payload))
            if report.status == "OK":
                usable.append(report)
            else:
                result["limitations"].append(
                    f"Contributor '{report.contributor}' excluded from cross-contributor "
                    f"comparison (status={report.status}"
                    + (f": {report.error}" if report.error else "") + ").")
        return usable

    # ------------------------------------------------------------------
    # disparity
    # ------------------------------------------------------------------
    def _disparity(self, reports: List[AgentReport]) -> Dict[str, Any]:
        """Flag contributors whose risk stands clear of their peers.

        Args:
            reports: Usable agent reports.

        Returns:
            Analysis dictionary including any findings raised.
        """
        findings: List[Dict[str, Any]] = []
        limitations: List[str] = []
        risks = [float(r.risk_score) for r in reports]
        n = len(risks)

        median = float(statistics.median(risks))
        mean = float(statistics.fmean(risks))
        spread = float(statistics.pstdev(risks))
        # Median absolute deviation, scaled to be comparable with a standard
        # deviation for normal data. Robust to the very outlier being sought.
        mad = float(statistics.median([abs(x - median) for x in risks])) * 1.4826

        # Hard ceiling on any z-score computable from n points.
        attainable = (n - 1) / math.sqrt(n) if n > 1 else 0.0
        if attainable < self.zscore_threshold:
            limitations.append(
                f"With {n} contributors the largest attainable disparity z-score is "
                f"{attainable:.2f}, below the configured threshold of "
                f"{self.zscore_threshold:.2f}; the z-test cannot fire. A robust "
                f"median-ratio rule (>{_DISPARITY_RATIO:.1f}x peer median and risk "
                f">{_DISPARITY_FLOOR:.2f}) was used instead.")

        outliers: List[Dict[str, Any]] = []
        for report in reports:
            risk = float(report.risk_score)
            peers = [float(r.risk_score) for r in reports if r.contributor != report.contributor]
            peer_median = float(statistics.median(peers)) if peers else 0.0
            zscore = (risk - mean) / spread if spread > 1e-9 else 0.0
            robust_z = (risk - median) / mad if mad > 1e-9 else 0.0
            ratio = (risk / peer_median) if peer_median > 1e-9 else (
                float("inf") if risk > _DISPARITY_FLOOR else 0.0)

            by_ratio = ratio >= _DISPARITY_RATIO and risk >= _DISPARITY_FLOOR
            by_z = zscore >= self.zscore_threshold and risk >= _DISPARITY_FLOOR
            if not (by_ratio or by_z):
                continue

            # Confidence grows with how far clear of the peers the contributor
            # sits, but is capped: with a handful of peers the estimate of
            # "normal" is itself weak, and certainty is never warranted.
            margin = min(1.0, (risk - peer_median) / max(peer_median, 0.15))
            confidence = min(0.90, 0.45 + 0.35 * margin + (0.10 if by_z else 0.0))

            entry = {
                "contributor": report.contributor,
                "risk_score": round(risk, 4),
                "peer_median": round(peer_median, 4),
                "ratio": round(ratio, 3) if math.isfinite(ratio) else None,
                "zscore": round(zscore, 3),
                "robust_zscore": round(robust_z, 3),
                "triggered_by": "median_ratio" if by_ratio else "zscore",
                "confidence": round(confidence, 3),
            }
            outliers.append(entry)

            finding = Finding.create(
                attack_class=AttackClass.CROSS_CONTRIBUTOR_DISPARITY.value,
                affected_asset=report.contributor,
                asset_type="contributor",
                confidence=confidence,
                reason=(
                    f"Contributor '{report.contributor}' carries risk {risk:.2f} against a "
                    f"peer median of {peer_median:.2f} ({ratio:.1f}x) across {n} contributors. "
                    f"Its integrity profile is materially worse than the rest of the office, "
                    f"which points to a source-specific problem rather than a dataset-wide one."),
                evidence=entry | {
                    "peer_risks": {r.contributor: round(float(r.risk_score), 4)
                                   for r in reports},
                    "attack_classes": report.attack_classes,
                    "num_findings": report.num_findings,
                    "num_samples": report.num_samples,
                    "attainable_max_zscore": round(attainable, 3),
                },
                module="meeting_room",
                detector="cross_contributor_disparity",
                contributor=report.contributor,
                prefix="MEET",
            )
            findings.append(finding.to_dict())

        return {
            "contributors": n,
            "median_risk": round(median, 4),
            "mean_risk": round(mean, 4),
            "stdev_risk": round(spread, 4),
            "mad_risk": round(mad, 4),
            "attainable_max_zscore": round(attainable, 3),
            "zscore_threshold": self.zscore_threshold,
            "zscore_usable": attainable >= self.zscore_threshold,
            "outliers": outliers,
            "findings": findings,
            "limitations": limitations,
        }

    # ------------------------------------------------------------------
    # class targeting
    # ------------------------------------------------------------------
    def _class_targeting(self, reports: List[AgentReport]) -> Dict[str, Any]:
        """Detect the same label being attacked from multiple contributors.

        Args:
            reports: Usable agent reports.

        Returns:
            Analysis dictionary including any findings raised.
        """
        findings: List[Dict[str, Any]] = []
        limitations: List[str] = []

        # Only poisoning classes carry a meaningful target label; duplicates
        # and OOD samples do not indicate an intended victim class.
        targeting_classes = {
            AttackClass.LABEL_FLIPPING.value, AttackClass.TRIGGER_INJECTION.value,
            AttackClass.MODEL_BACKDOOR.value, AttackClass.SYSTEMATIC_MISLABELING.value,
        }

        per_class: Dict[str, Dict[str, Any]] = {}
        for report in reports:
            for finding in report.findings:
                if str(finding.get("attack_class")) not in targeting_classes:
                    continue
                # Only corroborated findings count: a single weak detector hit
                # replicated across contributors is coincidence, not a campaign.
                if float(finding.get("confidence", 0.0) or 0.0) < 0.50:
                    continue
                evidence = finding.get("evidence") or {}
                # Evidence key names differ per detector. For a label flip the
                # targeted class is the one samples were flipped *into*
                # (`assigned_class`); for a spectral trigger hit it is the
                # class the signature concentrates in.
                label = None
                for key in ("target_class", "assigned_class", "spectral_class",
                            "assigned_label", "label", "class"):
                    if evidence.get(key) is not None:
                        label = evidence[key]
                        break
                if label is None:
                    continue
                # Detectors name the same class differently ("class_2" from the
                # label-flip detector, 2 from the spectral detector). Normalise
                # to a bare identifier so the two are recognised as one class.
                key = _normalise_class(label)
                bucket = per_class.setdefault(
                    key, {"contributors": {}, "finding_ids": [], "confidences": []})
                bucket["contributors"].setdefault(report.contributor, 0)
                bucket["contributors"][report.contributor] += 1
                bucket["finding_ids"].append(finding.get("finding_id"))
                bucket["confidences"].append(float(finding.get("confidence", 0.0) or 0.0))

        targeted: List[Dict[str, Any]] = []
        for label, bucket in sorted(per_class.items()):
            contributors = bucket["contributors"]
            if len(contributors) < 2:
                continue
            mean_conf = float(statistics.fmean(bucket["confidences"]))
            # Confidence rises with the number of independent contributors
            # converging on one label, capped short of certainty.
            confidence = min(0.92, 0.40 + 0.18 * len(contributors) + 0.25 * (mean_conf - 0.5))
            confidence = max(0.40, confidence)
            entry = {
                "target_class": label,
                "contributors": contributors,
                "num_contributors": len(contributors),
                "total_findings": sum(contributors.values()),
                "mean_confidence": round(mean_conf, 3),
                "confidence": round(confidence, 3),
            }
            targeted.append(entry)

            finding = Finding.create(
                attack_class=AttackClass.COORDINATED_ATTACK.value,
                affected_asset=f"class:{label}",
                asset_type="class",
                confidence=confidence,
                reason=(
                    f"Class '{label}' is targeted by poisoning findings from "
                    f"{len(contributors)} independent contributors "
                    f"({', '.join(sorted(contributors))}). Independent sources converging on "
                    f"one label is a hallmark of a coordinated campaign rather than "
                    f"unrelated collection errors."),
                evidence=entry | {"evidence_finding_ids": bucket["finding_ids"][:25]},
                module="meeting_room",
                detector="cross_contributor_class_targeting",
                prefix="MEET",
            )
            findings.append(finding.to_dict())

        if not per_class:
            limitations.append(
                "No poisoning findings carried an identifiable target class; "
                "cross-contributor class targeting could not be assessed.")

        return {"targeted_classes": targeted, "classes_examined": len(per_class),
                "findings": findings, "limitations": limitations}

    # ------------------------------------------------------------------
    # trust inversion
    # ------------------------------------------------------------------
    def _trust_inversion(self, reports: List[AgentReport]) -> Dict[str, Any]:
        """Flag vetted contributors performing worse than unvetted peers.

        Args:
            reports: Usable agent reports.

        Returns:
            Analysis dictionary including any findings raised.
        """
        findings: List[Dict[str, Any]] = []
        limitations: List[str] = []
        inversions: List[Dict[str, Any]] = []

        ranked = [(r, _TRUST_RANK.get(str(r.trust_level).upper(), 0)) for r in reports]
        if len({rank for _, rank in ranked}) < 2:
            limitations.append(
                "All contributors share one trust level; trust inversion was not assessed.")
            return {"inversions": [], "findings": [], "limitations": limitations}

        for report, rank in ranked:
            lower = [(r, k) for r, k in ranked if k < rank]
            if not lower:
                continue
            best_lower = min(float(r.risk_score) for r, _ in lower)
            risk = float(report.risk_score)
            # Require both a clear absolute gap and a material risk level, so a
            # trivial ordering wobble among clean contributors is not escalated.
            if risk - best_lower < 0.15 or risk < _DISPARITY_FLOOR:
                continue

            confidence = min(0.85, 0.45 + (risk - best_lower))
            entry = {
                "contributor": report.contributor,
                "trust_level": report.trust_level,
                "risk_score": round(risk, 4),
                "best_lower_trust_risk": round(best_lower, 4),
                "gap": round(risk - best_lower, 4),
                "confidence": round(confidence, 3),
            }
            inversions.append(entry)

            finding = Finding.create(
                attack_class=AttackClass.CONTRIBUTOR_RISK.value,
                affected_asset=report.contributor,
                asset_type="contributor",
                confidence=confidence,
                reason=(
                    f"Trust inversion: '{report.contributor}' is registered as "
                    f"{report.trust_level} yet scores {risk:.2f}, worse than the best "
                    f"less-trusted contributor at {best_lower:.2f}. A vetted supply-chain "
                    f"source behaving worse than an unvetted one undermines the "
                    f"procurement assumption and warrants supplier review."),
                evidence=entry | {
                    "trust_levels": {r.contributor: r.trust_level for r in reports},
                    "risk_scores": {r.contributor: round(float(r.risk_score), 4)
                                    for r in reports},
                },
                module="meeting_room",
                detector="trust_inversion",
                contributor=report.contributor,
                prefix="MEET",
            )
            findings.append(finding.to_dict())

        return {"inversions": inversions, "findings": findings, "limitations": limitations}

    # ------------------------------------------------------------------
    # model disagreement
    # ------------------------------------------------------------------
    @staticmethod
    def _build_probes(num_probes: int, shape: tuple,
                      reference_images: Optional[Sequence[str | Path]] = None
                      ) -> tuple:
        """Build a deterministic probe batch for model comparison.

        Uniform noise is a poor probe set: it lies far off the data manifold,
        where models saturate to a single class and disagree on nothing. That
        yields a 0% mismatch rate for genuinely different models — a false
        negative. Real images are used when available; otherwise structured
        synthetic probes (oriented gratings, blobs and gradients) are generated,
        which excite the convolutional filters the way natural imagery does.

        Args:
            num_probes: Number of probes to build.
            shape: Model input shape ``(C, H, W)``.
            reference_images: Optional real image paths, strongly preferred.

        Returns:
            Tuple of ``(probes, source, notes)``.
        """
        channels, height, width = shape
        notes: List[str] = []

        if reference_images:
            try:
                from src.utils.image_utils import load_image

                collected = []
                for path in list(reference_images)[:num_probes]:
                    # load_image yields HWC uint8; models expect CHW float in 0..1.
                    raw = load_image(path, size=(width, height),
                                     grayscale=(channels == 1))
                    if raw is None:
                        continue
                    array = np.asarray(raw, dtype=np.float32) / 255.0
                    if array.ndim == 2:
                        array = array[None, :, :]
                    else:
                        array = np.transpose(array, (2, 0, 1))
                    if array.shape[0] != channels:
                        array = (np.repeat(array[:1], channels, axis=0)
                                 if array.shape[0] == 1 else array[:channels])
                    collected.append(array)
                if len(collected) >= 2:
                    return np.stack(collected), "reference_images", notes
                notes.append("Too few reference images loaded; fell back to "
                             "structured synthetic probes.")
            except Exception as exc:
                notes.append(f"Reference image probes unavailable ({exc}); fell back "
                             f"to structured synthetic probes.")

        rng = np.random.default_rng(1337)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        yy /= max(height - 1, 1)
        xx /= max(width - 1, 1)

        probes = np.zeros((num_probes, channels, height, width), dtype=np.float32)
        for i in range(num_probes):
            angle = float(rng.uniform(0.0, np.pi))
            freq = float(rng.uniform(2.0, 8.0))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            grating = 0.5 + 0.35 * np.sin(
                2.0 * np.pi * freq * (xx * np.cos(angle) + yy * np.sin(angle)) + phase)
            cy, cx = float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8))
            blob = 0.45 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2)
                                   / (2.0 * float(rng.uniform(0.02, 0.10)))))
            base = np.clip(grating + blob, 0.0, 1.0)
            for c in range(channels):
                tint = 1.0 + 0.10 * (c - (channels - 1) / 2.0)
                probes[i, c] = np.clip(base * tint
                                       + rng.normal(0.0, 0.02, size=base.shape), 0.0, 1.0)

        notes.append("Structured synthetic probes (oriented gratings plus blobs) were "
                     "used: uniform noise is off-manifold and makes distinct models "
                     "collapse to the same output, masking real disagreement.")
        return probes, "synthetic_structured", notes

    def _model_disagreement(self, models: List[str | Path],
                            num_probes: int,
                            reference_images: Optional[Sequence[str | Path]] = None
                            ) -> Dict[str, Any]:
        """Compare candidate models on identical deterministic probes.

        Two models that should be functionally equivalent but disagree on a
        fixed probe set differ internally; at least one has been modified. The
        probes are generated from a fixed seed so the comparison is reproducible
        and needs no external data.

        Args:
            models: Model paths to compare.
            num_probes: Number of probe images to synthesise.

        Returns:
            Analysis dictionary including any findings raised.
        """
        findings: List[Dict[str, Any]] = []
        limitations: List[str] = []
        pairs: List[Dict[str, Any]] = []

        if len(models) < 2:
            limitations.append(
                f"Model disagreement needs at least 2 models; {len(models)} supplied.")
            return {"models": [str(m) for m in models], "pairs": [],
                    "findings": [], "limitations": limitations}

        from src.loaders.model_loader import load_model

        loaded: List[Dict[str, Any]] = []
        for source in models:
            handle = load_model(source)
            if not handle.can_predict:
                limitations.append(
                    f"Model '{source}' could not be queried (access_level="
                    f"{handle.access_level}); excluded from disagreement analysis.")
                continue
            loaded.append({"path": str(source), "model": handle})

        if len(loaded) < 2:
            limitations.append(
                "Fewer than 2 queryable models; disagreement was not assessed.")
            return {"models": [str(m) for m in models], "pairs": [],
                    "findings": [], "limitations": limitations}

        shape = next((entry["model"].input_shape for entry in loaded
                      if entry["model"].input_shape), (3, 32, 32))
        probes, probe_source, probe_notes = self._build_probes(
            int(num_probes), tuple(shape), reference_images)
        limitations.extend(probe_notes)

        predictions: Dict[str, np.ndarray] = {}
        for entry in loaded:
            output = entry["model"].predict(probes)
            if output is None:
                limitations.append(
                    f"Model '{entry['path']}' returned no output on the probe set.")
                continue
            array = np.asarray(output)
            predictions[entry["path"]] = (array.argmax(axis=1) if array.ndim > 1
                                          else array.ravel())

        names = sorted(predictions)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                left, right = names[i], names[j]
                a, b = predictions[left], predictions[right]
                size = min(len(a), len(b))
                if size == 0:
                    continue
                mismatch = float(np.mean(a[:size] != b[:size]))
                entry = {
                    "model_a": left, "model_b": right,
                    "probes": size,
                    "mismatch_rate": round(mismatch, 4),
                    "threshold": self.disagreement_threshold,
                    "disagree": mismatch > self.disagreement_threshold,
                }
                pairs.append(entry)
                if not entry["disagree"]:
                    continue

                confidence = min(0.90, 0.50 + mismatch)
                finding = Finding.create(
                    attack_class=AttackClass.MODEL_DISAGREEMENT.value,
                    affected_asset=f"{Path(left).name} vs {Path(right).name}",
                    asset_type="model",
                    confidence=confidence,
                    reason=(
                        f"Models '{Path(left).name}' and '{Path(right).name}' disagree on "
                        f"{mismatch:.1%} of {size} identical deterministic probes, above the "
                        f"{self.disagreement_threshold:.0%} threshold. Models that should be "
                        f"functionally equivalent differ internally; at least one has been "
                        f"altered relative to the other."),
                    evidence=entry,
                    module="meeting_room",
                    detector="model_disagreement",
                    prefix="MEET",
                )
                findings.append(finding.to_dict())

        probe_description = {
            "reference_images": "real reference images",
            "synthetic_structured": "structured synthetic probes",
        }.get(probe_source, probe_source)
        limitations.append(
            f"Model disagreement was measured on {probe_description}: it proves two models "
            f"differ, but cannot attribute which one is authentic without a trusted "
            f"reference.")

        return {"models": [str(m) for m in models], "queried": names, "pairs": pairs,
                "probe_source": probe_source, "num_probes": int(len(probes)),
                "findings": findings, "limitations": limitations}

    # ------------------------------------------------------------------
    # verdict
    # ------------------------------------------------------------------
    @staticmethod
    def _verdict(reports: List[AgentReport],
                 findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Produce the office's consolidated verdict and recommendations.

        Args:
            reports: Usable agent reports.
            findings: Findings raised during the meeting.

        Returns:
            Dictionary with ``verdict``, ``narrative``, ``recommendations`` and
            ``worst_contributor``.
        """
        recommendations: List[str] = []
        if not reports:
            return {"verdict": "NOT_ASSESSED",
                    "narrative": "No contributor was successfully scanned; "
                                 "no integrity statement can be made.",
                    "recommendations": ["Re-run the office scan against a readable dataset."],
                    "worst_contributor": None}

        risks = [float(r.risk_score) for r in reports]
        worst = max(reports, key=lambda r: float(r.risk_score))
        peak = max(risks)
        compromised = [r for r in reports if str(r.verdict).upper() == "COMPROMISED"]
        critical = sum(1 for f in findings
                       if str(f.get("severity", "")).upper() == "CRITICAL")

        if peak >= 0.75 or compromised:
            verdict = "COMPROMISED"
        elif peak >= 0.45 or findings:
            verdict = "CAUTION"
        else:
            verdict = "CLEAN"

        narrative = (
            f"{len(reports)} contributor(s) were scanned independently. "
            f"Peak contributor risk was {peak:.2f} ('{worst.contributor}', "
            f"{worst.trust_level}). The meeting raised {len(findings)} "
            f"cross-contributor finding(s)"
            + (f", {critical} of them CRITICAL" if critical else "")
            + f". Office verdict: {verdict}.")

        if compromised:
            recommendations.append(
                "Quarantine data from: "
                + ", ".join(sorted(r.contributor for r in compromised))
                + " until the flagged samples are reviewed by an analyst.")
        if verdict != "CLEAN":
            recommendations.append(
                f"Prioritise manual review of '{worst.contributor}' — highest risk at {peak:.2f}.")
        if any(f.get("detector") == "trust_inversion" for f in findings):
            recommendations.append(
                "Re-open supplier vetting: a vetted contributor is behaving worse than an "
                "unvetted peer.")
        if any(f.get("detector") == "cross_contributor_class_targeting" for f in findings):
            recommendations.append(
                "Treat as a coordinated campaign: multiple contributors target the same class. "
                "Escalate to the security cell rather than handling per-contributor.")
        if any(f.get("detector") == "model_disagreement" for f in findings):
            recommendations.append(
                "Re-establish a trusted model baseline: candidate models are not equivalent.")
        if not recommendations:
            recommendations.append(
                "No cross-contributor anomaly found. Continue routine monitoring; this "
                "statement covers only the checks listed in the coverage section.")

        return {"verdict": verdict, "narrative": narrative,
                "recommendations": recommendations,
                "worst_contributor": worst.contributor}


def hold_meeting(office_result: Dict[str, Any],
                 models: Optional[Sequence[str | Path]] = None,
                 num_probes: int = 100,
                 reference_images: Optional[Sequence[str | Path]] = None
                 ) -> Dict[str, Any]:
    """Convenience wrapper running a full cross-contributor meeting.

    Args:
        office_result: Result from :meth:`src.agents.manager.OfficeManager.run`.
        models: Optional model paths to cross-check.
        num_probes: Number of canonical probes for model comparison.

    Returns:
        Meeting result dictionary as described in :meth:`MeetingRoom.convene`.
    """
    return MeetingRoom().convene(office_result, models=models, num_probes=num_probes,
                                 reference_images=reference_images)


__all__ = ["MeetingRoom", "hold_meeting"]
