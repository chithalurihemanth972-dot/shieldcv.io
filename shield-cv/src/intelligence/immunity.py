"""
ADVERSARIAL IMMUNITY SCORE (AIS) — a single 0-100 posture figure.

The AIS answers one question for a commander: *how much can this pipeline be
trusted right now?* It is a weighted aggregate over four pillars — data
integrity, model integrity, provenance integrity and drift stability.

Two design rules make the number honest rather than merely reassuring:

**1. Unassessed is not the same as clean.** A pillar that was never evaluated
contributes a neutral, explicitly-flagged value and caps the overall confidence.
A pipeline nobody checked must never score 100.

**2. Severity dominates volume.** One CRITICAL finding damages the score far
more than twenty LOW ones. A scoring function that averages severity lets an
attacker hide a backdoor behind a crowd of trivial issues.

The score is always returned with its component breakdown and the specific
reasons for every deduction, so it can be argued with rather than simply
believed.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.config import get_config
from src.utils.helpers import clamp
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

# Severity damage weights. Deliberately super-linear: the gap between CRITICAL
# and HIGH is larger than the gap between MEDIUM and LOW, because a single
# confirmed backdoor is categorically worse than many cosmetic anomalies.
SEVERITY_DAMAGE: Dict[str, float] = {
    "CRITICAL": 1.00,
    "HIGH": 0.55,
    "MEDIUM": 0.22,
    "LOW": 0.07,
    "INFO": 0.02,
}

PILLARS = ("data_integrity", "model_integrity", "provenance", "drift_stability")

PILLAR_LABELS = {
    "data_integrity": "Training data integrity",
    "model_integrity": "Model integrity",
    "provenance": "Inference provenance",
    "drift_stability": "Operational drift stability",
}

PILLAR_MODULES = {
    "data_integrity": ("data_scanner", "trigger_detector"),
    "model_integrity": ("model_auditor",),
    "provenance": ("crypto_chain",),
    "drift_stability": ("drift_detector",),
}

BAND_DESCRIPTIONS = {
    "HARDENED": "Suitable for operational reliance, subject to routine monitoring.",
    "ADEQUATE": "Usable with named mitigations and an increased monitoring cadence.",
    "MARGINAL": "Not suitable for reliance on its own; remediate before operational use.",
    "COMPROMISED": "Do not rely on this pipeline. Quarantine and re-accredit.",
}


class ImmunityScorer:
    """Computes the Adversarial Immunity Score from module results.

    Attributes:
        cfg: Effective configuration.
        weights: Per-pillar weights, normalised to sum to 1.
        bands: Score cut-offs for the posture bands.
    """

    MODULE_NAME = "immunity"

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialise the scorer and read its weights and bands.

        Args:
            config: Optional configuration override.
        """
        self.cfg = config or get_config()
        section = self.cfg.section("immunity")
        raw_weights = section.get("weights", {}) or {}
        weights = {name: float(raw_weights.get(name, 0.25)) for name in PILLARS}
        total = sum(weights.values()) or 1.0
        # Normalise so a mis-specified config cannot silently change the scale.
        self.weights = {name: value / total for name, value in weights.items()}
        bands = section.get("bands", {}) or {}
        self.bands = {
            "hardened": float(bands.get("hardened", 85)),
            "adequate": float(bands.get("adequate", 70)),
            "marginal": float(bands.get("marginal", 50)),
        }

    # ------------------------------------------------------------------
    def score(self, module_results: Dict[str, Dict[str, Any]],
              threat_story: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compute the Adversarial Immunity Score.

        Args:
            module_results: Mapping of module name to result dictionary.
            threat_story: Optional threat-story result. A confirmed campaign
                applies an additional penalty, because correlated compromise is
                worse than the sum of its parts.

        Returns:
            Dictionary with ``score``, ``band``, ``components``, ``deductions``,
            ``confidence``, ``assessment`` and ``recommendations``.
        """
        started = time.time()
        result: Dict[str, Any] = {
            "module": self.MODULE_NAME, "score": 0.0, "band": "UNKNOWN",
            "components": {}, "deductions": [], "confidence": 0.0,
            "assessment": "", "recommendations": [], "limitations": [],
            "duration_seconds": 0.0,
        }

        try:
            components: Dict[str, Dict[str, Any]] = {}
            assessed_weight = 0.0

            for pillar in PILLARS:
                component = self._score_pillar(pillar, module_results)
                components[pillar] = component
                if component["assessed"]:
                    assessed_weight += self.weights[pillar]

            # Weighted aggregate. Unassessed pillars contribute their neutral
            # value so the score is defined, but they are excluded from the
            # confidence and called out explicitly.
            total = 0.0
            for pillar, component in components.items():
                total += self.weights[pillar] * float(component["score"])

            campaign_penalty, campaign_note = self._campaign_penalty(threat_story)
            raw_score = clamp(total - campaign_penalty, 0.0, 100.0)

            deductions: List[Dict[str, Any]] = []
            for pillar, component in components.items():
                for reason in component.get("reasons", []):
                    deductions.append({
                        "pillar": pillar,
                        "pillar_label": PILLAR_LABELS[pillar],
                        "weight": round(self.weights[pillar], 3),
                        **reason,
                    })
            if campaign_penalty > 0:
                deductions.append({
                    "pillar": "cross_module",
                    "pillar_label": "Cross-module correlation",
                    "points": round(campaign_penalty, 2),
                    "reason": campaign_note,
                })

            deductions.sort(key=lambda d: d.get("points", 0.0), reverse=True)

            unassessed = [PILLAR_LABELS[p] for p in PILLARS
                          if not components[p]["assessed"]]
            confidence = round(clamp(assessed_weight), 3)
            if unassessed:
                result["limitations"].append(
                    f"{len(unassessed)} of {len(PILLARS)} pillars were not assessed "
                    f"({', '.join(unassessed)}). Their neutral contribution means this "
                    "score describes only what was actually examined — it is not a "
                    "statement that the unexamined pillars are sound.")

            band = self._band(raw_score, confidence, bool(unassessed))
            result.update({
                "score": round(raw_score, 1),
                "band": band,
                "band_description": BAND_DESCRIPTIONS.get(band, ""),
                "components": components,
                "deductions": deductions[:20],
                "confidence": confidence,
                "weights": {k: round(v, 3) for k, v in self.weights.items()},
                "pillars_assessed": len(PILLARS) - len(unassessed),
                "pillars_total": len(PILLARS),
                "assessment": self._assessment(raw_score, band, components,
                                               unassessed, confidence),
                "recommendations": self._recommendations(components, band, unassessed),
                "duration_seconds": round(time.time() - started, 2),
            })

            LOGGER.info("Adversarial Immunity Score: %.1f/100 (%s, confidence %.2f)",
                        result["score"], band, confidence)
            return result
        except Exception as exc:
            LOGGER.error("Immunity scoring failed: %s", exc, exc_info=True)
            result["limitations"].append(f"Scoring aborted: {exc}")
            result["assessment"] = f"Immunity score could not be computed: {exc}"
            result["duration_seconds"] = round(time.time() - started, 2)
            return result

    # ------------------------------------------------------------------
    def _score_pillar(self, pillar: str,
                      module_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Score one pillar out of 100.

        Args:
            pillar: Pillar key.
            module_results: All module results.

        Returns:
            Component dictionary with ``score``, ``assessed`` and ``reasons``.
        """
        component: Dict[str, Any] = {
            "label": PILLAR_LABELS[pillar], "score": 70.0, "assessed": False,
            "findings": 0, "reasons": [],
            "note": "NOT ASSESSED — neutral value applied, not a clean result.",
        }
        try:
            payloads = [module_results[m] for m in PILLAR_MODULES[pillar]
                        if isinstance(module_results.get(m), dict)]
            if not payloads:
                return component

            findings: List[Dict[str, Any]] = []
            for payload in payloads:
                findings.extend(payload.get("findings", []) or [])

            component["assessed"] = True
            component["findings"] = len(findings)
            component["note"] = ""

            if not findings:
                component["score"] = 100.0
                component["reasons"] = []
                return component

            # Damage model: the worst finding sets the floor, and additional
            # findings add diminishing further damage. Using a sum alone would
            # let volume swamp severity; using the max alone would ignore
            # breadth entirely.
            severities = Counter(str(f.get("severity", "INFO")).upper() for f in findings)
            damages = sorted(
                (SEVERITY_DAMAGE.get(str(f.get("severity", "INFO")).upper(), 0.05)
                 * float(f.get("confidence", 0.5)) for f in findings),
                reverse=True)

            # Saturating damage curve. Two failure modes must be avoided:
            #   * a linear sum drives any large dataset to 0 and destroys all
            #     discrimination between "flawed" and "catastrophic";
            #   * using the worst finding alone ignores breadth entirely.
            # The worst finding therefore sets a floor of damage, and breadth
            # adds a saturating contribution that approaches — but never
            # reaches — total loss. Scores stay comparable across corpora of
            # very different sizes.
            worst = damages[0]
            tail = sum(value / (index + 2.0) for index, value in enumerate(damages[1:]))
            breadth = 1.0 - float(np.exp(-tail / 6.0))
            damage = clamp(0.55 * worst + 0.45 * breadth, 0.0, 0.98)
            component["score"] = round(100.0 * (1.0 - damage), 1)
            component["damage_worst"] = round(float(worst), 4)
            component["damage_breadth"] = round(float(breadth), 4)

            reasons: List[Dict[str, Any]] = []
            for severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
                count = severities.get(severity, 0)
                if not count:
                    continue
                reasons.append({
                    "severity": severity,
                    "count": count,
                    "points": round(100.0 * SEVERITY_DAMAGE.get(severity, 0.05)
                                    * self.weights[pillar], 2),
                    "reason": (f"{count} {severity} finding(s) in "
                               f"{PILLAR_LABELS[pillar].lower()}"),
                })
            component["reasons"] = reasons
            component["severity_counts"] = dict(severities)
            return component
        except Exception as exc:
            LOGGER.error("_score_pillar(%s) failed: %s", pillar, exc)
            component["note"] = f"Scoring error: {exc}"
            return component

    def _campaign_penalty(self, threat_story: Optional[Dict[str, Any]]) -> tuple:
        """Compute the additional penalty for a correlated campaign.

        Args:
            threat_story: Threat-story result, if available.

        Returns:
            Tuple of ``(points, explanation)``.
        """
        try:
            if not isinstance(threat_story, dict):
                return 0.0, ""
            assessment = threat_story.get("campaign_assessment", {}) or {}
            if not assessment.get("campaign_detected"):
                return 0.0, ""
            confidence = float(assessment.get("confidence", 0.0))
            patterns = assessment.get("patterns", [])
            # Correlated compromise is worse than independent faults: it implies
            # an adversary with reach, so remediating one stage is insufficient.
            points = clamp(8.0 * confidence + 2.0 * len(patterns), 0.0, 25.0)
            return points, (
                f"Cross-module correlation identified a coordinated campaign "
                f"({', '.join(patterns)}) at {confidence:.0%} confidence. Correlated "
                "compromise implies adversary reach across pipeline stages, so the "
                "posture is worse than the individual pillar scores suggest.")
        except Exception as exc:
            LOGGER.error("_campaign_penalty failed: %s", exc)
            return 0.0, ""

    def _band(self, score: float, confidence: float, has_gaps: bool) -> str:
        """Map a score to a posture band.

        Args:
            score: The computed score.
            confidence: Fraction of total weight actually assessed.
            has_gaps: Whether any pillar went unassessed.

        Returns:
            Band name.
        """
        try:
            if score >= self.bands["hardened"]:
                band = "HARDENED"
            elif score >= self.bands["adequate"]:
                band = "ADEQUATE"
            elif score >= self.bands["marginal"]:
                band = "MARGINAL"
            else:
                band = "COMPROMISED"

            # A high score earned from partial coverage cannot claim the top
            # band. Certifying a pipeline as HARDENED on the strength of checks
            # that were never run is exactly the failure mode this framework
            # exists to prevent.
            if band == "HARDENED" and (has_gaps or confidence < 0.85):
                return "ADEQUATE"
            return band
        except Exception:
            return "UNKNOWN"

    def _assessment(self, score: float, band: str, components: Dict[str, Dict[str, Any]],
                    unassessed: Sequence[str], confidence: float) -> str:
        """Compose the analyst-facing summary of the score.

        Args:
            score: Final score.
            band: Posture band.
            components: Per-pillar components.
            unassessed: Labels of unassessed pillars.
            confidence: Assessment coverage.

        Returns:
            Summary paragraph.
        """
        try:
            assessed = {k: v for k, v in components.items() if v["assessed"]}
            parts = [f"Adversarial Immunity Score {score:.1f}/100 ({band})."]

            if assessed:
                ordered = sorted(assessed.items(), key=lambda kv: kv[1]["score"])
                weakest, weakest_data = ordered[0]
                parts.append(
                    f"Weakest pillar: {PILLAR_LABELS[weakest]} at "
                    f"{weakest_data['score']:.0f}/100 with "
                    f"{weakest_data['findings']} finding(s).")
                if len(ordered) > 1:
                    strongest, strongest_data = ordered[-1]
                    parts.append(f"Strongest: {PILLAR_LABELS[strongest]} at "
                                 f"{strongest_data['score']:.0f}/100.")

            if unassessed:
                parts.append(
                    f"Coverage is incomplete — {', '.join(unassessed)} "
                    f"{'was' if len(unassessed) == 1 else 'were'} not assessed, so "
                    f"{confidence:.0%} of the scoring weight rests on actual evidence. "
                    "The remainder is a neutral placeholder and must not be read as a "
                    "clean result.")
            else:
                parts.append("All four pillars were assessed.")

            parts.append(BAND_DESCRIPTIONS.get(band, ""))
            return " ".join(p for p in parts if p)
        except Exception as exc:
            LOGGER.error("_assessment failed: %s", exc)
            return f"Score {score:.1f}/100 ({band})."

    def _recommendations(self, components: Dict[str, Dict[str, Any]],
                         band: str, unassessed: Sequence[str]) -> List[str]:
        """Produce prioritised recommendations.

        Args:
            components: Per-pillar components.
            band: Posture band.
            unassessed: Labels of unassessed pillars.

        Returns:
            Ordered list of recommendations.
        """
        recommendations: List[str] = []
        try:
            if band == "COMPROMISED":
                recommendations.append(
                    "Suspend operational reliance on this pipeline pending re-accreditation.")

            ordered = sorted((kv for kv in components.items() if kv[1]["assessed"]),
                             key=lambda kv: kv[1]["score"])
            for pillar, data in ordered:
                if data["score"] >= 90:
                    continue
                counts = data.get("severity_counts", {})
                critical = counts.get("CRITICAL", 0)
                high = counts.get("HIGH", 0)
                if critical:
                    recommendations.append(
                        f"{PILLAR_LABELS[pillar]}: quarantine and manually adjudicate the "
                        f"{critical} CRITICAL finding(s) before any further use.")
                elif high:
                    recommendations.append(
                        f"{PILLAR_LABELS[pillar]}: review the {high} HIGH finding(s); "
                        "each carries specific evidence and a recommended disposition.")
                else:
                    recommendations.append(
                        f"{PILLAR_LABELS[pillar]}: scored {data['score']:.0f}/100 — "
                        "schedule routine remediation of the reported findings.")

            for label in unassessed:
                recommendations.append(
                    f"Run the {label.lower()} assessment: this pillar is currently "
                    "unevaluated and its score is a placeholder, not a clean result.")

            return recommendations[:10]
        except Exception as exc:
            LOGGER.error("_recommendations failed: %s", exc)
            return recommendations


def compute_immunity_score(module_results: Dict[str, Dict[str, Any]],
                           threat_story: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience wrapper computing the Adversarial Immunity Score.

    Args:
        module_results: Mapping of module name to result dictionary.
        threat_story: Optional threat-story result.

    Returns:
        Immunity score dictionary.
    """
    try:
        return ImmunityScorer().score(module_results, threat_story=threat_story)
    except Exception as exc:
        LOGGER.error("compute_immunity_score failed: %s", exc)
        return {"module": "immunity", "score": 0.0, "band": "UNKNOWN",
                "limitations": [str(exc)]}


__all__ = ["ImmunityScorer", "compute_immunity_score", "PILLARS", "SEVERITY_DAMAGE"]
