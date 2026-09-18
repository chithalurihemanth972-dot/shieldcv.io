"""
THREAT STORY ENGINE — cross-module correlation and narrative construction.

Individual findings are evidence; a threat story is an *argument*. This engine
reads the outputs of all four scanners together and looks for structure that no
single module can see:

* **Kill-chain correlation** — a poisoned dataset, a backdoored model and a
  tampered inference record that all reference the same target class describe
  one campaign, not three coincidences.
* **Coordinated attack** — several contributors exhibiting the same attack class
  with the same parameters implies collusion or a shared upstream compromise.
* **Cover-up** — provenance tampering or drift that appears where other findings
  cluster suggests an attempt to erase evidence.
* **Targeting** — findings concentrating on one class or one contributor.

A deliberate design rule: this engine NEVER invents severity. A correlation can
raise the *priority* of reviewing existing findings and can emit its own
correlation finding, but it cannot upgrade an underlying finding's confidence.
Correlation is suggestive, and a narrative that inflates its own evidence is
worse than no narrative at all.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional


from src.config import get_config
from src.reporting.schema import (
    AttackClass,
    Finding,
    summarize_findings,
)
from src.utils.helpers import clamp, noisy_or
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

# Attack classes grouped by the pipeline stage they attack. Used to decide
# whether a set of findings spans a kill chain or sits in one stage.
STAGE_OF_ATTACK: Dict[str, str] = {
    AttackClass.TRIGGER_INJECTION.value: "DATA",
    AttackClass.LABEL_FLIPPING.value: "DATA",
    AttackClass.NEAR_DUPLICATE_FLOODING.value: "DATA",
    AttackClass.OUT_OF_DISTRIBUTION.value: "DATA",
    AttackClass.SYSTEMATIC_MISLABELING.value: "DATA",
    AttackClass.MODEL_BACKDOOR.value: "MODEL",
    AttackClass.MODEL_SUBSTITUTION.value: "MODEL",
    AttackClass.WEIGHT_ANOMALY.value: "MODEL",
    AttackClass.ACTIVATION_ANOMALY.value: "MODEL",
    AttackClass.BEHAVIORAL_DEVIATION.value: "MODEL",
    AttackClass.INFERENCE_TAMPER.value: "INFERENCE",
    AttackClass.INFERENCE_REPLAY.value: "INFERENCE",
    AttackClass.CHAIN_BREAK.value: "INFERENCE",
    AttackClass.SIGNATURE_INVALID.value: "INFERENCE",
    AttackClass.DISTRIBUTION_SHIFT.value: "OPERATIONS",
}

STAGE_ORDER = ("DATA", "MODEL", "INFERENCE", "OPERATIONS")


class ThreatStoryEngine:
    """Correlates findings across modules into coherent threat narratives.

    Attributes:
        cfg: Effective configuration.
        stories: Narratives produced by the most recent correlation.
        findings: Correlation findings emitted.
    """

    MODULE_NAME = "threat_story"

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialise the engine.

        Args:
            config: Optional configuration override.
        """
        self.cfg = config or get_config()
        # Materiality floor. Correlating low-confidence findings manufactures
        # narratives out of noise: a benign corpus with incidental near
        # duplicates at ~0.4 confidence would otherwise produce a
        # "coordinated attack across contributors" story. Only findings the
        # detectors are reasonably sure about are eligible to become evidence
        # in a narrative. The weak findings are still reported by their own
        # modules — they are simply not promoted into a campaign claim.
        self.min_correlation_confidence = float(
            self.cfg.get("intelligence.min_correlation_confidence", 0.50))
        self.stories: List[Dict[str, Any]] = []
        self.findings: List[Finding] = []

    # ------------------------------------------------------------------
    def correlate(self, module_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Correlate the results of several modules into threat stories.

        Args:
            module_results: Mapping of module name to that module's result
                dictionary, e.g. ``{"data_scanner": {...}, "model_auditor": {...}}``.
                Missing modules are handled: their absence becomes a stated
                coverage gap rather than an assumption of safety.

        Returns:
            Dictionary with ``stories``, ``findings``, ``summary``,
            ``campaign_assessment``, ``coverage`` and ``limitations``.
        """
        started = time.time()
        self.stories = []
        self.findings = []
        limitations: List[str] = []

        result: Dict[str, Any] = {
            "module": self.MODULE_NAME, "stories": [], "findings": [],
            "summary": {}, "campaign_assessment": {}, "coverage": {},
            "limitations": [], "duration_seconds": 0.0,
        }

        try:
            all_findings = self._gather(module_results)
            findings = [f for f in all_findings
                        if float(f.get("confidence", 0.0)) >= self.min_correlation_confidence]
            result["coverage"] = self._coverage(module_results, limitations)
            result["total_findings_considered"] = len(findings)
            result["findings_below_materiality"] = len(all_findings) - len(findings)
            result["min_correlation_confidence"] = self.min_correlation_confidence
            if result["findings_below_materiality"]:
                limitations.append(
                    f"{result['findings_below_materiality']} finding(s) below the "
                    f"{self.min_correlation_confidence:.2f} materiality floor were excluded "
                    "from correlation. They remain in their originating module's report; "
                    "they are simply too weak to support a campaign-level claim.")

            if not findings:
                result["campaign_assessment"] = {
                    "campaign_detected": False, "confidence": 0.0,
                    "assessment": ("No findings of sufficient confidence were raised by any "
                                   "module that ran. "
                                   "Note that this is not proof of integrity — see the "
                                   "coverage statement for what was actually assessed."),
                }
                result["limitations"] = limitations
                result["summary"] = summarize_findings([])
                result["duration_seconds"] = round(time.time() - started, 2)
                return result

            # Each correlator is independent and additive.
            self.stories.extend(self._correlate_kill_chain(findings))
            self.stories.extend(self._correlate_target_class(findings))
            self.stories.extend(self._correlate_contributors(findings))
            self.stories.extend(self._correlate_cover_up(findings))

            self.stories.sort(key=lambda s: s.get("confidence", 0.0), reverse=True)
            for story in self.stories:
                self._emit_finding(story)

            result["stories"] = self.stories
            result["findings"] = [f.to_dict() for f in self.findings]
            result["summary"] = summarize_findings(result["findings"])
            result["campaign_assessment"] = self._campaign_assessment(findings, self.stories)
            result["limitations"] = limitations
            result["duration_seconds"] = round(time.time() - started, 2)

            LOGGER.info("Threat correlation complete: %d story(ies) from %d finding(s)",
                        len(self.stories), len(findings))
            return result
        except Exception as exc:
            LOGGER.error("Threat correlation failed: %s", exc, exc_info=True)
            limitations.append(f"Correlation aborted: {exc}")
            result["limitations"] = limitations
            result["summary"] = summarize_findings([])
            result["duration_seconds"] = round(time.time() - started, 2)
            return result

    # ------------------------------------------------------------------
    def _gather(self, module_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten findings from every module, tagging each with its stage.

        Args:
            module_results: Per-module results.

        Returns:
            List of finding dictionaries with ``_module`` and ``_stage`` added.
        """
        gathered: List[Dict[str, Any]] = []
        try:
            for module_name, payload in (module_results or {}).items():
                if not isinstance(payload, dict):
                    continue
                for finding in payload.get("findings", []) or []:
                    if not isinstance(finding, dict):
                        continue
                    enriched = dict(finding)
                    enriched["_module"] = finding.get("module") or module_name
                    enriched["_stage"] = STAGE_OF_ATTACK.get(
                        finding.get("attack_class", ""), "UNKNOWN")
                    gathered.append(enriched)
            return gathered
        except Exception as exc:
            LOGGER.error("_gather failed: %s", exc)
            return gathered

    def _coverage(self, module_results: Dict[str, Dict[str, Any]],
                  limitations: List[str]) -> Dict[str, Any]:
        """Record which modules contributed and which did not.

        Args:
            module_results: Per-module results.
            limitations: Mutable list receiving coverage gap statements.

        Returns:
            Coverage dictionary.
        """
        expected = {
            "data_scanner": "training data integrity",
            "model_auditor": "model integrity",
            "crypto_chain": "inference provenance",
            "drift_detector": "operational distribution shift",
        }
        coverage: Dict[str, Any] = {"assessed": [], "not_assessed": []}
        try:
            for module, description in expected.items():
                payload = (module_results or {}).get(module)
                if isinstance(payload, dict) and payload:
                    # `.get(key, [])` returns None when the key is present but
                    # null, so the default never applies. Module results arrive
                    # from callers outside this package, so coerce explicitly
                    # rather than trusting the shape.
                    findings = payload.get("findings") or []
                    coverage["assessed"].append({"module": module, "covers": description,
                                                 "findings": len(findings)})
                else:
                    coverage["not_assessed"].append({"module": module, "covers": description})
                    limitations.append(
                        f"{module} did not run: {description} was NOT assessed. Correlations "
                        "involving this stage of the pipeline cannot be evaluated.")
            coverage["stages_covered"] = len(coverage["assessed"])
            coverage["stages_total"] = len(expected)
            return coverage
        except Exception as exc:
            LOGGER.error("_coverage failed: %s", exc)
            return coverage

    # ------------------------------------------------------------------
    def _correlate_kill_chain(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect an attack spanning multiple pipeline stages.

        Args:
            findings: All findings.

        Returns:
            Zero or one kill-chain story.
        """
        try:
            by_stage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for finding in findings:
                if finding["_stage"] not in STAGE_ORDER:
                    continue
                # A benign explanation disqualifies a finding as kill-chain
                # evidence. Natural operational drift (a change of season, a
                # new sensor) is a real finding worth acting on, but it is NOT
                # evidence of an adversary, and letting it count as a
                # compromised "stage" turns bad weather into a cyber incident.
                if self._has_benign_explanation(finding):
                    continue
                by_stage[finding["_stage"]].append(finding)

            stages_hit = [s for s in STAGE_ORDER if by_stage.get(s)]
            if len(stages_hit) < 2:
                return []

            # Require at least one corroborated or high-confidence finding in
            # each stage. Two stages of weak, single-method signals is not a
            # campaign; it is two detectors having a bad day.
            if not all(any(self._is_corroborated(f) for f in by_stage[stage])
                       for stage in stages_hit):
                return []

            # Confidence comes from the strongest finding in each stage, fused
            # with a noisy-OR. Breadth across stages is what makes a kill chain
            # compelling, so it is reflected in the fusion rather than bolted on.
            per_stage_best = [max(float(f.get("confidence", 0.0)) for f in by_stage[s])
                              for s in stages_hit]
            # Correlation is inferential, so it is capped below certainty no
            # matter how many stages agree. A narrative must never present
            # itself as stronger evidence than the measurements underneath it.
            confidence = min(0.95, clamp(
                noisy_or(per_stage_best) * (0.70 + 0.07 * len(stages_hit))))

            narrative_parts: List[str] = []
            for stage in stages_hit:
                items = by_stage[stage]
                classes = Counter(f["attack_class"] for f in items)
                described = ", ".join(f"{count}x {name}" for name, count in classes.most_common(3))
                narrative_parts.append(f"{stage} stage — {described}")

            evidence_ids = [f.get("finding_id", "") for f in findings
                            if f["_stage"] in stages_hit][:40]

            return [{
                "story_id": "STORY-KILLCHAIN",
                "title": f"Multi-stage compromise spanning {len(stages_hit)} pipeline stages",
                "pattern": "KILL_CHAIN",
                "attack_class": AttackClass.COORDINATED_ATTACK.value,
                "confidence": round(confidence, 4),
                "stages": stages_hit,
                "narrative": (
                    "Findings are present at multiple independent stages of the pipeline: "
                    + "; ".join(narrative_parts)
                    + ". An adversary who only had access to training data could not also "
                      "alter model weights or inference records. Findings at "
                    + f"{len(stages_hit)} separate stages therefore indicate either a "
                      "single actor with broad access, or multiple coordinated actors — "
                      "in both cases the appropriate assumption is a deliberate campaign "
                      "rather than independent quality defects."),
                "implication": (
                    "Treat the whole pipeline as compromised until each stage is "
                    "independently re-accredited. Remediating one stage alone will not "
                    "remove the adversary's access."),
                "evidence_finding_ids": evidence_ids,
                "supporting_finding_count": sum(len(by_stage[s]) for s in stages_hit),
            }]
        except Exception as exc:
            LOGGER.error("_correlate_kill_chain failed: %s", exc)
            return []

    def _has_benign_explanation(self, finding: Dict[str, Any]) -> bool:
        """Judge whether a finding was already explained as non-adversarial.

        Args:
            finding: A finding dictionary.

        Returns:
            ``True`` when the producing module attributed the finding to a
            natural cause.
        """
        try:
            evidence = finding.get("evidence", {}) or {}
            if evidence.get("shift_type") == "NATURAL_OPERATIONAL_DRIFT":
                return True
            if evidence.get("natural_signatures") and not evidence.get("suspicious_signals"):
                return True
            return False
        except Exception:
            return False

    def _correlate_target_class(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect findings converging on a single target class.

        Args:
            findings: All findings.

        Returns:
            Stories for each consistently targeted class.
        """
        stories: List[Dict[str, Any]] = []
        try:
            by_class: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
            for finding in findings:
                evidence = finding.get("evidence", {}) or {}
                for key in ("target_class", "spectral_class", "class", "to_class",
                            "new_label", "predicted_class"):
                    if key in evidence and evidence[key] is not None:
                        by_class[str(evidence[key])].append(finding)
                        break

            for class_id, items in by_class.items():
                modules = {f["_module"] for f in items}
                stages = {f["_stage"] for f in items}
                # A single module repeatedly naming a class is just that
                # module's output. Two independent modules agreeing on the same
                # class is a genuine correlation.
                if len(modules) < 2 or len(items) < 3:
                    continue

                confidence = clamp(noisy_or(
                    [float(f.get("confidence", 0.0)) for f in items][:6]) * 0.85)
                classes = Counter(f["attack_class"] for f in items)

                stories.append({
                    "story_id": f"STORY-TARGET-{class_id}",
                    "title": f"Convergent targeting of class {class_id}",
                    "pattern": "TARGETED_CLASS",
                    "attack_class": AttackClass.COORDINATED_ATTACK.value,
                    "confidence": round(confidence, 4),
                    "target_class": class_id,
                    "narrative": (
                        f"{len(items)} findings from {len(modules)} independent modules "
                        f"({', '.join(sorted(modules))}) all reference class {class_id}: "
                        + ", ".join(f"{count}x {name}" for name, count in classes.most_common())
                        + ". Independent detectors converging on the same class is the "
                          "signature of a targeted backdoor: the adversary's objective is to "
                          "control what the system reports for this specific class, and the "
                          "same objective leaves traces in the data, in the weights, or in "
                          "both."),
                    "implication": (
                        f"Any operational decision that depends on class {class_id} should be "
                        "treated as unreliable. Prioritise manual review of all assets "
                        "associated with this class."),
                    "evidence_finding_ids": [f.get("finding_id", "") for f in items][:40],
                    "supporting_finding_count": len(items),
                    "stages": sorted(x for x in stages if x in STAGE_ORDER),
                })
            return stories
        except Exception as exc:
            LOGGER.error("_correlate_target_class failed: %s", exc)
            return stories

    def _correlate_contributors(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect coordinated behaviour across multiple contributors.

        Sharing an attack *class* is far too weak a test — several contributors
        can independently trip the same detector, and on a benign corpus they
        routinely do. Genuine coordination shows up as a shared attack
        *signature*: the same class with the same distinguishing parameters
        (trigger location, target class, flip direction). This correlator
        therefore compares signatures, and additionally requires the underlying
        findings to be corroborated rather than single-method, because a
        narrative built on uncorroborated detections inherits their error rate.

        Args:
            findings: All findings above the materiality floor.

        Returns:
            Zero or one coordinated-contributor story.
        """
        try:
            by_contributor: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for finding in findings:
                contributor = finding.get("contributor")
                if not contributor or contributor == "_unattributed":
                    continue
                # Contributor-level roll-ups are summaries of the very findings
                # being correlated; including them would double count.
                if finding.get("attack_class") == AttackClass.CONTRIBUTOR_RISK.value:
                    continue
                if self._is_corroborated(finding):
                    by_contributor[str(contributor)].append(finding)

            if len(by_contributor) < 2:
                return []

            signatures: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
                lambda: defaultdict(list))
            for name, items in by_contributor.items():
                for finding in items:
                    signatures[self._attack_signature(finding)][name].append(finding)

            best_signature, best_map = "", {}
            for signature, contributor_map in signatures.items():
                if len(contributor_map) >= 2 and len(contributor_map) > len(best_map):
                    best_signature, best_map = signature, contributor_map
            if not best_map:
                return []

            involved = sorted(best_map)
            relevant = [f for items in best_map.values() for f in items]
            confidence = clamp(noisy_or(
                sorted((float(f.get("confidence", 0.0)) for f in relevant),
                       reverse=True)[:5]) * 0.80)

            # Trust context is reporting metadata only — it never alters
            # detector confidence, only how hard the story is escalated.
            trust_context = []
            for name in involved:
                meta = self.cfg.contributor(name)
                trust_context.append(f"{meta.get('display_name', name)} "
                                     f"[{meta.get('trust_level', 'UNVERIFIED')}]")

            return [{
                "story_id": "STORY-COORDINATED",
                "title": f"Identical attack signature across {len(involved)} contributors",
                "pattern": "COORDINATED_CONTRIBUTORS",
                "attack_class": AttackClass.COORDINATED_ATTACK.value,
                "confidence": round(confidence, 4),
                "contributors": involved,
                "shared_signature": best_signature,
                "shared_attack_classes": sorted({f["attack_class"] for f in relevant}),
                "narrative": (
                    f"The attack signature '{best_signature}' appears in corroborated "
                    f"findings from {len(involved)} different contributors "
                    f"({'; '.join(trust_context)}). Matching on the full signature — the "
                    "attack class together with its distinguishing parameters — rather than "
                    "merely on the attack class means these submissions share the same "
                    "specific artefact, not just the same category of defect. Contributors "
                    "are nominally independent sources, so a shared artefact points to a "
                    "common poisoned upstream dataset, shared tooling that introduces it, "
                    "or deliberate collusion."),
                "implication": (
                    "Investigate what these contributors have in common — shared collection "
                    "platform, annotation vendor, or preprocessing tooling — before accepting "
                    "further submissions from any of them."),
                "evidence_finding_ids": [f.get("finding_id", "") for f in relevant][:40],
                "supporting_finding_count": len(relevant),
                "stages": sorted({f["_stage"] for f in relevant} & set(STAGE_ORDER)),
            }]
        except Exception as exc:
            LOGGER.error("_correlate_contributors failed: %s", exc)
            return []

    def _is_corroborated(self, finding: Dict[str, Any]) -> bool:
        """Judge whether a finding rests on more than one independent signal.

        Args:
            finding: A finding dictionary.

        Returns:
            ``True`` when multiple detection methods agreed, or when the
            finding is strong enough to stand alone.
        """
        try:
            evidence = finding.get("evidence", {}) or {}
            agreed = int(evidence.get("detection_methods_agreed", 0) or 0)
            if agreed >= 2:
                return True
            # A very high single-method confidence is still admissible; the bar
            # is deliberately well above the materiality floor.
            return float(finding.get("confidence", 0.0)) >= 0.85
        except Exception:
            return False

    def _attack_signature(self, finding: Dict[str, Any]) -> str:
        """Build a comparable signature for an attack.

        Args:
            finding: A finding dictionary.

        Returns:
            Signature string combining the attack class with its distinguishing
            parameters.
        """
        try:
            evidence = finding.get("evidence", {}) or {}
            parts = [str(finding.get("attack_class", "UNKNOWN"))]
            for key in ("corner", "patch_corner", "target_class", "spectral_class",
                        "to_class", "new_label", "block_size"):
                value = evidence.get(key)
                if value is not None:
                    parts.append(f"{key}={value}")
            return "|".join(parts)
        except Exception:
            return str(finding.get("attack_class", "UNKNOWN"))

    def _correlate_cover_up(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect provenance tampering co-occurring with substantive findings.

        Args:
            findings: All findings.

        Returns:
            Zero or one cover-up story.
        """
        try:
            concealment_classes = {
                AttackClass.INFERENCE_TAMPER.value,
                AttackClass.INFERENCE_REPLAY.value,
                AttackClass.CHAIN_BREAK.value,
                AttackClass.SIGNATURE_INVALID.value,
            }
            concealment = [f for f in findings if f["attack_class"] in concealment_classes]
            substantive = [f for f in findings
                           if f["_stage"] in ("DATA", "MODEL")]

            if not concealment or not substantive:
                return []

            confidence = clamp(noisy_or(
                [max(float(f.get("confidence", 0.0)) for f in concealment),
                 max(float(f.get("confidence", 0.0)) for f in substantive)]) * 0.82)

            classes = Counter(f["attack_class"] for f in concealment)
            return [{
                "story_id": "STORY-COVERUP",
                "title": "Provenance tampering alongside substantive compromise",
                "pattern": "COVER_UP",
                "attack_class": AttackClass.COVER_UP_ATTEMPT.value,
                "confidence": round(confidence, 4),
                "narrative": (
                    f"{len(concealment)} provenance integrity failures ("
                    + ", ".join(f"{count}x {name}" for name, count in classes.most_common())
                    + f") occur alongside {len(substantive)} findings in the data and/or "
                      "model stages. The audit trail exists to record what the system did; "
                      "damage to that trail occurring at the same time as evidence of "
                      "compromise is consistent with an attempt to remove the record of the "
                      "attack rather than with incidental corruption. Accidental corruption "
                      "does not preferentially occur where the evidence is."),
                "implication": (
                    "Assume the surviving inference records understate the scope of the "
                    "incident. Recover records from an independent copy if one exists, and "
                    "treat the affected period as unaccounted for."),
                "evidence_finding_ids": ([f.get("finding_id", "") for f in concealment][:20]
                                         + [f.get("finding_id", "") for f in substantive][:20]),
                "supporting_finding_count": len(concealment) + len(substantive),
                "stages": sorted({f["_stage"] for f in concealment + substantive}
                                 & set(STAGE_ORDER)),
            }]
        except Exception as exc:
            LOGGER.error("_correlate_cover_up failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    def _emit_finding(self, story: Dict[str, Any]) -> None:
        """Convert a story into a correlation Finding.

        Args:
            story: The story dictionary.
        """
        try:
            self.findings.append(Finding.create(
                attack_class=story["attack_class"],
                affected_asset=story["story_id"],
                confidence=float(story["confidence"]),
                reason=f"{story['title']}. {story['narrative']} {story['implication']}",
                evidence={
                    "pattern": story["pattern"],
                    "stages": story.get("stages", []),
                    "supporting_finding_count": story.get("supporting_finding_count", 0),
                    "evidence_finding_ids": story.get("evidence_finding_ids", []),
                    "contributors": story.get("contributors", []),
                    "target_class": story.get("target_class"),
                    "shared_attack_classes": story.get("shared_attack_classes", []),
                    # Correlation strength, not detection strength. Stated
                    # explicitly so nobody mistakes a narrative for a measurement.
                    "threshold": "correlation requires >=2 independent sources",
                },
                module=self.MODULE_NAME,
                detector=story["pattern"],
                asset_type="campaign",
                prefix="STORY",
            ))
            # related_findings is set after construction: it is not a
            # create() parameter, and the link back to the underlying
            # evidence is what makes a correlation finding auditable.
            self.findings[-1].related_findings = list(
                story.get("evidence_finding_ids", []))[:20]
        except Exception as exc:
            LOGGER.error("_emit_finding failed for %s: %s", story.get("story_id"), exc)

    def _campaign_assessment(self, findings: List[Dict[str, Any]],
                             stories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarise whether the evidence describes a coordinated campaign.

        Args:
            findings: All findings considered.
            stories: Correlation stories produced.

        Returns:
            Campaign assessment dictionary.
        """
        try:
            if not stories:
                return {
                    "campaign_detected": False,
                    "confidence": 0.0,
                    "assessment": (
                        f"{len(findings)} finding(s) were raised but no cross-module "
                        "correlation was established. The findings are best treated as "
                        "independent issues until further evidence links them."),
                    "patterns": [],
                }

            best = max(float(s.get("confidence", 0.0)) for s in stories)
            patterns = sorted({s["pattern"] for s in stories})
            stages = sorted({stage for s in stories for stage in s.get("stages", [])
                             if stage in STAGE_ORDER})

            return {
                "campaign_detected": True,
                "confidence": round(best, 4),
                "patterns": patterns,
                "stages_involved": stages,
                "story_count": len(stories),
                "assessment": (
                    f"{len(stories)} correlation pattern(s) identified ({', '.join(patterns)}) "
                    f"spanning the {', '.join(stages)} stage(s), with a maximum correlation "
                    f"confidence of {best:.0%}. The evidence is more consistent with a "
                    "deliberate, coordinated campaign than with independent defects."),
            }
        except Exception as exc:
            LOGGER.error("_campaign_assessment failed: %s", exc)
            return {"campaign_detected": False, "confidence": 0.0, "assessment": str(exc)}


def build_threat_story(module_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Convenience wrapper correlating module results into threat stories.

    Args:
        module_results: Mapping of module name to result dictionary.

    Returns:
        Correlation result dictionary.
    """
    try:
        return ThreatStoryEngine().correlate(module_results)
    except Exception as exc:
        LOGGER.error("build_threat_story failed: %s", exc)
        return {"module": "threat_story", "stories": [], "findings": [],
                "limitations": [str(exc)]}


__all__ = ["ThreatStoryEngine", "build_threat_story", "STAGE_OF_ATTACK"]
