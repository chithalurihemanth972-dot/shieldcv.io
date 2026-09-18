"""
COMMANDER'S BRIEFING GENERATOR.

Turns the structured output of every module into a short prose briefing an
officer can read in under a minute.

Two generation paths, in order of preference:

1. **Local LLM via Ollama** at ``http://localhost:11434`` (``llama3.2:3b`` or
   ``phi3:mini``). This is the ONLY network call SHIELD-CV ever makes, it is to
   loopback only, and it is optional.
2. **Deterministic template** — used whenever Ollama is absent, times out, or
   returns something unusable.

The template path is not a degraded afterthought: it is the accredited default,
and it produces a complete, correct briefing on its own. An air-gapped
deployment with no LLM loses fluency, not information.

**Every briefing carries the mandatory disclaimer.** LLM-generated text is
advisory, may contain errors, and must be verified by an analyst against the
structured findings, which remain the authoritative record.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

from src.config import get_config
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

DISCLAIMER = "AI-generated. Verify with analyst."

SYSTEM_PROMPT = (
    "You are a military cyber-security analyst writing a briefing for an Indian Army "
    "commanding officer. Write in plain, factual English. Be concise and specific. "
    "Use ONLY the facts provided; never invent findings, numbers, or attribution. "
    "If evidence is incomplete, say so plainly. No markdown, no bullet symbols, no "
    "preamble — just the briefing text in short paragraphs."
)


class BriefingGenerator:
    """Generates a commander's briefing from module results.

    Attributes:
        cfg: Effective configuration.
        host: Ollama endpoint.
        model: Preferred Ollama model name.
        enabled: Whether LLM generation may be attempted.
    """

    MODULE_NAME = "briefing"

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialise the generator and read its settings.

        Args:
            config: Optional configuration override.
        """
        self.cfg = config or get_config()
        section = self.cfg.section("briefing")
        self.host = str(section.get("ollama_host", "http://localhost:11434"))
        self.model = str(section.get("model", "llama3.2:3b"))
        self.fallback_models = list(section.get("fallback_models", ["phi3:mini"]))
        self.timeout = float(section.get("timeout_seconds", 45))
        self.enabled = bool(section.get("enabled", True))
        self.max_tokens = int(section.get("max_tokens", 600))

    # ------------------------------------------------------------------
    def generate(self, module_results: Dict[str, Dict[str, Any]],
                 threat_story: Optional[Dict[str, Any]] = None,
                 immunity: Optional[Dict[str, Any]] = None,
                 use_llm: Optional[bool] = None) -> Dict[str, Any]:
        """Generate the commander's briefing.

        Args:
            module_results: Mapping of module name to result dictionary.
            threat_story: Optional threat-story result.
            immunity: Optional immunity score result.
            use_llm: Force LLM on/off. Defaults to the configured value.

        Returns:
            Dictionary with ``briefing``, ``source``, ``disclaimer``,
            ``facts`` and ``limitations``.
        """
        started = time.time()
        result: Dict[str, Any] = {
            "module": self.MODULE_NAME, "briefing": "", "source": "template",
            "disclaimer": DISCLAIMER, "model": None, "facts": {},
            "limitations": [], "duration_seconds": 0.0,
        }

        try:
            facts = self._extract_facts(module_results, threat_story, immunity)
            result["facts"] = facts

            # The template briefing is always produced. It is the authoritative
            # fallback and also the sanity check against the LLM: if the model
            # returns something unusable, we already hold a correct briefing.
            template_text = self._template_briefing(facts)

            attempt_llm = self.enabled if use_llm is None else bool(use_llm)
            if attempt_llm:
                llm_text, model_used, error = self._ollama_briefing(facts)
                if llm_text:
                    result["briefing"] = llm_text
                    result["source"] = "ollama"
                    result["model"] = model_used
                    result["template_briefing"] = template_text
                else:
                    result["briefing"] = template_text
                    result["limitations"].append(
                        f"Local LLM unavailable ({error}); deterministic template briefing "
                        "used instead. Content is complete — only the phrasing differs.")
            else:
                result["briefing"] = template_text
                result["limitations"].append(
                    "LLM briefing disabled by configuration; template briefing used.")

            # The disclaimer is appended for BOTH paths. A template briefing is
            # still a machine-generated summary of machine findings, and the
            # structured findings remain the authoritative record either way.
            if DISCLAIMER not in result["briefing"]:
                result["briefing"] = f"{result['briefing']}\n\n[{DISCLAIMER}]"

            result["duration_seconds"] = round(time.time() - started, 2)
            LOGGER.info("Briefing generated via %s (%.1fs)",
                        result["source"], result["duration_seconds"])
            return result
        except Exception as exc:
            LOGGER.error("Briefing generation failed: %s", exc, exc_info=True)
            result["limitations"].append(f"Briefing generation error: {exc}")
            result["briefing"] = (
                "Briefing could not be generated. Refer to the structured findings "
                f"in the full report. Error: {exc}\n\n[{DISCLAIMER}]")
            result["duration_seconds"] = round(time.time() - started, 2)
            return result

    # ------------------------------------------------------------------
    def _extract_facts(self, module_results: Dict[str, Dict[str, Any]],
                       threat_story: Optional[Dict[str, Any]],
                       immunity: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Distil module results into the minimal fact set a briefing needs.

        Passing raw results to an LLM invites hallucination and blows the
        context window. Extracting a small, explicit fact set means the model
        has nothing to invent from.

        Args:
            module_results: Per-module results.
            threat_story: Optional threat-story result.
            immunity: Optional immunity result.

        Returns:
            Dictionary of briefing facts.
        """
        facts: Dict[str, Any] = {
            "modules_run": [], "modules_not_run": [], "total_findings": 0,
            "severity_counts": {}, "top_findings": [], "contributors_at_risk": [],
            "stories": [], "immunity": None, "verdicts": {},
        }
        try:
            expected = {
                "data_scanner": "training data",
                "model_auditor": "model weights",
                "crypto_chain": "inference records",
                "drift_detector": "operational drift",
            }
            all_findings: List[Dict[str, Any]] = []

            for module, description in expected.items():
                payload = (module_results or {}).get(module)
                if not isinstance(payload, dict) or not payload:
                    facts["modules_not_run"].append(description)
                    continue
                facts["modules_run"].append(description)
                findings = payload.get("findings", []) or []
                all_findings.extend(findings)
                summary = payload.get("summary", {}) or {}
                if summary.get("verdict"):
                    facts["verdicts"][description] = summary["verdict"]
                if module == "drift_detector" and payload.get("shift_type"):
                    facts["drift_type"] = payload["shift_type"]
                    facts["drift_recommendation"] = payload.get("recommendation", "")
                if module == "model_auditor":
                    facts["model_access_level"] = payload.get("access_level")

            facts["total_findings"] = len(all_findings)
            counts: Dict[str, int] = {}
            for finding in all_findings:
                key = str(finding.get("severity", "INFO"))
                counts[key] = counts.get(key, 0) + 1
            facts["severity_counts"] = counts

            ranked = sorted(all_findings,
                            key=lambda f: float(f.get("confidence", 0.0)), reverse=True)
            for finding in ranked[:5]:
                facts["top_findings"].append({
                    "attack_class": finding.get("attack_class"),
                    "asset": finding.get("affected_asset"),
                    "severity": finding.get("severity"),
                    "confidence": round(float(finding.get("confidence", 0.0)), 2),
                    "contributor": finding.get("contributor"),
                    "disposition": finding.get("disposition"),
                })

            data_payload = (module_results or {}).get("data_scanner", {}) or {}
            for name, data in (data_payload.get("contributor_risk", {}) or {}).items():
                if isinstance(data, dict) and float(data.get("risk", 0.0)) >= 0.60:
                    meta = self.cfg.contributor(name)
                    facts["contributors_at_risk"].append({
                        "name": meta.get("display_name", name),
                        "trust_level": meta.get("trust_level", "UNVERIFIED"),
                        "registered": meta.get("registered", False),
                        "risk": round(float(data.get("risk", 0.0)), 2),
                        "affected_samples": data.get("affected_samples", 0),
                    })

            if isinstance(threat_story, dict):
                assessment = threat_story.get("campaign_assessment", {}) or {}
                facts["campaign_detected"] = bool(assessment.get("campaign_detected"))
                facts["campaign_confidence"] = round(
                    float(assessment.get("confidence", 0.0)), 2)
                for story in (threat_story.get("stories", []) or [])[:4]:
                    facts["stories"].append({
                        "title": story.get("title"),
                        "pattern": story.get("pattern"),
                        "confidence": round(float(story.get("confidence", 0.0)), 2),
                        "implication": story.get("implication", ""),
                    })

            if isinstance(immunity, dict):
                facts["immunity"] = {
                    "score": immunity.get("score"),
                    "band": immunity.get("band"),
                    "pillars_assessed": immunity.get("pillars_assessed"),
                    "pillars_total": immunity.get("pillars_total"),
                    "weakest": min(
                        ((k, v.get("score", 100)) for k, v in
                         (immunity.get("components", {}) or {}).items()
                         if v.get("assessed")),
                        key=lambda kv: kv[1], default=(None, None))[0],
                    "recommendations": (immunity.get("recommendations", []) or [])[:3],
                }
            return facts
        except Exception as exc:
            LOGGER.error("_extract_facts failed: %s", exc)
            return facts

    # ------------------------------------------------------------------
    def _ollama_briefing(self, facts: Dict[str, Any]) -> tuple:
        """Attempt to generate the briefing with a local Ollama model.

        Args:
            facts: Extracted fact set.

        Returns:
            Tuple of ``(text, model_used, error)``. ``text`` is empty on failure.
        """
        available = self._available_models()
        if available is None:
            return "", None, "Ollama not reachable on localhost:11434"

        candidates = [self.model, *self.fallback_models]
        usable = [m for m in candidates if self._model_present(m, available)]
        if not usable:
            return "", None, (f"no suitable model installed (looked for "
                              f"{', '.join(candidates)}; found "
                              f"{', '.join(available) if available else 'none'})")

        prompt = self._build_prompt(facts)
        for model_name in usable:
            try:
                payload = json.dumps({
                    "model": model_name,
                    "prompt": prompt,
                    "system": SYSTEM_PROMPT,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": self.max_tokens},
                }).encode("utf-8")

                request = urllib.request.Request(
                    f"{self.host}/api/generate", data=payload,
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))

                text = str(body.get("response", "")).strip()
                # Guard against a model that returns an apology, an empty
                # string, or a single truncated line. Anything unusable falls
                # through to the template rather than being published.
                if len(text) < 120:
                    LOGGER.warning("Ollama model %s returned %d chars; rejecting",
                                   model_name, len(text))
                    continue
                return text, model_name, ""
            except urllib.error.URLError as exc:
                LOGGER.warning("Ollama request failed for %s: %s", model_name, exc)
            except Exception as exc:
                LOGGER.warning("Ollama generation error for %s: %s", model_name, exc)
        return "", None, "all candidate models failed to produce usable output"

    def _available_models(self) -> Optional[List[str]]:
        """List models installed in the local Ollama instance.

        Returns:
            List of model names, or ``None`` when Ollama is unreachable.
        """
        try:
            request = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(request, timeout=min(5.0, self.timeout)) as response:
                body = json.loads(response.read().decode("utf-8"))
            return [str(m.get("name", "")) for m in body.get("models", [])]
        except Exception as exc:
            LOGGER.info("Ollama unavailable (expected in air-gapped deployment): %s", exc)
            return None

    def _model_present(self, wanted: str, available: Sequence[str]) -> bool:
        """Check whether a model is installed, tolerating tag suffixes.

        Args:
            wanted: Desired model name.
            available: Installed model names.

        Returns:
            ``True`` when a matching model is present.
        """
        try:
            base = wanted.split(":")[0]
            return any(name == wanted or name.split(":")[0] == base for name in available)
        except Exception:
            return False

    def _build_prompt(self, facts: Dict[str, Any]) -> str:
        """Construct the LLM prompt from the fact set.

        Args:
            facts: Extracted facts.

        Returns:
            Prompt string.
        """
        try:
            return (
                "Write a commander's briefing of at most 250 words on the integrity "
                "assessment of an Army computer vision pipeline.\n\n"
                "Cover, in this order: (1) the overall verdict and what it means for "
                "operational reliance; (2) the most serious specific findings; (3) which "
                "contributors or assets are implicated; (4) what was NOT assessed; "
                "(5) the immediate recommended action.\n\n"
                "State uncertainty where the data shows it. Do not add findings that are "
                "not listed below.\n\n"
                f"ASSESSMENT DATA (JSON):\n{json.dumps(facts, indent=2, default=str)}")
        except Exception as exc:
            LOGGER.error("_build_prompt failed: %s", exc)
            return json.dumps(facts, default=str)

    # ------------------------------------------------------------------
    def _template_briefing(self, facts: Dict[str, Any]) -> str:
        """Build the deterministic template briefing.

        Args:
            facts: Extracted facts.

        Returns:
            Briefing text.
        """
        try:
            lines: List[str] = ["COMMANDER'S BRIEFING — CV PIPELINE INTEGRITY", ""]

            immunity = facts.get("immunity") or {}
            if immunity.get("score") is not None:
                lines.append(
                    f"OVERALL POSTURE: Adversarial Immunity Score "
                    f"{immunity['score']}/100 ({immunity.get('band', 'UNKNOWN')}), based on "
                    f"{immunity.get('pillars_assessed', 0)} of "
                    f"{immunity.get('pillars_total', 4)} assurance pillars.")
            else:
                lines.append("OVERALL POSTURE: immunity score not computed for this run.")

            counts = facts.get("severity_counts", {})
            critical = counts.get("CRITICAL", 0)
            high = counts.get("HIGH", 0)
            if facts.get("total_findings"):
                lines.append(
                    f"A total of {facts['total_findings']} finding(s) were raised: "
                    f"{critical} CRITICAL, {high} HIGH, "
                    f"{counts.get('MEDIUM', 0)} MEDIUM, {counts.get('LOW', 0)} LOW.")
            else:
                lines.append("No findings were raised by the modules that ran.")
            lines.append("")

            if facts.get("campaign_detected"):
                lines.append(
                    f"ASSESSMENT: Evidence indicates a COORDINATED CAMPAIGN "
                    f"(correlation confidence {facts.get('campaign_confidence', 0):.0%}), "
                    "not a set of unrelated defects.")
                for story in facts.get("stories", []):
                    lines.append(f"  - {story['title']} ({story['confidence']:.0%}). "
                                 f"{story.get('implication', '')}")
                lines.append("")

            if facts.get("top_findings"):
                lines.append("MOST SERIOUS FINDINGS:")
                for finding in facts["top_findings"]:
                    attribution = (f", attributed to {finding['contributor']}"
                                   if finding.get("contributor") else "")
                    lines.append(
                        f"  - [{finding['severity']}] {finding['attack_class']} affecting "
                        f"{finding['asset']}{attribution} "
                        f"(confidence {finding['confidence']:.0%}, "
                        f"disposition {finding.get('disposition', 'REVIEW')}).")
                lines.append("")

            if facts.get("contributors_at_risk"):
                lines.append("CONTRIBUTORS REQUIRING ACTION:")
                for entry in facts["contributors_at_risk"]:
                    registered = ("" if entry.get("registered")
                                  else " — NOT in the contributor registry")
                    lines.append(
                        f"  - {entry['name']} [{entry['trust_level']}{registered}]: "
                        f"risk {entry['risk']:.2f}, {entry['affected_samples']} "
                        "affected sample(s).")
                lines.append("")

            if facts.get("drift_type"):
                lines.append(f"OPERATIONAL DRIFT: {facts['drift_type']}. "
                             f"{facts.get('drift_recommendation', '')}")
                lines.append("")

            # Stating the coverage gap is mandatory. A briefing that lists only
            # what was found invites the reader to assume everything else is
            # sound, which is the single most dangerous misreading of this tool.
            if facts.get("modules_not_run"):
                lines.append(
                    "NOT ASSESSED: " + ", ".join(facts["modules_not_run"])
                    + ". No conclusion about these areas can be drawn from this report.")
            else:
                lines.append("COVERAGE: all four assurance areas were assessed.")
            if facts.get("model_access_level"):
                lines.append(f"Model access level achieved: {facts['model_access_level']}.")
            lines.append("")

            lines.append("RECOMMENDED ACTION:")
            recommendations = (immunity.get("recommendations")
                               or ["Review the structured findings and adjudicate each "
                                   "quarantine recommendation."])
            for recommendation in recommendations:
                lines.append(f"  - {recommendation}")

            return "\n".join(lines).strip()
        except Exception as exc:
            LOGGER.error("_template_briefing failed: %s", exc)
            return f"Template briefing failed: {exc}"


def generate_briefing(module_results: Dict[str, Dict[str, Any]],
                      threat_story: Optional[Dict[str, Any]] = None,
                      immunity: Optional[Dict[str, Any]] = None,
                      use_llm: Optional[bool] = None) -> Dict[str, Any]:
    """Convenience wrapper generating a commander's briefing.

    Args:
        module_results: Mapping of module name to result dictionary.
        threat_story: Optional threat-story result.
        immunity: Optional immunity score result.
        use_llm: Force LLM usage on or off.

    Returns:
        Briefing result dictionary.
    """
    try:
        return BriefingGenerator().generate(module_results, threat_story=threat_story,
                                            immunity=immunity, use_llm=use_llm)
    except Exception as exc:
        LOGGER.error("generate_briefing failed: %s", exc)
        return {"module": "briefing", "briefing": f"Briefing unavailable: {exc}",
                "source": "error", "disclaimer": DISCLAIMER, "limitations": [str(exc)]}


__all__ = ["BriefingGenerator", "generate_briefing", "DISCLAIMER"]
