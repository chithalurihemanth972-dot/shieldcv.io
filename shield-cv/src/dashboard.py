"""SHIELD-CV Streamlit operations dashboard.

A dark, military-styled console over the same engines the CLI drives. Five
tabs mirror the assurance workflow:

* **DATA SCANNER**  — contributor dataset integrity
* **MODEL AUDIT**   — weight, backdoor and behavioural analysis
* **CRYPTO LOCK**   — provenance chain with a live, editable tamper demo
* **DRIFT MONITOR** — operational vs adversarial distribution shift
* **THREAT STORY**  — correlated narratives, immunity gauge and LLM briefing

Run with::

    streamlit run src/dashboard.py

Design notes. Scans are expensive, so every result is held in
``st.session_state`` and nothing recomputes on an unrelated widget interaction.
Long operations are wrapped in spinners with explicit status text, because a
silent multi-second pause reads as a hang. Every panel that reports a verdict
also renders its limitations — the dashboard must never present a conclusion
more confident than the evidence behind it.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow `streamlit run src/dashboard.py` from the project root without install.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import get_config
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

SEVERITY_COLOR = {
    "CRITICAL": "#ff3b30",
    "HIGH": "#ff9500",
    "MEDIUM": "#ffcc00",
    "LOW": "#32ade6",
    "INFO": "#8e8e93",
}

VERDICT_COLOR = {
    "COMPROMISED": "#ff3b30",
    "SUSPICIOUS": "#ff9500",
    "CAUTION": "#ffcc00",
    "MINOR_ANOMALIES": "#ffcc00",
    "CLEAN": "#30d158",
    "NOT_ASSESSED": "#8e8e93",
    "UNKNOWN": "#8e8e93",
}

BAND_COLOR = {
    "HARDENED": "#30d158",
    "ADEQUATE": "#9ede73",
    "MARGINAL": "#ffcc00",
    "COMPROMISED": "#ff3b30",
}

DARK_CSS = """
<style>
  .stApp { background-color: #0b0f14; color: #d7e0ea; }
  section[data-testid="stSidebar"] { background-color: #111822; }
  h1, h2, h3, h4 { color: #7fd1ff !important; letter-spacing: 0.5px; }
  .shield-banner {
      border: 1px solid #1f6feb; border-left: 5px solid #1f6feb;
      background: linear-gradient(90deg, #0d1b2a 0%, #0b0f14 100%);
      padding: 14px 18px; border-radius: 6px; margin-bottom: 14px;
  }
  .shield-title { font-size: 26px; font-weight: 700; color: #7fd1ff; margin: 0; }
  .shield-sub { font-size: 13px; color: #8fa3b8; margin: 2px 0 0 0; }
  .verdict-chip {
      display: inline-block; padding: 6px 16px; border-radius: 4px;
      font-weight: 700; font-size: 15px; color: #0b0f14;
  }
  .metric-card {
      background: #111822; border: 1px solid #1e2a38; border-radius: 6px;
      padding: 12px 14px; margin-bottom: 8px;
  }
  .metric-value { font-size: 26px; font-weight: 700; }
  .metric-label { font-size: 12px; color: #8fa3b8; text-transform: uppercase; }
  .limitation {
      border-left: 3px solid #ffcc00; background: #1a1710;
      padding: 7px 11px; margin: 5px 0; font-size: 13px; color: #e8d9a0;
  }
  .covered {
      border-left: 3px solid #30d158; background: #0f1a12;
      padding: 7px 11px; margin: 5px 0; font-size: 13px; color: #b6e8c2;
  }
  .stDataFrame { border: 1px solid #1e2a38; }
</style>
"""


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _state(key: str, default: Any = None) -> Any:
    """Read a value from Streamlit session state.

    Args:
        key: State key.
        default: Value returned when the key is absent.

    Returns:
        The stored value or ``default``.
    """
    return st.session_state.get(key, default)


def _banner() -> None:
    """Render the dashboard header."""
    try:
        cfg = get_config()
        name = cfg.get("project.name", "SHIELD-CV")
        full = cfg.get("project.full_name", "")
        version = cfg.get("project.version", "1.0.0")
        classification = cfg.get("project.classification", "RESTRICTED")
    except Exception:
        name, full, version, classification = "SHIELD-CV", "", "1.0.0", "RESTRICTED"

    st.markdown(
        f"""<div class="shield-banner">
              <p class="shield-title">{name} &nbsp;<span style="font-size:14px;
                 color:#8fa3b8;">v{version}</span></p>
              <p class="shield-sub">{full}</p>
              <p class="shield-sub">{classification} &nbsp;·&nbsp; 100% offline
                 &nbsp;·&nbsp; no data leaves this host</p>
            </div>""",
        unsafe_allow_html=True)


def _verdict_chip(verdict: str) -> str:
    """Build an HTML chip for a verdict.

    Args:
        verdict: Verdict label.

    Returns:
        HTML string.
    """
    colour = VERDICT_COLOR.get(str(verdict).upper(), "#8e8e93")
    return (f'<span class="verdict-chip" style="background:{colour};">'
            f'{verdict}</span>')


def _metric(label: str, value: Any, colour: str = "#d7e0ea") -> str:
    """Build an HTML metric card.

    Args:
        label: Metric caption.
        value: Metric value.
        colour: Value colour.

    Returns:
        HTML string.
    """
    return (f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="color:{colour};">{value}</div></div>')


def _show_findings(findings: List[Dict[str, Any]], key: str) -> None:
    """Render a findings table with a severity filter and evidence drill-down.

    Args:
        findings: Finding dictionaries.
        key: Unique widget key prefix.
    """
    if not findings:
        st.success("No findings raised.")
        return

    severities = [s for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")
                  if any(str(f.get("severity")).upper() == s for f in findings)]
    chosen = st.multiselect("Filter by severity", severities, default=severities,
                            key=f"{key}_filter")
    subset = [f for f in findings if str(f.get("severity")).upper() in chosen]

    st.caption(f"{len(subset)} of {len(findings)} findings")
    st.dataframe(
        [{
            "ID": f.get("finding_id", ""),
            "Severity": f.get("severity", ""),
            "Conf": round(float(f.get("confidence", 0.0)), 2),
            "Attack class": f.get("attack_class", ""),
            "Asset": str(f.get("affected_asset", ""))[:46],
            "Contributor": f.get("contributor") or "—",
            "Disposition": f.get("recommended_disposition") or f.get("disposition", ""),
        } for f in subset],
        use_container_width=True, hide_index=True)

    if subset:
        options = [f.get("finding_id", "") for f in subset]
        selected = st.selectbox("Inspect a finding", options, key=f"{key}_detail")
        record = next((f for f in subset if f.get("finding_id") == selected), None)
        if record:
            st.markdown(f"**Reason.** {record.get('reason', '')}")
            st.json(record.get("evidence") or {})


def _show_limitations(limitations: List[str],
                      coverage: Optional[Dict[str, Any]] = None) -> None:
    """Render the coverage statement and limitations.

    Args:
        limitations: Limitation strings.
        coverage: Optional coverage statement.
    """
    if coverage:
        supported = coverage.get("supported") or []
        unsupported = coverage.get("unsupported") or []
        if supported:
            st.markdown("##### Detected in this run")
            for item in supported[:12]:
                st.markdown(f'<div class="covered">+ {item}</div>',
                            unsafe_allow_html=True)
        if unsupported:
            st.markdown("##### NOT assessed — blind spots")
            for item in unsupported[:12]:
                st.markdown(f'<div class="limitation">− {item}</div>',
                            unsafe_allow_html=True)

    if limitations:
        st.markdown(f"##### Limitations ({len(limitations)})")
        for item in limitations[:15]:
            st.markdown(f'<div class="limitation">! {item}</div>',
                        unsafe_allow_html=True)


def _summary_row(summary: Dict[str, Any]) -> None:
    """Render verdict, risk and severity counts as metric cards.

    Args:
        summary: Summary dictionary.
    """
    if not summary:
        return
    counts = summary.get("by_severity") or {}
    verdict = str(summary.get("verdict", "UNKNOWN"))
    columns = st.columns(4)
    with columns[0]:
        st.markdown(_verdict_chip(verdict), unsafe_allow_html=True)
    with columns[1]:
        st.markdown(_metric("Risk score",
                            f"{float(summary.get('risk_score', 0.0)):.3f}",
                            VERDICT_COLOR.get(verdict.upper(), "#d7e0ea")),
                    unsafe_allow_html=True)
    with columns[2]:
        st.markdown(_metric("Findings", summary.get("total_findings", 0)),
                    unsafe_allow_html=True)
    with columns[3]:
        critical = int(counts.get("CRITICAL", 0) or 0)
        high = int(counts.get("HIGH", 0) or 0)
        st.markdown(_metric("Critical / High", f"{critical} / {high}",
                            SEVERITY_COLOR["CRITICAL"] if critical else "#d7e0ea"),
                    unsafe_allow_html=True)


def _demo_path(*parts: str) -> str:
    """Build a path inside the demo directory.

    Args:
        *parts: Path segments below the demo root.

    Returns:
        String path.
    """
    try:
        return str(get_config().path("demo").joinpath(*parts))
    except Exception:
        return str(ROOT.joinpath("demo", *parts))


# ---------------------------------------------------------------------------
# tab 1 — data scanner
# ---------------------------------------------------------------------------
def tab_data_scanner() -> None:
    """Render the dataset integrity tab."""
    st.subheader("Data Scanner — contributed dataset integrity")
    st.caption("Trigger injection · label flipping · near-duplicate flooding · "
               "out-of-distribution · contributor risk")

    columns = st.columns([3, 1, 1])
    with columns[0]:
        path = st.text_input("Dataset root", _demo_path("data", "poisoned"),
                             key="ds_path")
    with columns[1]:
        max_images = st.number_input("Max images (0 = all)", 0, 10000, 0,
                                     key="ds_max")
    with columns[2]:
        st.write("")
        run = st.button("RUN SCAN", type="primary", use_container_width=True)

    if run:
        try:
            from src.scanners.data_scanner import DataIntegrityEngine
            with st.spinner(f"Scanning {path} — embedding images and running "
                            f"five detectors…"):
                result = DataIntegrityEngine().scan(
                    path, max_images=int(max_images) or None)
            st.session_state["data_result"] = result
        except Exception as exc:
            st.error(f"Scan failed: {type(exc).__name__}: {exc}")
            LOGGER.error("dashboard scan failed: %s\n%s", exc, traceback.format_exc())

    result = _state("data_result")
    if not result:
        st.info("Run a scan to populate this tab.")
        return

    dataset = result.get("dataset") or {}
    if int(dataset.get("num_samples", 0) or 0) == 0:
        st.error("NOT ASSESSED — no readable images were found. "
                 "This is not a clean result.")
        _show_limitations(result.get("limitations") or [])
        return

    _summary_row(result.get("summary") or {})
    st.caption(f"{dataset.get('num_samples', 0)} samples · "
               f"{dataset.get('num_classes', 0)} classes · "
               f"format {dataset.get('format', 'unknown')}")

    risks = result.get("contributor_risk") or {}
    if risks:
        st.markdown("#### Contributor risk")
        st.dataframe(
            [{
                "Contributor": name,
                "Risk": round(float(entry.get("risk", 0.0)), 3),
                "Findings": entry.get("findings", 0),
                "Samples": entry.get("samples", 0),
                "Assessment": str(entry.get("assessment", ""))[:90],
            } for name, entry in sorted(
                risks.items(), key=lambda kv: -float(kv[1].get("risk", 0.0)))],
            use_container_width=True, hide_index=True)

    st.markdown("#### Findings")
    _show_findings(result.get("findings") or [], "data")
    _show_limitations(result.get("limitations") or [])


# ---------------------------------------------------------------------------
# tab 2 — model audit
# ---------------------------------------------------------------------------
def tab_model_audit() -> None:
    """Render the model integrity tab."""
    st.subheader("Model Audit — contributed model integrity")
    st.caption("Weight fingerprint · per-layer statistics · Neural Cleanse · "
               "activation clustering · black-box fingerprinting")

    columns = st.columns([3, 3, 1])
    with columns[0]:
        model_path = st.text_input(
            "Model", _demo_path("models", "backdoored_model_scripted.pt"),
            key="ma_model")
    with columns[1]:
        reference = st.text_input(
            "Trusted reference (optional)",
            _demo_path("models", "clean_model_scripted.pt"), key="ma_ref")
    with columns[2]:
        st.write("")
        run = st.button("RUN AUDIT", type="primary", use_container_width=True)

    skip = st.checkbox("Skip Neural Cleanse (much faster)", value=False,
                       key="ma_skip")

    if run:
        try:
            from src.scanners.model_auditor import ModelIntegrityEngine
            with st.spinner("Auditing model — this runs a trigger search and "
                            "may take a minute…"):
                result = ModelIntegrityEngine().audit(
                    model_path, reference_path=reference or None,
                    run_neural_cleanse=not skip)
            st.session_state["model_result"] = result
        except Exception as exc:
            st.error(f"Audit failed: {type(exc).__name__}: {exc}")
            LOGGER.error("dashboard audit failed: %s\n%s", exc, traceback.format_exc())

    result = _state("model_result")
    if not result:
        st.info("Run an audit to populate this tab.")
        return

    access = str(result.get("access_level", "UNKNOWN"))
    if access.upper() == "UNAVAILABLE":
        st.error("NOT ASSESSED — the model could not be opened.")
        _show_limitations(result.get("limitations") or [])
        return

    _summary_row(result.get("summary") or {})
    columns = st.columns(3)
    with columns[0]:
        st.markdown(_metric("Access level", access,
                            "#30d158" if "WHITE" in access.upper() else "#ffcc00"),
                    unsafe_allow_html=True)
    with columns[1]:
        st.markdown(_metric("Confidence",
                            f"{float(result.get('confidence', 0.0)):.2f}"),
                    unsafe_allow_html=True)
    with columns[2]:
        cleanse = result.get("neural_cleanse") or {}
        flagged = cleanse.get("flagged_classes") or []
        st.markdown(_metric("Backdoor classes",
                            ", ".join(map(str, flagged)) if flagged else "none",
                            "#ff3b30" if flagged else "#30d158"),
                    unsafe_allow_html=True)

    cleanse = result.get("neural_cleanse") or {}
    if cleanse.get("available"):
        st.markdown("#### Neural Cleanse")
        st.caption(f"Mode {cleanse.get('mode', '?')} · "
                   f"{cleanse.get('classes_scanned', 0)} class(es) scanned · "
                   f"median mask L1 {cleanse.get('median_mask_l1', 'n/a')} · "
                   f"probe quality {cleanse.get('probe_quality', 'n/a')}")
        candidates = cleanse.get("candidates") or []
        if candidates:
            st.dataframe(candidates, use_container_width=True, hide_index=True)

    weights = result.get("weight_statistics") or {}
    layers = weights.get("layers") or []
    if layers:
        st.markdown("#### Per-layer weight statistics")
        st.dataframe(layers, use_container_width=True, hide_index=True)

    st.markdown("#### Findings")
    _show_findings(result.get("findings") or [], "model")
    _show_limitations(result.get("limitations") or [],
                      result.get("coverage_statement"))


# ---------------------------------------------------------------------------
# tab 3 — crypto lock
# ---------------------------------------------------------------------------
def tab_crypto_lock() -> None:
    """Render the provenance chain tab, including the live tamper demo."""
    st.subheader("Crypto Lock — inference provenance chain")
    st.caption("Ed25519 signatures · SHA-256 hash chain · nonce replay detection")

    columns = st.columns([3, 1])
    with columns[0]:
        records_path = st.text_input(
            "Records file", _demo_path("records", "clean_chain.json"),
            key="cc_path")
    with columns[1]:
        st.write("")
        run = st.button("VERIFY CHAIN", type="primary", use_container_width=True)

    if run:
        try:
            from src.scanners.crypto_chain import InferenceProvenanceEngine
            with st.spinner("Verifying chain links, signatures and nonces…"):
                st.session_state["crypto_result"] = \
                    InferenceProvenanceEngine().verify(records_path)
            st.session_state["crypto_source"] = records_path
        except Exception as exc:
            st.error(f"Verification failed: {type(exc).__name__}: {exc}")
            LOGGER.error("dashboard verify failed: %s\n%s", exc, traceback.format_exc())

    result = _state("crypto_result")
    if result:
        valid = bool(result.get("chain_valid", False))
        columns = st.columns(3)
        with columns[0]:
            st.markdown(_verdict_chip("VALID" if valid else "BROKEN")
                        .replace("#8e8e93", "#30d158" if valid else "#ff3b30"),
                        unsafe_allow_html=True)
        with columns[1]:
            st.markdown(_metric("Records", result.get("num_records", 0)),
                        unsafe_allow_html=True)
        with columns[2]:
            first = result.get("first_break_index")
            st.markdown(_metric("First break",
                                "none" if first is None else f"index {first}",
                                "#30d158" if first is None else "#ff3b30"),
                        unsafe_allow_html=True)

        _summary_row(result.get("summary") or {})
        st.markdown("#### Findings")
        _show_findings(result.get("findings") or [], "crypto")
        _show_limitations(result.get("limitations") or [])

    st.divider()
    st.markdown("### Live tamper demonstration")
    st.caption("Edit any field of a signed record and re-verify. The chain is "
               "designed so that *any* change breaks it — and so that an "
               "attacker who recomputes the hash still fails the signature check.")
    _tamper_demo()


def _tamper_demo() -> None:
    """Render the interactive record-editing tamper demo."""
    try:
        source = st.text_input("Chain to tamper with",
                               _demo_path("records", "clean_chain.json"),
                               key="tamper_src")
        if st.button("LOAD CHAIN", key="tamper_load"):
            from src.crypto.chain import _as_dict
            from src.loaders.record_loader import load_records

            loaded = load_records(source)
            # load_records yields InferenceRecord dataclasses (or a wrapper on
            # older paths). Normalise to plain dicts so fields can be edited
            # freely in the UI and fed straight back into verify_chain.
            sequence = (loaded if isinstance(loaded, list)
                        else getattr(loaded, "records", None) or [])
            st.session_state["tamper_records"] = [
                dict(r) if isinstance(r, dict) else dict(_as_dict(r))
                for r in sequence]
            st.session_state["tamper_result"] = None

        records = _state("tamper_records")
        if not records:
            st.info("Load a chain to begin.")
            return

        st.caption(f"{len(records)} records loaded.")
        index = st.number_input("Record index to edit", 0, max(len(records) - 1, 0),
                                min(5, len(records) - 1), key="tamper_idx")
        record = records[int(index)]

        editable = [k for k in ("output", "output_hash", "input_hash",
                                "model_hash", "config_hash", "timestamp_ns",
                                "sequence_number", "nonce", "prev_record_hash",
                                "signature", "record_id", "contributor")
                    if k in record]
        field = st.selectbox("Field", editable, key="tamper_field")
        current = json.dumps(record.get(field)) if isinstance(
            record.get(field), (dict, list)) else str(record.get(field))
        st.text_area("Current value", current, height=70, disabled=True,
                     key="tamper_cur")
        replacement = st.text_input("New value", current, key="tamper_new")

        columns = st.columns(3)
        with columns[0]:
            apply_edit = st.button("APPLY EDIT & RE-VERIFY", type="primary",
                                   key="tamper_apply")
        with columns[1]:
            rehash = st.checkbox("Attacker also recomputes the record hash",
                                 key="tamper_rehash")
        with columns[2]:
            if st.button("RESET CHAIN", key="tamper_reset"):
                st.session_state["tamper_records"] = None
                st.session_state["tamper_result"] = None

        if apply_edit:
            tampered = [dict(r) for r in records]
            target = tampered[int(index)]
            try:
                target[field] = json.loads(replacement)
            except Exception:
                target[field] = replacement

            if rehash:
                # Simulate a capable attacker who repairs the hash they broke.
                # The Ed25519 signature still fails, which is the whole point of
                # signing rather than merely hashing.
                from src.crypto.chain import compute_record_hash
                target["record_hash"] = compute_record_hash(target)

            from src.crypto.chain import verify_chain
            outcome = verify_chain(tampered)
            st.session_state["tamper_result"] = {
                "valid": bool(getattr(outcome, "valid", False)),
                "first_break_index": getattr(outcome, "first_break_index", None),
                "breaks": [
                    {"index": getattr(b, "index", None),
                     "type": str(getattr(b, "break_type", "")),
                     "detail": str(getattr(b, "detail", ""))[:160]}
                    for b in (getattr(outcome, "breaks", None) or [])][:12],
            }

        outcome = _state("tamper_result")
        if outcome:
            if outcome["valid"]:
                st.success("Chain still verifies — that edit was not detected.")
            else:
                st.error(f"TAMPER DETECTED — chain broken at index "
                         f"{outcome['first_break_index']}")
                st.dataframe(outcome["breaks"], use_container_width=True,
                             hide_index=True)
                st.caption("Re-hashing the edited record repairs the link but "
                           "cannot forge the Ed25519 signature, so the tamper "
                           "still surfaces.")
    except Exception as exc:
        st.error(f"Tamper demo failed: {type(exc).__name__}: {exc}")
        LOGGER.error("tamper demo failed: %s\n%s", exc, traceback.format_exc())


# ---------------------------------------------------------------------------
# tab 4 — drift monitor
# ---------------------------------------------------------------------------
def tab_drift_monitor() -> None:
    """Render the distribution shift tab."""
    st.subheader("Drift Monitor — operational vs adversarial shift")
    st.caption("Five pixel properties · MMD · centroid drift · variance ratio")

    columns = st.columns([3, 3, 1])
    with columns[0]:
        baseline = st.text_input("Baseline corpus", _demo_path("data", "clean"),
                                 key="dr_base")
    with columns[1]:
        current = st.text_input("Current corpus",
                                _demo_path("data", "drift", "injected"),
                                key="dr_cur")
    with columns[2]:
        st.write("")
        run = st.button("COMPARE", type="primary", use_container_width=True)

    max_images = st.slider("Max images per corpus", 30, 300, 120, key="dr_max")

    if run:
        try:
            from src.scanners.drift_detector import DistributionShiftDetector
            with st.spinner("Comparing pixel and feature distributions…"):
                st.session_state["drift_result"] = \
                    DistributionShiftDetector().compare(
                        baseline, current, max_images=int(max_images))
        except Exception as exc:
            st.error(f"Drift comparison failed: {type(exc).__name__}: {exc}")
            LOGGER.error("dashboard drift failed: %s\n%s", exc, traceback.format_exc())

    result = _state("drift_result")
    if not result:
        st.info("Run a comparison to populate this tab.")
        return

    shift_type = str(result.get("shift_type", "UNKNOWN"))
    colour = ("#ff3b30" if "SUSPICIOUS" in shift_type else
              "#ffcc00" if "MIXED" in shift_type else "#30d158")
    columns = st.columns(4)
    with columns[0]:
        st.markdown(_metric("Shift type", shift_type.replace("_", " "), colour),
                    unsafe_allow_html=True)
    with columns[1]:
        st.markdown(_metric("Risk", f"{float(result.get('risk_score', 0.0)):.3f}",
                            colour), unsafe_allow_html=True)
    with columns[2]:
        st.markdown(_metric("Severity", result.get("severity", "NONE")),
                    unsafe_allow_html=True)
    with columns[3]:
        st.markdown(_metric("Confidence",
                            f"{float(result.get('confidence', 0.0)):.2f}"),
                    unsafe_allow_html=True)

    st.markdown(f"**Characterization.** {result.get('characterization', '')}")
    st.info(f"**Recommendation.** {result.get('recommendation', '')}")

    pixel = (result.get("pixel_analysis") or {}).get("properties") or {}
    if pixel:
        st.markdown("#### Pixel-level properties")
        st.dataframe(
            [{
                "Property": name,
                "Baseline": round(float(entry.get("baseline_mean", 0.0)), 4),
                "Current": round(float(entry.get("current_mean", 0.0)), 4),
                "Z-score": round(float(entry.get("zscore", 0.0)), 3),
                "Rel. change": round(float(entry.get("relative_change", 0.0)), 4),
                "Shifted": "yes" if entry.get("shifted") else "no",
            } for name, entry in pixel.items()],
            use_container_width=True, hide_index=True)

    feature = result.get("feature_analysis") or {}
    if feature:
        st.markdown("#### Feature-level analysis")
        st.json({k: v for k, v in feature.items() if not isinstance(v, (list, dict))})

    signatures = result.get("natural_signatures") or []
    if signatures:
        st.markdown("#### Matched natural signatures")
        st.write(signatures)

    st.markdown("#### Findings")
    _show_findings(result.get("findings") or [], "drift")
    _show_limitations(result.get("limitations") or [])


# ---------------------------------------------------------------------------
# tab 5 — threat story
# ---------------------------------------------------------------------------
def tab_threat_story() -> None:
    """Render the correlation, immunity and briefing tab."""
    st.subheader("Threat Story — correlation, immunity and command briefing")
    st.caption("Cross-module correlation · Adversarial Immunity Score · "
               "local LLM briefing with deterministic fallback")

    available = {
        "data_scanner": _state("data_result"),
        "model_auditor": _state("model_result"),
        "crypto_chain": _state("crypto_result"),
        "drift_detector": _state("drift_result"),
    }
    present = {k: v for k, v in available.items() if v}

    if not present:
        st.warning("No module results yet. Run at least one scan in the other "
                   "tabs, then return here. Correlation across two or more "
                   "modules produces far stronger conclusions than any single "
                   "module alone.")
        return

    st.write("Modules available for correlation: "
             + ", ".join(f"`{k}`" for k in present))
    missing = [k for k in available if k not in present]
    if missing:
        st.info("Not yet run (these pillars will count as NOT ASSESSED, not as "
                "clean): " + ", ".join(f"`{k}`" for k in missing))

    columns = st.columns([1, 1, 2])
    with columns[0]:
        use_llm = st.checkbox("Use local LLM", value=False, key="ts_llm",
                              help="Queries Ollama on localhost. Falls back to "
                                   "a deterministic template when unavailable.")
    with columns[1]:
        persist = st.checkbox("Save report", value=True, key="ts_save")
    with columns[2]:
        st.write("")
        run = st.button("CORRELATE & BRIEF", type="primary",
                        use_container_width=True)

    if run:
        try:
            from src.intelligence.briefing import generate_briefing
            from src.intelligence.immunity import compute_immunity_score
            from src.intelligence.threat_story import build_threat_story
            from src.reporting.report_generator import generate_report

            with st.spinner("Correlating findings across modules…"):
                story = build_threat_story(present)
                immunity = compute_immunity_score(present, story)
            with st.spinner("Generating command briefing…"):
                briefing = generate_briefing(present, story, immunity,
                                             use_llm=use_llm)
            with st.spinner("Building assurance report…"):
                report = generate_report(present, story, immunity, briefing,
                                         target={"modules": list(present)},
                                         persist=persist)
            st.session_state.update({"story": story, "immunity": immunity,
                                     "briefing": briefing, "report": report})
        except Exception as exc:
            st.error(f"Correlation failed: {type(exc).__name__}: {exc}")
            LOGGER.error("dashboard correlate failed: %s\n%s", exc,
                         traceback.format_exc())

    immunity = _state("immunity")
    story = _state("story")
    briefing = _state("briefing")
    report = _state("report")

    if not immunity:
        st.info("Press **CORRELATE & BRIEF** to build the threat picture.")
        return

    score = float(immunity.get("score", 0.0))
    band = str(immunity.get("band", "UNKNOWN"))
    colour = BAND_COLOR.get(band, "#8e8e93")

    st.markdown("#### Adversarial Immunity Score")
    columns = st.columns([1, 2])
    with columns[0]:
        st.markdown(
            f"""<div class="metric-card" style="text-align:center;
                   border-color:{colour};">
                  <div class="metric-value" style="font-size:46px;
                       color:{colour};">{score:.1f}</div>
                  <div class="metric-label">out of 100</div>
                  <div style="margin-top:8px;">{_verdict_chip(band)
                      .replace('#8e8e93', colour)}</div>
                </div>""", unsafe_allow_html=True)
    with columns[1]:
        st.progress(min(max(score / 100.0, 0.0), 1.0))
        st.caption(immunity.get("band_description", ""))
        st.caption(f"Assessment confidence "
                   f"{float(immunity.get('confidence', 0.0)):.2f} · "
                   f"{immunity.get('pillars_assessed', 0)} of "
                   f"{immunity.get('pillars_total', 4)} pillars assessed")

    components = immunity.get("components") or {}
    if components:
        st.dataframe(
            [{
                "Pillar": entry.get("label", key),
                "Score": round(float(entry.get("score", 0.0)), 1),
                "Assessed": "yes" if entry.get("assessed") else "NO",
                "Findings": entry.get("findings", 0),
                "Note": str(entry.get("note", ""))[:70],
            } for key, entry in components.items()],
            use_container_width=True, hide_index=True)

    if story:
        assessment = story.get("campaign_assessment") or {}
        if assessment.get("campaign_detected"):
            st.error(f"COORDINATED CAMPAIGN DETECTED — correlation confidence "
                     f"{float(assessment.get('confidence', 0.0)):.0%}. "
                     f"Patterns: {', '.join(assessment.get('patterns') or [])}")
        stories = story.get("stories") or []
        if stories:
            st.markdown(f"#### Threat narratives ({len(stories)})")
            for entry in stories:
                with st.expander(
                        f"[{entry.get('pattern')}] {entry.get('title')} — "
                        f"confidence {float(entry.get('confidence', 0.0)):.2f}"):
                    st.markdown(f"**Narrative.** {entry.get('narrative', '')}")
                    st.markdown(f"**Implication.** {entry.get('implication', '')}")
                    st.caption("Evidence: "
                               + ", ".join(entry.get("evidence_finding_ids") or []))
        else:
            st.success("No cross-module campaign pattern met the materiality "
                       "floor. Individual findings may still require review.")

    if briefing:
        st.markdown("#### Commander's briefing")
        st.caption(f"Source: {briefing.get('source', 'template')}"
                   + (f" · model {briefing.get('model')}"
                      if briefing.get("model") else ""))
        st.text_area("Briefing", briefing.get("briefing", ""), height=380,
                     key="ts_brief")
        st.warning(briefing.get("disclaimer", "AI-generated. Verify with analyst."))

    if report:
        st.markdown("#### Assurance report")
        columns = st.columns(3)
        with columns[0]:
            st.markdown(_metric("Findings", len(report.get("findings") or [])),
                        unsafe_allow_html=True)
        with columns[1]:
            st.markdown(_metric("Schema valid",
                                "yes" if report.get("schema_valid") else "NO",
                                "#30d158" if report.get("schema_valid")
                                else "#ff3b30"), unsafe_allow_html=True)
        with columns[2]:
            audit = report.get("audit_trail") or {}
            st.markdown(_metric("Audit chain",
                                "valid" if audit.get("chain_valid") else "n/a",
                                "#30d158" if audit.get("chain_valid")
                                else "#8e8e93"), unsafe_allow_html=True)
        if report.get("report_path"):
            st.success(f"Report written to `{report['report_path']}`")
        st.caption(f"report_hash = {str(report.get('report_hash', ''))[:48]}…")
        st.download_button(
            "Download report JSON",
            json.dumps(report, indent=2, default=str),
            file_name=f"{report.get('report_id', 'shield-report')}.json",
            mime="application/json")
        _show_limitations(report.get("limitations") or [], report.get("coverage"))


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Configure the page and render all five tabs."""
    st.set_page_config(page_title="SHIELD-CV", page_icon="🛡",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    _banner()

    with st.sidebar:
        st.markdown("### Mission")
        st.caption("Integrity assurance for multi-contributor CV pipelines. "
                   "Datasets, models and inference records are evaluated "
                   "independently, then correlated.")
        st.divider()
        st.markdown("### Operating constraints")
        st.caption("· 100% offline — the only permitted network call is a local "
                   "Ollama instance\n\n"
                   "· No retraining of contributed models\n\n"
                   "· CPU-only; no GPU required\n\n"
                   "· Every finding carries reason, evidence, confidence and "
                   "disposition")
        st.divider()
        st.markdown("### Session")
        for label, key in (("Data scan", "data_result"),
                           ("Model audit", "model_result"),
                           ("Chain verify", "crypto_result"),
                           ("Drift", "drift_result"),
                           ("Correlation", "immunity")):
            st.caption(f"{'🟢' if _state(key) else '⚪'} {label}")
        if st.button("Clear session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    tabs = st.tabs(["DATA SCANNER", "MODEL AUDIT", "CRYPTO LOCK",
                    "DRIFT MONITOR", "THREAT STORY"])
    renderers = (tab_data_scanner, tab_model_audit, tab_crypto_lock,
                 tab_drift_monitor, tab_threat_story)
    for tab, renderer in zip(tabs, renderers):
        with tab:
            try:
                renderer()
            except Exception as exc:  # pragma: no cover - UI guard
                st.error(f"Tab failed to render: {type(exc).__name__}: {exc}")
                LOGGER.error("tab render failed: %s\n%s", exc, traceback.format_exc())


main()
