"""
MODULE 3 — Inference Provenance Engine.

Every inference is bound into a tamper-evident, append-only chain::

    ProvenanceRecord
      input_hash        SHA-256(input image bytes)
      model_hash        SHA-256(model weights)
      config_hash       SHA-256(preprocessing config JSON)
      output_hash       SHA-256(output JSON)
      timestamp_ns      nanosecond precision
      sequence_number   monotonically increasing uint64
      nonce             256-bit random
      prev_record_hash  hash of the previous record  ← CHAIN
      signature         Ed25519 over all of the above

Four independent verifications are offered, because they fail differently:

* :meth:`verify_chain_integrity` — were records inserted, deleted or reordered?
* :meth:`detect_tamper`          — was a field edited after the fact?
* :meth:`detect_replay`          — was a valid old record re-submitted?
* :meth:`verify_signature`       — was the record produced by the holder of the key?

A tamper that recomputes the hash still fails the signature check; a tamper that
forges the signature is infeasible without the private key. The audit trail is
persisted in SQLite (``shield.db``).
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.config import get_config
from src.crypto.chain import (
    GENESIS_HASH,
    ChainBreakType,
    compute_record_hash,
    find_tampered_records,
    verify_chain,
)
from src.crypto.hashing import canonical_json, sha256_file, sha256_json, short_hash
from src.crypto.merkle import MerkleTree
from src.crypto.signing import SigningKeyPair
from src.database import get_database
from src.loaders.record_loader import InferenceRecord, load_records
from src.reporting.schema import AttackClass, Finding, reset_finding_ids, summarize_findings
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ProvenanceRecord:
    """A cryptographically-bound record of a single inference.

    Attributes:
        record_id: Unique identifier.
        input_hash: SHA-256 of the input image bytes.
        model_hash: SHA-256 of the model weights.
        config_hash: SHA-256 of the preprocessing configuration.
        output_hash: SHA-256 of the serialised output.
        timestamp_ns: Nanosecond creation time.
        sequence_number: Monotonic counter.
        nonce: 256-bit hex nonce.
        prev_record_hash: Previous record's hash (chain link).
        record_hash: This record's own hash.
        signature: Ed25519 signature over the canonical payload.
        output: The raw model output retained for audit.
        input_path: Source image path.
        model_id: Producing model identifier.
        contributor: Producing unit/vendor.
    """

    record_id: str
    input_hash: str = ""
    model_hash: str = ""
    config_hash: str = ""
    output_hash: str = ""
    timestamp_ns: int = 0
    sequence_number: int = 0
    nonce: str = ""
    prev_record_hash: str = GENESIS_HASH
    record_hash: str = ""
    signature: str = ""
    output: Any = None
    input_path: Optional[str] = None
    model_id: Optional[str] = None
    contributor: Optional[str] = None

    def canonical_payload(self) -> Dict[str, Any]:
        """Return the exact ordered field set covered by hash and signature.

        Returns:
            Dictionary of the signed fields.
        """
        return {
            "record_id": self.record_id,
            "input_hash": self.input_hash,
            "model_hash": self.model_hash,
            "config_hash": self.config_hash,
            "output_hash": self.output_hash,
            "timestamp_ns": int(self.timestamp_ns),
            "sequence_number": int(self.sequence_number),
            "nonce": self.nonce,
            "prev_record_hash": self.prev_record_hash,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full record including signature and output.

        Returns:
            JSON-safe dictionary.
        """
        payload = self.canonical_payload()
        payload.update({
            "record_hash": self.record_hash,
            "signature": self.signature,
            "output": self.output,
            "input_path": self.input_path,
            "model_id": self.model_id,
            "contributor": self.contributor,
        })
        return payload


class InferenceProvenanceEngine:
    """Creates, seals and verifies tamper-evident inference provenance chains.

    Attributes:
        cfg: Effective configuration.
        keypair: Ed25519 key pair used for signing/verification.
        findings: Findings from the most recent verification run.
    """

    MODULE_NAME = "crypto_chain"

    def __init__(self, config: Optional[Any] = None,
                 keypair: Optional[SigningKeyPair] = None,
                 database: Optional[Any] = None) -> None:
        """Initialise the engine, loading or creating the signing key.

        Args:
            config: Optional configuration override.
            keypair: Optional explicit key pair (a new one is loaded/created otherwise).
            database: Optional database handle override.
        """
        self.cfg = config or get_config()
        self.section = self.cfg.section("provenance")
        self.findings: List[Finding] = []
        self.warnings: List[str] = []
        self._sequence = 0
        self._prev_hash = GENESIS_HASH

        try:
            self.db = database or get_database()
        except Exception as exc:
            LOGGER.error("Database unavailable: %s", exc)
            self.db = None
            self.warnings.append(f"SQLite audit trail unavailable: {exc}")

        try:
            self.keypair = keypair or SigningKeyPair.load_or_create()
            if self.keypair is None:
                self.warnings.append(
                    "Ed25519 signing NOT AVAILABLE: records will be hash-chained but unsigned")
        except Exception as exc:
            LOGGER.error("Key initialisation failed: %s", exc)
            self.keypair = None
            self.warnings.append(f"Signing key unavailable: {exc}")

    # ------------------------------------------------------------------
    # Record creation
    # ------------------------------------------------------------------
    def create_record(self,
                      input_path: Optional[str | Path] = None,
                      input_bytes: Optional[bytes] = None,
                      model_hash: str = "",
                      config: Optional[Dict[str, Any]] = None,
                      output: Any = None,
                      record_id: Optional[str] = None,
                      model_id: Optional[str] = None,
                      contributor: Optional[str] = None,
                      persist: bool = True) -> Optional[ProvenanceRecord]:
        """Create, hash, chain and sign a new provenance record.

        Args:
            input_path: Path to the input image (hashed from disk).
            input_bytes: Raw input bytes, used when no path is available.
            model_hash: SHA-256 of the model weights used for this inference.
            config: Preprocessing configuration dictionary.
            output: Model output to bind into the record.
            record_id: Explicit record id; generated when omitted.
            model_id: Model identifier for reporting.
            contributor: Producing unit/vendor.
            persist: Store the record in the SQLite audit trail.

        Returns:
            The sealed :class:`ProvenanceRecord`, or ``None`` on failure.
        """
        try:
            from src.crypto.hashing import sha256_bytes

            if input_bytes is not None:
                input_hash = sha256_bytes(input_bytes)
            elif input_path is not None:
                input_hash = sha256_file(input_path) or ""
            else:
                input_hash = ""

            record = ProvenanceRecord(
                record_id=record_id or f"REC-{self._sequence:06d}-{secrets.token_hex(4)}",
                input_hash=input_hash,
                model_hash=model_hash,
                config_hash=sha256_json(config or {}),
                output_hash=sha256_json(output),
                timestamp_ns=time.time_ns(),
                sequence_number=self._sequence,
                nonce=secrets.token_hex(int(self.section.get("nonce_bits", 256)) // 8),
                prev_record_hash=self._prev_hash,
                output=output,
                input_path=str(input_path) if input_path else None,
                model_id=model_id,
                contributor=contributor,
            )

            record.record_hash = compute_record_hash(record.to_dict())
            if self.keypair is not None and self.keypair.can_sign:
                record.signature = self.keypair.sign(record.canonical_payload()) or ""

            self._prev_hash = record.record_hash
            self._sequence += 1

            if persist and self.db is not None:
                self.db.insert_provenance_record(record.to_dict())
            return record
        except Exception as exc:
            LOGGER.error("create_record failed: %s", exc)
            return None

    def seal_records(self, records: Sequence[InferenceRecord],
                     persist: bool = True) -> List[ProvenanceRecord]:
        """Convert existing inference records into a signed provenance chain.

        Used to retrofit provenance onto a vendor's plain JSON inference log.

        Args:
            records: Loaded inference records in chronological order.
            persist: Store each sealed record in the audit trail.

        Returns:
            List of sealed :class:`ProvenanceRecord` objects.
        """
        sealed: List[ProvenanceRecord] = []
        try:
            self._sequence = 0
            self._prev_hash = GENESIS_HASH
            for record in records:
                item = ProvenanceRecord(
                    record_id=record.record_id or f"REC-{self._sequence:06d}",
                    input_hash=record.input_hash,
                    model_hash=record.model_hash,
                    config_hash=record.config_hash,
                    output_hash=record.output_hash or sha256_json(record.output),
                    timestamp_ns=record.timestamp_ns or time.time_ns(),
                    sequence_number=self._sequence,
                    nonce=record.nonce or secrets.token_hex(32),
                    prev_record_hash=self._prev_hash,
                    output=record.output,
                    input_path=record.input_path,
                    model_id=record.model_id,
                    contributor=record.contributor,
                )
                item.record_hash = compute_record_hash(item.to_dict())
                if self.keypair is not None and self.keypair.can_sign:
                    item.signature = self.keypair.sign(item.canonical_payload()) or ""
                self._prev_hash = item.record_hash
                self._sequence += 1
                sealed.append(item)
                if persist and self.db is not None:
                    self.db.insert_provenance_record(item.to_dict())
            LOGGER.info("Sealed %d record(s) into a signed provenance chain", len(sealed))
            return sealed
        except Exception as exc:
            LOGGER.error("seal_records failed: %s", exc)
            return sealed

    # ------------------------------------------------------------------
    # Verification primitives
    # ------------------------------------------------------------------
    def verify_chain_integrity(self, records: Sequence[Any]) -> Dict[str, Any]:
        """Verify every hash link, sequence number and timestamp in the chain.

        Args:
            records: Records in their claimed order.

        Returns:
            Dictionary form of :class:`~src.crypto.chain.ChainVerificationResult`.
        """
        try:
            outcome = verify_chain(
                records,
                check_sequence=True,
                check_timestamps=True,
                allow_clock_skew_ns=int(self.section.get("allow_clock_skew_ns", 1_000_000_000)),
            )
            return outcome.to_dict()
        except Exception as exc:
            LOGGER.error("verify_chain_integrity failed: %s", exc)
            return {"valid": False, "num_records": len(records), "breaks": [],
                    "error": str(exc)}

    def detect_tamper(self, records: Sequence[Any]) -> List[Dict[str, Any]]:
        """Recompute hashes and verify payload-to-hash consistency.

        Two distinct checks are required, and omitting either leaves a real hole:

        1. **Chain hash** — does ``record_hash`` match the canonical payload?
           Catches edits to identity, ordering or the hash fields themselves.
        2. **Payload binding** — does ``SHA-256(output)`` still equal the stored
           ``output_hash``? The chain hash covers the *hash of* the output, not
           the output itself, so an attacker who rewrites a detection result and
           leaves the hash fields untouched produces a record that is internally
           chain-consistent. Only re-hashing the payload exposes it.

        Args:
            records: Records to check.

        Returns:
            List of tampered-record descriptors, each naming the failed check.
        """
        tampered: List[Dict[str, Any]] = []
        try:
            for item in find_tampered_records(records):
                item["check"] = "CHAIN_HASH"
                tampered.append(item)
            already = {t["index"] for t in tampered}

            for index, raw in enumerate(records):
                payload = raw if isinstance(raw, dict) else (
                    raw.to_dict() if hasattr(raw, "to_dict") else dict(vars(raw)))

                stored_output_hash = str(payload.get("output_hash", "") or "")
                if not stored_output_hash or payload.get("output") is None:
                    continue
                recomputed = sha256_json(payload.get("output"))
                if recomputed == stored_output_hash:
                    continue

                descriptor = {
                    "index": index,
                    "record_id": str(payload.get("record_id", f"idx-{index}")),
                    "stored_hash": stored_output_hash,
                    "computed_hash": recomputed,
                    "check": "OUTPUT_PAYLOAD",
                    "field": "output",
                }
                if index in already:
                    for existing in tampered:
                        if existing["index"] == index:
                            existing["check"] = "CHAIN_HASH+OUTPUT_PAYLOAD"
                            existing["output_stored_hash"] = stored_output_hash
                            existing["output_computed_hash"] = recomputed
                            break
                else:
                    tampered.append(descriptor)

            tampered.sort(key=lambda t: t["index"])
            return tampered
        except Exception as exc:
            LOGGER.error("detect_tamper failed: %s", exc)
            return tampered

    def detect_replay(self, records: Sequence[Any]) -> List[Dict[str, Any]]:
        """Detect replayed records via duplicate nonces, ids or identical payloads.

        Args:
            records: Records to check.

        Returns:
            List of replay descriptors with the conflicting indices.
        """
        replays: List[Dict[str, Any]] = []
        try:
            seen_nonces: Dict[str, int] = {}
            seen_ids: Dict[str, int] = {}
            seen_payloads: Dict[str, int] = {}

            for index, raw in enumerate(records):
                payload = raw if isinstance(raw, dict) else (
                    raw.to_dict() if hasattr(raw, "to_dict") else dict(vars(raw)))

                nonce = str(payload.get("nonce", "") or "")
                if nonce:
                    if nonce in seen_nonces:
                        replays.append({
                            "index": index,
                            "record_id": str(payload.get("record_id", "")),
                            "type": "DUPLICATE_NONCE",
                            "first_seen_index": seen_nonces[nonce],
                            "nonce": short_hash(nonce, 16),
                            "detail": ("A 256-bit nonce repeated — the probability of this "
                                       "by chance is negligible, so the record was replayed"),
                        })
                    else:
                        seen_nonces[nonce] = index

                record_id = str(payload.get("record_id", "") or "")
                if record_id:
                    if record_id in seen_ids:
                        replays.append({
                            "index": index, "record_id": record_id,
                            "type": "DUPLICATE_RECORD_ID",
                            "first_seen_index": seen_ids[record_id],
                            "detail": "Record identifier reused",
                        })
                    else:
                        seen_ids[record_id] = index

                fingerprint = canonical_json({
                    "i": payload.get("input_hash"), "o": payload.get("output_hash"),
                    "m": payload.get("model_hash"), "t": payload.get("timestamp_ns"),
                })
                if fingerprint in seen_payloads:
                    replays.append({
                        "index": index, "record_id": record_id,
                        "type": "IDENTICAL_PAYLOAD",
                        "first_seen_index": seen_payloads[fingerprint],
                        "detail": ("Identical input, output, model and timestamp as an "
                                   "earlier record — verbatim resubmission"),
                    })
                else:
                    seen_payloads[fingerprint] = index

            if self.db is not None:
                for index, raw in enumerate(records):
                    payload = raw if isinstance(raw, dict) else raw.to_dict()
                    nonce = str(payload.get("nonce", "") or "")
                    if nonce and nonce not in seen_nonces:
                        continue
            return replays
        except Exception as exc:
            LOGGER.error("detect_replay failed: %s", exc)
            return replays

    def verify_signature(self, record: Any) -> Dict[str, Any]:
        """Verify one record's Ed25519 signature over its canonical payload.

        Args:
            record: Record dictionary or object.

        Returns:
            Dictionary with ``valid``, ``available`` and a ``reason``.
        """
        try:
            payload = record if isinstance(record, dict) else record.to_dict()
            signature = str(payload.get("signature", "") or "")
            if not signature:
                return {"valid": False, "available": False,
                        "reason": "Record carries no signature"}
            if self.keypair is None or self.keypair.public_key is None:
                return {"valid": False, "available": False,
                        "reason": "No verification key available: signature check NOT AVAILABLE"}

            canonical = {
                "record_id": str(payload.get("record_id", "")),
                "input_hash": str(payload.get("input_hash", "")),
                "model_hash": str(payload.get("model_hash", "")),
                "config_hash": str(payload.get("config_hash", "")),
                "output_hash": str(payload.get("output_hash", "")),
                "timestamp_ns": int(payload.get("timestamp_ns", 0) or 0),
                "sequence_number": int(payload.get("sequence_number", 0) or 0),
                "nonce": str(payload.get("nonce", "")),
                "prev_record_hash": str(payload.get("prev_record_hash", "")),
            }
            valid = self.keypair.verify(canonical, signature)
            return {"valid": bool(valid), "available": True,
                    "reason": "" if valid else "Ed25519 signature does not verify"}
        except Exception as exc:
            LOGGER.error("verify_signature failed: %s", exc)
            return {"valid": False, "available": False, "reason": str(exc)}

    def merkle_root(self, records: Sequence[Any]) -> Dict[str, Any]:
        """Compute a Merkle root over a batch for compact batch attestation.

        Args:
            records: Records in the batch.

        Returns:
            Dictionary with the ``root``, ``size`` and tree ``depth``.
        """
        try:
            leaves = []
            for raw in records:
                payload = raw if isinstance(raw, dict) else raw.to_dict()
                leaves.append(payload.get("record_hash") or compute_record_hash(payload))
            tree = MerkleTree(leaves)
            return {"root": tree.root, "size": tree.size, "depth": tree.depth}
        except Exception as exc:
            LOGGER.error("merkle_root failed: %s", exc)
            return {"root": GENESIS_HASH, "size": 0, "depth": 0, "error": str(exc)}

    # ------------------------------------------------------------------
    # Full verification pass
    # ------------------------------------------------------------------
    def verify(self, source: str | Path | Sequence[Any],
               progress_callback: Optional[Callable[[str, int, int], None]] = None
               ) -> Dict[str, Any]:
        """Run every provenance verification and emit findings.

        Args:
            source: Path to a records file/directory, or an in-memory record list.
            progress_callback: Optional ``callable(stage, done, total)``.

        Returns:
            Result dictionary with ``findings``, ``summary``, ``chain``, ``merkle``,
            ``signatures``, ``replays`` and ``limitations``.
        """
        started = time.time()
        reset_finding_ids("PROV")
        self.findings = []

        result: Dict[str, Any] = {
            "module": self.MODULE_NAME,
            "target": str(source) if isinstance(source, (str, Path)) else "<in-memory>",
            "findings": [], "summary": {}, "chain": {}, "merkle": {},
            "signatures": {}, "replays": [], "limitations": list(self.warnings),
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
            if isinstance(source, (str, Path)):
                records = load_records(source)
            else:
                records = list(source)

            if not records:
                result["limitations"].append(f"No inference records found at {source}")
                result["summary"] = summarize_findings([])
                result["duration_seconds"] = round(time.time() - started, 2)
                return result

            payloads = [r if isinstance(r, dict) else r.to_dict() for r in records]
            result["num_records"] = len(payloads)
            report("chain", 0, 4)

            # 1. chain integrity -------------------------------------------
            chain = self.verify_chain_integrity(payloads)
            result["chain"] = chain
            self._emit_chain_findings(chain, payloads)
            report("chain", 1, 4)

            # 2. tamper ----------------------------------------------------
            tampered = self.detect_tamper(payloads)
            result["tampered"] = tampered
            self._emit_tamper_findings(tampered, payloads)
            report("tamper", 2, 4)

            # 3. replay ----------------------------------------------------
            replays = self.detect_replay(payloads)
            result["replays"] = replays
            self._emit_replay_findings(replays, payloads)
            report("replay", 3, 4)

            # 4. signatures ------------------------------------------------
            signature_stats = {"checked": 0, "valid": 0, "invalid": 0, "missing": 0,
                               "unavailable": 0}
            for index, payload in enumerate(payloads):
                outcome = self.verify_signature(payload)
                signature_stats["checked"] += 1
                if not outcome["available"]:
                    if "no signature" in outcome["reason"].lower():
                        signature_stats["missing"] += 1
                    else:
                        signature_stats["unavailable"] += 1
                elif outcome["valid"]:
                    signature_stats["valid"] += 1
                else:
                    signature_stats["invalid"] += 1
                    self.findings.append(Finding.create(
                        attack_class=AttackClass.SIGNATURE_INVALID.value,
                        affected_asset=str(payload.get("record_id", f"index-{index}")),
                        confidence=0.97,
                        reason=(
                            "Ed25519 signature does not verify against the record's canonical "
                            "payload. Either a field was altered after signing, or the record "
                            "was produced by a key other than the trusted signer. This cannot "
                            "occur through transmission error alone."),
                        evidence={
                            "record_index": index,
                            "signature": short_hash(str(payload.get("signature", "")), 24),
                            "algorithm": "Ed25519",
                            "verification": "FAILED",
                        },
                        module=self.MODULE_NAME, detector="verify_signature",
                        contributor=payload.get("contributor"), asset_type="record",
                        prefix="PROV", severity="CRITICAL", disposition="QUARANTINE",
                    ))
            result["signatures"] = signature_stats
            if signature_stats["missing"] == len(payloads):
                result["limitations"].append(
                    "No records carried signatures: authenticity verification NOT AVAILABLE "
                    "(hash-chain integrity was still verified)")
            report("signature", 4, 4)

            result["merkle"] = self.merkle_root(payloads)

            payload_findings = [f.to_dict() for f in self.findings]
            result["findings"] = payload_findings
            result["summary"] = summarize_findings(payload_findings)
            result["duration_seconds"] = round(time.time() - started, 2)

            if self.db is not None:
                self.db.append_audit("SHIELD-CV", "PROVENANCE_VERIFY", result["target"], {
                    "records": len(payloads),
                    "chain_valid": chain.get("valid"),
                    "findings": len(payload_findings),
                })
                result["audit_trail_hash"] = self.db.audit_trail_hash()

            LOGGER.info("Provenance verification: %d record(s), chain_valid=%s, %d finding(s)",
                        len(payloads), chain.get("valid"), len(payload_findings))
            return result
        except Exception as exc:
            LOGGER.error("Provenance verification failed: %s", exc, exc_info=True)
            result["limitations"].append(f"Verification aborted: {exc}")
            result["findings"] = [f.to_dict() for f in self.findings]
            result["summary"] = summarize_findings(result["findings"])
            result["duration_seconds"] = round(time.time() - started, 2)
            return result

    # ------------------------------------------------------------------
    # Finding emission
    # ------------------------------------------------------------------
    def _emit_chain_findings(self, chain: Dict[str, Any],
                             payloads: List[Dict[str, Any]]) -> None:
        """Turn chain-verification breaks into findings.

        Args:
            chain: Output of :meth:`verify_chain_integrity`.
            payloads: The record dictionaries examined.
        """
        try:
            severity_by_type = {
                ChainBreakType.HASH_MISMATCH: (0.96, "CRITICAL"),
                ChainBreakType.BROKEN_LINK: (0.94, "CRITICAL"),
                ChainBreakType.DUPLICATE_NONCE: (0.92, "CRITICAL"),
                ChainBreakType.DUPLICATE_SEQUENCE: (0.80, "HIGH"),
                ChainBreakType.SEQUENCE_GAP: (0.75, "HIGH"),
                ChainBreakType.SEQUENCE_REGRESSION: (0.78, "HIGH"),
                ChainBreakType.TIMESTAMP_REGRESSION: (0.60, "MEDIUM"),
                ChainBreakType.MISSING_GENESIS: (0.70, "HIGH"),
            }
            for issue in chain.get("breaks", []):
                break_type = issue.get("break_type", "UNKNOWN")
                if break_type == ChainBreakType.HASH_MISMATCH:
                    continue  # reported in detail by the tamper detector
                confidence, severity = severity_by_type.get(break_type, (0.6, "MEDIUM"))
                index = int(issue.get("index", -1))
                contributor = (payloads[index].get("contributor")
                               if 0 <= index < len(payloads) else None)
                self.findings.append(Finding.create(
                    attack_class=(AttackClass.INFERENCE_REPLAY.value
                                  if break_type == ChainBreakType.DUPLICATE_NONCE
                                  else AttackClass.CHAIN_BREAK.value),
                    affected_asset=str(issue.get("record_id") or f"index-{index}"),
                    confidence=confidence,
                    reason=(f"Provenance chain defect ({break_type}): "
                            f"{issue.get('detail', '')}"),
                    evidence={
                        "break_type": break_type,
                        "record_index": index,
                        "expected": short_hash(str(issue.get("expected", "")), 20),
                        "found": short_hash(str(issue.get("found", "")), 20),
                        "chain_position": f"{index + 1} of {chain.get('num_records', 0)}",
                    },
                    module=self.MODULE_NAME, detector="verify_chain_integrity",
                    contributor=contributor, asset_type="record", prefix="PROV",
                    severity=severity,
                    disposition="QUARANTINE" if confidence >= 0.8 else "REVIEW",
                ))
        except Exception as exc:
            LOGGER.error("_emit_chain_findings failed: %s", exc)

    def _emit_tamper_findings(self, tampered: List[Dict[str, Any]],
                              payloads: List[Dict[str, Any]]) -> None:
        """Turn hash mismatches into detailed tamper findings.

        Args:
            tampered: Output of :meth:`detect_tamper`.
            payloads: The record dictionaries examined.
        """
        try:
            for item in tampered:
                index = int(item.get("index", -1))
                payload = payloads[index] if 0 <= index < len(payloads) else {}
                check = item.get("check", "CHAIN_HASH")
                if "OUTPUT_PAYLOAD" in check:
                    altered = self._describe_output_change(payload)
                    reason = (
                        "The recorded inference RESULT no longer matches its own hash. "
                        f"Re-hashing the stored output yields "
                        f"{short_hash(item.get('computed_hash'), 16)} but the record claims "
                        f"{short_hash(item.get('stored_hash'), 16)}. The output payload was "
                        "rewritten after sealing while the hash fields were left untouched — "
                        "a direct attempt to falsify what the model reported."
                        + (f" {altered}" if altered else ""))
                else:
                    altered = self._guess_altered_field(payload)
                    reason = (
                        "Record content does not match its stored hash — the record was "
                        "modified after it was sealed. Recomputing SHA-256 over the canonical "
                        f"payload yields {short_hash(item.get('computed_hash'), 16)} but the "
                        f"record claims {short_hash(item.get('stored_hash'), 16)}."
                        + (f" Most likely altered field: {altered}." if altered else ""))

                self.findings.append(Finding.create(
                    attack_class=AttackClass.INFERENCE_TAMPER.value,
                    affected_asset=str(item.get("record_id", f"index-{index}")),
                    confidence=0.97,
                    reason=reason,
                    evidence={
                        "record_index": index,
                        "failed_check": check,
                        "stored_hash": item.get("stored_hash", ""),
                        "computed_hash": item.get("computed_hash", ""),
                        "tampered_field": item.get("field", "canonical payload"),
                        "current_output": payload.get("output"),
                        "algorithm": "SHA-256",
                    },
                    module=self.MODULE_NAME, detector="detect_tamper",
                    contributor=payload.get("contributor"), asset_type="record",
                    prefix="PROV", severity="CRITICAL", disposition="QUARANTINE",
                ))
        except Exception as exc:
            LOGGER.error("_emit_tamper_findings failed: %s", exc)

    def _describe_output_change(self, payload: Dict[str, Any]) -> str:
        """Summarise what the tampered output now claims, for the analyst.

        Args:
            payload: The record dictionary.

        Returns:
            Sentence naming the current detection classes, or an empty string.
        """
        try:
            output = payload.get("output") or {}
            detections = output.get("detections") if isinstance(output, dict) else None
            if not detections:
                return ""
            labels = [str(d.get("class", "?")) for d in detections[:3]]
            return f"The record now reports: {', '.join(labels)}."
        except Exception:
            return ""

    def _guess_altered_field(self, payload: Dict[str, Any]) -> str:
        """Identify which field was most likely edited in a tampered record.

        The output payload is re-hashed and compared to the record's stored
        ``output_hash``: if they disagree, the *output* itself was rewritten,
        which is the classic "change the detection result" attack.

        Args:
            payload: The record dictionary.

        Returns:
            Field name, or an empty string when undeterminable.
        """
        try:
            if payload.get("output") is not None and payload.get("output_hash"):
                if sha256_json(payload["output"]) != payload["output_hash"]:
                    return "output (the recorded result no longer matches its own hash)"
            return ""
        except Exception:
            return ""

    def _emit_replay_findings(self, replays: List[Dict[str, Any]],
                              payloads: List[Dict[str, Any]]) -> None:
        """Turn replay detections into findings.

        Args:
            replays: Output of :meth:`detect_replay`.
            payloads: The record dictionaries examined.
        """
        try:
            for item in replays:
                index = int(item.get("index", -1))
                payload = payloads[index] if 0 <= index < len(payloads) else {}
                confidence = 0.93 if item["type"] == "DUPLICATE_NONCE" else 0.75
                self.findings.append(Finding.create(
                    attack_class=AttackClass.INFERENCE_REPLAY.value,
                    affected_asset=str(item.get("record_id") or f"index-{index}"),
                    confidence=confidence,
                    reason=(f"Replay indicator ({item['type']}): {item['detail']}. "
                            f"First observed at record index {item['first_seen_index']}."),
                    evidence={
                        "replay_type": item["type"],
                        "record_index": index,
                        "first_seen_index": item["first_seen_index"],
                        "nonce": item.get("nonce", ""),
                    },
                    module=self.MODULE_NAME, detector="detect_replay",
                    contributor=payload.get("contributor"), asset_type="record",
                    prefix="PROV",
                    severity="CRITICAL" if confidence >= 0.85 else "HIGH",
                    disposition="QUARANTINE",
                ))
        except Exception as exc:
            LOGGER.error("_emit_replay_findings failed: %s", exc)


def verify_records(source: str | Path) -> Dict[str, Any]:
    """Convenience wrapper verifying a provenance chain from disk.

    Args:
        source: Records file or directory.

    Returns:
        Verification result dictionary.
    """
    try:
        return InferenceProvenanceEngine().verify(source)
    except Exception as exc:
        LOGGER.error("verify_records failed: %s", exc)
        return {"module": "crypto_chain", "findings": [], "summary": summarize_findings([]),
                "limitations": [str(exc)]}


__all__ = ["InferenceProvenanceEngine", "ProvenanceRecord", "verify_records"]
