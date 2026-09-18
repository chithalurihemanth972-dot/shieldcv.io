"""
Hash-chain construction and integrity verification.

Each provenance record commits to its predecessor via ``prev_record_hash``.
Altering, deleting or reordering any record breaks every link downstream, which
:func:`verify_chain` localises to the exact index where the chain first fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.crypto.hashing import ZERO_HASH, sha256_json
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

GENESIS_HASH = ZERO_HASH

CHAINED_FIELDS = (
    "record_id", "input_hash", "model_hash", "config_hash", "output_hash",
    "timestamp_ns", "sequence_number", "nonce", "prev_record_hash",
)


class ChainBreakType:
    """Enumeration of the ways a hash chain can fail verification."""

    HASH_MISMATCH = "HASH_MISMATCH"
    BROKEN_LINK = "BROKEN_LINK"
    SEQUENCE_GAP = "SEQUENCE_GAP"
    SEQUENCE_REGRESSION = "SEQUENCE_REGRESSION"
    DUPLICATE_NONCE = "DUPLICATE_NONCE"
    DUPLICATE_SEQUENCE = "DUPLICATE_SEQUENCE"
    TIMESTAMP_REGRESSION = "TIMESTAMP_REGRESSION"
    MISSING_GENESIS = "MISSING_GENESIS"
    PAYLOAD_MISMATCH = "PAYLOAD_MISMATCH"


@dataclass
class ChainBreak:
    """A single detected defect in a hash chain.

    Attributes:
        index: Position in the supplied record list.
        record_id: Identifier of the offending record.
        break_type: One of :class:`ChainBreakType`.
        expected: Value the verifier expected.
        found: Value actually present.
        detail: Human-readable explanation.
    """

    index: int
    record_id: str
    break_type: str
    expected: str = ""
    found: str = ""
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the break for evidence payloads.

        Returns:
            JSON-safe dictionary.
        """
        return {
            "index": self.index,
            "record_id": self.record_id,
            "break_type": self.break_type,
            "expected": self.expected,
            "found": self.found,
            "detail": self.detail,
        }


@dataclass
class ChainVerificationResult:
    """Outcome of verifying a whole chain.

    Attributes:
        valid: ``True`` only when no breaks were found.
        num_records: Records examined.
        breaks: All detected defects.
        head_hash: Hash of the final record.
        first_break_index: Index where the chain first failed (``-1`` if clean).
    """

    valid: bool = True
    num_records: int = 0
    breaks: List[ChainBreak] = field(default_factory=list)
    head_hash: str = GENESIS_HASH
    first_break_index: int = -1

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the verification result.

        Returns:
            JSON-safe dictionary suitable for a finding's evidence block.
        """
        return {
            "valid": self.valid,
            "num_records": self.num_records,
            "num_breaks": len(self.breaks),
            "breaks": [b.to_dict() for b in self.breaks],
            "head_hash": self.head_hash,
            "first_break_index": self.first_break_index,
        }


def _as_dict(record: Any) -> Dict[str, Any]:
    """Coerce a record object or dataclass into a plain dictionary.

    Args:
        record: Record instance or mapping.

    Returns:
        Dictionary view of the record.
    """
    try:
        if isinstance(record, dict):
            return record
        if hasattr(record, "to_dict"):
            return record.to_dict()
        return dict(vars(record))
    except Exception:
        return {}


def compute_record_hash(record: Any) -> str:
    """Compute the canonical hash of a record over the chained fields only.

    Args:
        record: Record dictionary or object.

    Returns:
        Hex digest binding identity, inputs, outputs, ordering and the predecessor.
    """
    try:
        payload = _as_dict(record)
        canonical = {}
        for key in CHAINED_FIELDS:
            value = payload.get(key)
            if key in ("timestamp_ns", "sequence_number"):
                canonical[key] = int(value or 0)
            else:
                canonical[key] = str(value or "")
        return sha256_json(canonical)
    except Exception as exc:
        LOGGER.error("compute_record_hash failed: %s", exc)
        return ZERO_HASH


def link_records(records: Sequence[Any]) -> List[Dict[str, Any]]:
    """Chain a list of records by populating ``prev_record_hash``/``record_hash``.

    Args:
        records: Records in chronological order.

    Returns:
        New list of record dictionaries with chain fields filled in.
    """
    linked: List[Dict[str, Any]] = []
    try:
        prev_hash = GENESIS_HASH
        for index, record in enumerate(records):
            payload = dict(_as_dict(record))
            payload["prev_record_hash"] = prev_hash
            payload.setdefault("sequence_number", index)
            payload["record_hash"] = compute_record_hash(payload)
            prev_hash = payload["record_hash"]
            linked.append(payload)
        return linked
    except Exception as exc:
        LOGGER.error("link_records failed: %s", exc)
        return linked


def verify_chain(records: Sequence[Any],
                 check_sequence: bool = True,
                 check_timestamps: bool = True,
                 allow_clock_skew_ns: int = 1_000_000_000) -> ChainVerificationResult:
    """Verify hash links, ordering, nonce uniqueness and timestamps of a chain.

    Args:
        records: Records in the order they claim to have been produced.
        check_sequence: Enforce strictly increasing sequence numbers.
        check_timestamps: Flag timestamps that go backwards beyond the skew budget.
        allow_clock_skew_ns: Tolerated backwards clock skew in nanoseconds.

    Returns:
        :class:`ChainVerificationResult` listing every defect found.
    """
    result = ChainVerificationResult(num_records=len(records))
    try:
        if not records:
            result.valid = True
            return result

        prev_hash = GENESIS_HASH
        prev_sequence: Optional[int] = None
        prev_timestamp: Optional[int] = None
        seen_nonces: Dict[str, int] = {}
        seen_sequences: Dict[int, int] = {}

        for index, raw in enumerate(records):
            payload = _as_dict(raw)
            record_id = str(payload.get("record_id", f"idx-{index}"))

            declared_prev = str(payload.get("prev_record_hash", "") or "")
            if index == 0 and declared_prev not in (GENESIS_HASH, "", prev_hash):
                result.breaks.append(ChainBreak(
                    index, record_id, ChainBreakType.MISSING_GENESIS,
                    expected=GENESIS_HASH, found=declared_prev,
                    detail="First record does not reference the genesis hash; "
                           "earlier records may have been deleted",
                ))
            elif declared_prev != prev_hash:
                result.breaks.append(ChainBreak(
                    index, record_id, ChainBreakType.BROKEN_LINK,
                    expected=prev_hash, found=declared_prev,
                    detail="Record does not link to its predecessor — insertion, "
                           "deletion or reordering detected",
                ))

            recomputed = compute_record_hash(payload)
            declared_hash = str(payload.get("record_hash", "") or "")
            if declared_hash and recomputed != declared_hash:
                result.breaks.append(ChainBreak(
                    index, record_id, ChainBreakType.HASH_MISMATCH,
                    expected=recomputed, found=declared_hash,
                    detail="Stored record hash does not match recomputed hash — "
                           "record content was modified after creation",
                ))

            # The chain binds output_hash, not the output itself. Without this
            # check an attacker can rewrite the recorded result and leave the
            # digest untouched: every link still verifies while the stored
            # answer is a lie. Re-derive the digest from the payload whenever
            # the payload is present.
            stored_output = payload.get("output", None)
            declared_output_hash = str(payload.get("output_hash", "") or "")
            if stored_output is not None and declared_output_hash:
                try:
                    recomputed_output = sha256_json(stored_output)
                    if recomputed_output != declared_output_hash:
                        result.breaks.append(ChainBreak(
                            index, record_id, ChainBreakType.PAYLOAD_MISMATCH,
                            detail=(f"stored output hashes to {recomputed_output[:16]}… "
                                    f"but the record declares "
                                    f"{declared_output_hash[:16]}…; the recorded "
                                    f"result has been altered")))
                except Exception as exc:
                    LOGGER.error("output hash re-derivation failed at %d: %s",
                                 index, exc)

            nonce = str(payload.get("nonce", "") or "")
            if nonce:
                if nonce in seen_nonces:
                    result.breaks.append(ChainBreak(
                        index, record_id, ChainBreakType.DUPLICATE_NONCE,
                        expected="unique nonce", found=nonce,
                        detail=f"Nonce reused from record index {seen_nonces[nonce]} — "
                               "replay attack indicator",
                    ))
                else:
                    seen_nonces[nonce] = index

            sequence = int(payload.get("sequence_number", index) or 0)
            if sequence in seen_sequences:
                result.breaks.append(ChainBreak(
                    index, record_id, ChainBreakType.DUPLICATE_SEQUENCE,
                    expected="unique sequence", found=str(sequence),
                    detail=f"Sequence number duplicates record index {seen_sequences[sequence]}",
                ))
            else:
                seen_sequences[sequence] = index

            if check_sequence and prev_sequence is not None:
                if sequence <= prev_sequence:
                    result.breaks.append(ChainBreak(
                        index, record_id, ChainBreakType.SEQUENCE_REGRESSION,
                        expected=f">{prev_sequence}", found=str(sequence),
                        detail="Sequence number did not increase monotonically",
                    ))
                elif sequence > prev_sequence + 1:
                    result.breaks.append(ChainBreak(
                        index, record_id, ChainBreakType.SEQUENCE_GAP,
                        expected=str(prev_sequence + 1), found=str(sequence),
                        detail=f"{sequence - prev_sequence - 1} record(s) missing from chain",
                    ))

            timestamp = int(payload.get("timestamp_ns", 0) or 0)
            if check_timestamps and prev_timestamp is not None and timestamp > 0:
                if timestamp < prev_timestamp - int(allow_clock_skew_ns):
                    result.breaks.append(ChainBreak(
                        index, record_id, ChainBreakType.TIMESTAMP_REGRESSION,
                        expected=f">={prev_timestamp}", found=str(timestamp),
                        detail="Timestamp moves backwards beyond tolerated clock skew",
                    ))

            prev_hash = declared_hash or recomputed
            prev_sequence = sequence
            if timestamp > 0:
                prev_timestamp = timestamp

        result.head_hash = prev_hash
        result.valid = not result.breaks
        result.first_break_index = result.breaks[0].index if result.breaks else -1
        if not result.valid:
            LOGGER.warning("Chain verification FAILED: %d break(s), first at index %d",
                           len(result.breaks), result.first_break_index)
        else:
            LOGGER.info("Chain verification PASSED for %d record(s)", len(records))
        return result
    except Exception as exc:
        LOGGER.error("verify_chain failed: %s", exc)
        result.valid = False
        result.breaks.append(ChainBreak(-1, "", "VERIFIER_ERROR", detail=str(exc)))
        return result


def find_tampered_records(records: Sequence[Any]) -> List[Dict[str, Any]]:
    """List records whose stored hash disagrees with their recomputed hash.

    Args:
        records: Records to inspect.

    Returns:
        List of ``{"index", "record_id", "stored_hash", "computed_hash"}`` dicts.
    """
    tampered: List[Dict[str, Any]] = []
    try:
        for index, raw in enumerate(records):
            payload = _as_dict(raw)
            stored = str(payload.get("record_hash", "") or "")
            if not stored:
                continue
            computed = compute_record_hash(payload)
            if computed != stored:
                tampered.append({
                    "index": index,
                    "record_id": str(payload.get("record_id", f"idx-{index}")),
                    "stored_hash": stored,
                    "computed_hash": computed,
                })
        return tampered
    except Exception as exc:
        LOGGER.error("find_tampered_records failed: %s", exc)
        return tampered


def chain_head(records: Sequence[Any]) -> str:
    """Return the hash of the last record in a chain.

    Args:
        records: Chain records.

    Returns:
        Head hex digest (genesis hash when empty).
    """
    try:
        if not records:
            return GENESIS_HASH
        payload = _as_dict(records[-1])
        return str(payload.get("record_hash") or compute_record_hash(payload))
    except Exception as exc:
        LOGGER.error("chain_head failed: %s", exc)
        return GENESIS_HASH


__all__ = [
    "verify_chain", "link_records", "compute_record_hash", "find_tampered_records",
    "chain_head", "ChainBreak", "ChainBreakType", "ChainVerificationResult",
    "GENESIS_HASH", "CHAINED_FIELDS",
]
