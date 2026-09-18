"""
Inference-record loader for SHIELD-CV provenance auditing.

Accepts a single JSON array, a JSON-Lines file, or a directory of per-record
JSON files, and normalises each entry into :class:`InferenceRecord`. Field names
vary wildly between vendors, so a synonym table maps them onto the canonical
provenance schema.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

# Maximum file size (bytes) for JSON record loading — prevents memory exhaustion
# on multi-GB files.  ~50 MB covers any realistic inference log.
_MAX_RECORD_FILE_SIZE: int = 50 * 1024 * 1024

# Maximum size (bytes) for the serialised ``output`` field per record.
_MAX_OUTPUT_SIZE: int = 1 * 1024 * 1024  # 1 MB

_FIELD_SYNONYMS: Dict[str, tuple] = {
    "record_id": ("record_id", "id", "uuid", "inference_id", "rid"),
    "input_hash": ("input_hash", "image_hash", "input_sha256", "input_digest"),
    "model_hash": ("model_hash", "weights_hash", "model_sha256", "model_digest"),
    "config_hash": ("config_hash", "preprocessing_hash", "preproc_hash", "cfg_hash"),
    "output_hash": ("output_hash", "result_hash", "prediction_hash", "output_digest"),
    "timestamp_ns": ("timestamp_ns", "timestamp", "ts", "time_ns", "created_at"),
    "sequence_number": ("sequence_number", "seq", "sequence", "index", "seq_no"),
    "nonce": ("nonce", "salt", "random"),
    "prev_record_hash": ("prev_record_hash", "previous_hash", "prev_hash", "parent_hash"),
    "signature": ("signature", "sig", "ed25519_signature"),
    "record_hash": ("record_hash", "self_hash", "digest", "hash"),
    "output": ("output", "prediction", "predictions", "result", "detections"),
    "input_path": ("input_path", "image_path", "image", "input", "file"),
    "model_id": ("model_id", "model", "model_name"),
    "contributor": ("contributor", "source", "vendor", "operator", "unit"),
}


def _pick(payload: Dict[str, Any], canonical: str) -> Any:
    """Retrieve a canonical field from a record using its known synonyms.

    Args:
        payload: Raw record dictionary.
        canonical: Canonical field name from :data:`_FIELD_SYNONYMS`.

    Returns:
        The first matching value, or ``None``.
    """
    try:
        for key in _FIELD_SYNONYMS.get(canonical, (canonical,)):
            if key in payload and payload[key] is not None:
                return payload[key]
        nested = payload.get("provenance") or payload.get("metadata")
        if isinstance(nested, dict):
            for key in _FIELD_SYNONYMS.get(canonical, (canonical,)):
                if key in nested and nested[key] is not None:
                    return nested[key]
        return None
    except Exception:
        return None


def _to_int(value: Any, default: int = 0) -> int:
    """Coerce a value to int, tolerating strings, floats and ISO timestamps.

    Args:
        value: Raw value.
        default: Returned when coercion fails.

    Returns:
        Integer value.
    """
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip()
        if text.isdigit():
            return int(text)
        from datetime import datetime
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1e9)
    except Exception:
        return default


@dataclass
class InferenceRecord:
    """One normalised inference-provenance record.

    Attributes:
        record_id: Unique record identifier.
        input_hash: SHA-256 of the input image bytes.
        model_hash: SHA-256 of the model weights used.
        config_hash: SHA-256 of the preprocessing configuration JSON.
        output_hash: SHA-256 of the serialised output.
        timestamp_ns: Nanosecond-precision creation time.
        sequence_number: Monotonically increasing counter.
        nonce: 256-bit hex nonce guarding against replay.
        prev_record_hash: Hash of the preceding record (chain link).
        signature: Ed25519 signature over the canonical payload.
        record_hash: Self hash as stored by the producer.
        output: The original model output payload.
        input_path: Path of the input image, when recorded.
        model_id: Identifier of the producing model.
        contributor: Producing contributor/unit.
        raw: The untouched source dictionary.
        source_file: File the record was read from.
    """

    record_id: str = ""
    input_hash: str = ""
    model_hash: str = ""
    config_hash: str = ""
    output_hash: str = ""
    timestamp_ns: int = 0
    sequence_number: int = 0
    nonce: str = ""
    prev_record_hash: str = ""
    signature: str = ""
    record_hash: str = ""
    output: Any = None
    input_path: Optional[str] = None
    model_id: Optional[str] = None
    contributor: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    source_file: Optional[str] = None

    def canonical_payload(self) -> Dict[str, Any]:
        """Return the exact field set covered by the hash and signature.

        Field order is fixed so signing and verification always agree.

        Returns:
            Ordered dictionary of signed fields.
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
        """Serialise the full record (payload + signature + output) to a dict.

        Returns:
            JSON-safe dictionary.
        """
        payload = self.canonical_payload()
        payload.update({
            "signature": self.signature,
            "record_hash": self.record_hash,
            "output": self.output,
            "input_path": self.input_path,
            "model_id": self.model_id,
            "contributor": self.contributor,
        })
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any],
                  source_file: Optional[str] = None) -> "InferenceRecord":
        """Build a record from an arbitrary vendor dictionary.

        Args:
            payload: Raw record mapping.
            source_file: Originating file path for traceability.

        Returns:
            Normalised :class:`InferenceRecord`.
        """
        try:
            output = _pick(payload, "output")
            # Enforce output size limit — oversized payloads exhaust memory and
            # bloat the SQLite database.
            if output is not None:
                try:
                    serialised = json.dumps(output, default=str)
                    if len(serialised.encode("utf-8")) > _MAX_OUTPUT_SIZE:
                        LOGGER.warning("Truncating oversized output field in record from %s "
                                       "(%d bytes)", source_file, len(serialised.encode("utf-8")))
                        output = {"_truncated": True,
                                  "_original_size": len(serialised.encode("utf-8")),
                                  "_note": "Output exceeded 1 MB limit and was truncated"}
                except Exception:
                    pass

            return cls(
                record_id=str(_pick(payload, "record_id") or ""),
                input_hash=str(_pick(payload, "input_hash") or ""),
                model_hash=str(_pick(payload, "model_hash") or ""),
                config_hash=str(_pick(payload, "config_hash") or ""),
                output_hash=str(_pick(payload, "output_hash") or ""),
                timestamp_ns=_to_int(_pick(payload, "timestamp_ns")),
                sequence_number=_to_int(_pick(payload, "sequence_number")),
                nonce=str(_pick(payload, "nonce") or ""),
                prev_record_hash=str(_pick(payload, "prev_record_hash") or ""),
                signature=str(_pick(payload, "signature") or ""),
                record_hash=str(_pick(payload, "record_hash") or ""),
                output=output,
                input_path=(str(_pick(payload, "input_path"))
                            if _pick(payload, "input_path") else None),
                model_id=(str(_pick(payload, "model_id"))
                          if _pick(payload, "model_id") else None),
                contributor=(str(_pick(payload, "contributor"))
                             if _pick(payload, "contributor") else None),
                raw=dict(payload),
                source_file=source_file,
            )
        except Exception as exc:
            LOGGER.warning("Malformed inference record skipped: %s", exc)
            return cls(raw=dict(payload) if isinstance(payload, dict) else {},
                       source_file=source_file)


def _iter_json_objects(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield record dictionaries from a JSON array, JSONL, or wrapper object.

    Args:
        path: File to read.

    Yields:
        Record dictionaries.
    """
    try:
        file_size = path.stat().st_size
        if file_size > _MAX_RECORD_FILE_SIZE:
            LOGGER.error("Record file %s exceeds %d byte limit (%d bytes) — skipping",
                         path, _MAX_RECORD_FILE_SIZE, file_size)
            return
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return
        if text.startswith("["):
            payload = json.loads(text)
            for item in payload:
                if isinstance(item, dict):
                    yield item
            return
        if text.startswith("{"):
            try:
                payload = json.loads(text)
                if isinstance(payload, dict):
                    for key in ("records", "inferences", "entries", "data", "chain"):
                        if isinstance(payload.get(key), list):
                            for item in payload[key]:
                                if isinstance(item, dict):
                                    yield item
                            return
                    yield payload
                    return
            except json.JSONDecodeError:
                pass
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip().rstrip(",")
            if not stripped or stripped in ("[", "]"):
                continue
            try:
                item = json.loads(stripped)
                if isinstance(item, dict):
                    yield item
            except json.JSONDecodeError:
                LOGGER.debug("Skipping unparsable line %d in %s", line_no, path.name)
    except Exception as exc:
        LOGGER.error("Could not read records from %s: %s", path, exc)


def load_records(source: str | Path) -> List[InferenceRecord]:
    """Load inference records from a file or a directory of files.

    Args:
        source: JSON/JSONL file, or a directory containing them.

    Returns:
        Records ordered by ``sequence_number`` then ``timestamp_ns``; empty on failure.
    """
    records: List[InferenceRecord] = []
    try:
        path = Path(source)
        files: List[Path]
        if path.is_dir():
            files = sorted(
                p for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in (".json", ".jsonl", ".ndjson")
            )
        elif path.is_file():
            files = [path]
        else:
            LOGGER.error("Record source not found: %s", path)
            return []

        for file_path in files:
            for payload in _iter_json_objects(file_path):
                records.append(InferenceRecord.from_dict(payload, source_file=str(file_path)))

        records.sort(key=lambda r: (r.sequence_number, r.timestamp_ns))
        LOGGER.info("Loaded %d inference record(s) from %s", len(records), path)
        return records
    except Exception as exc:
        LOGGER.error("load_records failed: %s", exc)
        return records


def save_records(records: List[InferenceRecord], path: str | Path,
                 jsonl: bool = False) -> bool:
    """Write records back to disk as JSON or JSON-Lines.

    Args:
        records: Records to serialise.
        path: Destination file path.
        jsonl: Emit one JSON object per line instead of a JSON array.

    Returns:
        ``True`` on success.
    """
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = [r.to_dict() for r in records]
        with target.open("w", encoding="utf-8") as handle:
            if jsonl:
                for item in payload:
                    handle.write(json.dumps(item) + "\n")
            else:
                json.dump(payload, handle, indent=2)
        LOGGER.info("Wrote %d record(s) to %s", len(records), target)
        return True
    except Exception as exc:
        LOGGER.error("save_records failed: %s", exc)
        return False


__all__ = ["InferenceRecord", "load_records", "save_records"]
