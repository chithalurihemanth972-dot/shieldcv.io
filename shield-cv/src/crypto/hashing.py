"""
SHA-256 hashing primitives for SHIELD-CV.

All provenance hashing flows through here so the digest of a given object is
byte-identical no matter which module computes it. JSON is always canonicalised
(sorted keys, no insignificant whitespace) before hashing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

DEFAULT_ALGORITHM = "sha256"
CHUNK_SIZE = 1 << 20  # 1 MiB
ZERO_HASH = "0" * 64


def sha256_bytes(data: bytes) -> str:
    """Hash a byte string with SHA-256.

    Args:
        data: Raw bytes.

    Returns:
        Lowercase hex digest.
    """
    try:
        return hashlib.sha256(data).hexdigest()
    except Exception as exc:
        LOGGER.error("sha256_bytes failed: %s", exc)
        return ZERO_HASH


def sha256_text(text: str, encoding: str = "utf-8") -> str:
    """Hash a text string with SHA-256.

    Args:
        text: Input string.
        encoding: Character encoding used before hashing.

    Returns:
        Lowercase hex digest.
    """
    try:
        return sha256_bytes(text.encode(encoding))
    except Exception as exc:
        LOGGER.error("sha256_text failed: %s", exc)
        return ZERO_HASH


def sha256_file(path: Union[str, Path], chunk_size: int = CHUNK_SIZE) -> Optional[str]:
    """Stream-hash a file with SHA-256 (constant memory).

    Args:
        path: File path.
        chunk_size: Read block size in bytes.

    Returns:
        Hex digest, or ``None`` when the file cannot be read.
    """
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(chunk_size), b""):
                digest.update(block)
        return digest.hexdigest()
    except FileNotFoundError:
        LOGGER.error("sha256_file: file not found: %s", path)
        return None
    except Exception as exc:
        LOGGER.error("sha256_file failed for %s: %s", path, exc)
        return None


def canonical_json(obj: Any) -> str:
    """Serialise any object to canonical JSON suitable for hashing/signing.

    Keys are sorted, separators are tight, non-ASCII is preserved, and numpy
    scalars/arrays are converted to plain Python types.

    Args:
        obj: Object to serialise.

    Returns:
        Canonical JSON string (``"null"`` when serialisation fails).
    """
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, default=_json_default)
    except Exception as exc:
        LOGGER.warning("canonical_json: primary serialisation failed (%s), "
                       "falling back to str(obj) — hash may be non-deterministic "
                       "for sets or dicts with non-sortable keys", exc)
        try:
            return json.dumps(str(obj))
        except Exception:
            return "null"


def _json_default(value: Any) -> Any:
    """Fallback JSON encoder for numpy, sets, bytes and paths.

    Args:
        value: Object the standard encoder rejected.

    Returns:
        A JSON-serialisable representation.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(str(v) for v in value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def sha256_json(obj: Any) -> str:
    """Hash an object via its canonical JSON encoding.

    Args:
        obj: Object to hash (dict, list, scalar...).

    Returns:
        Lowercase hex digest.
    """
    return sha256_text(canonical_json(obj))


def sha256_array(array: np.ndarray) -> str:
    """Hash a numpy array including dtype and shape.

    Args:
        array: Input array.

    Returns:
        Lowercase hex digest.
    """
    try:
        contiguous = np.ascontiguousarray(array)
        digest = hashlib.sha256()
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.tobytes())
        return digest.hexdigest()
    except Exception as exc:
        LOGGER.error("sha256_array failed: %s", exc)
        return ZERO_HASH


def sha256_state_dict(state_dict: Dict[str, np.ndarray]) -> str:
    """Hash a model state dict deterministically (sorted by layer name).

    Args:
        state_dict: Mapping of layer name to numpy array.

    Returns:
        Lowercase hex digest (:data:`ZERO_HASH` when empty).
    """
    try:
        if not state_dict:
            return ZERO_HASH
        digest = hashlib.sha256()
        for name in sorted(state_dict.keys()):
            array = np.ascontiguousarray(state_dict[name])
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(str(array.shape).encode("utf-8"))
            digest.update(array.tobytes())
        return digest.hexdigest()
    except Exception as exc:
        LOGGER.error("sha256_state_dict failed: %s", exc)
        return ZERO_HASH


def hash_concat(*hashes: str) -> str:
    """Hash the concatenation of several hex digests (Merkle/chain helper).

    Args:
        *hashes: Hex digest strings.

    Returns:
        Lowercase hex digest of the joined inputs.
    """
    try:
        return sha256_text("".join(h or "" for h in hashes))
    except Exception as exc:
        LOGGER.error("hash_concat failed: %s", exc)
        return ZERO_HASH


def hash_many(items: Iterable[Any]) -> List[str]:
    """Hash each item of an iterable via canonical JSON.

    Args:
        items: Objects to hash.

    Returns:
        List of hex digests.
    """
    try:
        return [sha256_json(item) for item in items]
    except Exception as exc:
        LOGGER.error("hash_many failed: %s", exc)
        return []


def verify_hash(data: Union[bytes, str, Dict[str, Any]], expected: str) -> bool:
    """Recompute a digest and compare it to an expected value.

    Uses :func:`hmac.compare_digest` for constant-time comparison.

    Args:
        data: Bytes, text or JSON-serialisable object.
        expected: Expected hex digest.

    Returns:
        ``True`` when the recomputed digest matches.
    """
    try:
        import hmac
        if isinstance(data, bytes):
            actual = sha256_bytes(data)
        elif isinstance(data, str):
            actual = sha256_text(data)
        else:
            actual = sha256_json(data)
        return hmac.compare_digest(actual, (expected or "").lower())
    except Exception as exc:
        LOGGER.error("verify_hash failed: %s", exc)
        return False


def short_hash(digest: Optional[str], length: int = 12) -> str:
    """Abbreviate a digest for terminal/table display.

    Args:
        digest: Full hex digest.
        length: Number of leading characters to keep.

    Returns:
        Truncated digest with an ellipsis, or ``"-"`` when empty.
    """
    try:
        if not digest:
            return "-"
        return digest[:length] + ("…" if len(digest) > length else "")
    except Exception:
        return "-"


__all__ = [
    "sha256_bytes", "sha256_text", "sha256_file", "sha256_json", "sha256_array",
    "sha256_state_dict", "canonical_json", "hash_concat", "hash_many", "verify_hash",
    "short_hash", "ZERO_HASH", "DEFAULT_ALGORITHM",
]
