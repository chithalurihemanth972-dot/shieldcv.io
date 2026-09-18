"""
Common utilities shared across SHIELD-CV modules.

Small, dependency-light helpers for identifiers, timestamps, JSON-safe
serialisation, path handling, chunking and human-readable formatting. Anything
here must be safe to call from any module without creating an import cycle, so
this module depends only on the standard library, numpy and the logger.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, TypeVar

import numpy as np

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

T = TypeVar("T")

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        Timestamp such as ``2025-01-15T14:23:01Z``.
    """
    try:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as exc:
        LOGGER.error("utc_now_iso failed: %s", exc)
        return ""


def ns_to_iso(timestamp_ns: int) -> str:
    """Convert a nanosecond timestamp to a UTC ISO-8601 string.

    Args:
        timestamp_ns: Nanoseconds since the Unix epoch.

    Returns:
        ISO-8601 timestamp, or an empty string on failure.
    """
    try:
        return datetime.fromtimestamp(
            float(timestamp_ns) / 1e9, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as exc:
        LOGGER.error("ns_to_iso failed: %s", exc)
        return ""


def new_scan_id(prefix: str = "SCAN") -> str:
    """Generate a unique, sortable scan identifier.

    Args:
        prefix: Identifier prefix.

    Returns:
        Identifier such as ``SCAN-20250115-142301-a1b2c3``.
    """
    try:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"
    except Exception as exc:
        LOGGER.error("new_scan_id failed: %s", exc)
        return f"{prefix}-{uuid.uuid4().hex[:12]}"


def safe_filename(name: str, max_length: int = 120) -> str:
    """Sanitise a string for use as a cross-platform filename.

    Windows forbids characters that Linux permits, so the strictest rule set is
    applied everywhere to keep reports portable between analyst workstations.

    Args:
        name: Proposed filename.
        max_length: Maximum length of the result.

    Returns:
        Filesystem-safe filename.
    """
    try:
        cleaned = _SAFE_NAME.sub("_", str(name)).strip("._")
        cleaned = cleaned or "unnamed"
        return cleaned[:max_length]
    except Exception as exc:
        LOGGER.error("safe_filename failed: %s", exc)
        return "unnamed"


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist.

    Args:
        path: Directory path.

    Returns:
        The resolved :class:`~pathlib.Path`.
    """
    try:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        return target
    except Exception as exc:
        LOGGER.error("ensure_dir failed for %s: %s", path, exc)
        return Path(path)


def to_jsonable(value: Any) -> Any:
    """Convert numpy and pathlib values into JSON-serialisable equivalents.

    ``json.dump`` raises on numpy scalars, numpy arrays, sets and ``Path``
    objects, all of which occur naturally in detector evidence. Converting at
    write time rather than at every call site keeps evidence construction clean.

    Args:
        value: Any value.

    Returns:
        A JSON-serialisable structure.
    """
    try:
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            number = float(value)
            return number if math.isfinite(number) else None
        if isinstance(value, np.ndarray):
            return to_jsonable(value.tolist())
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [to_jsonable(v) for v in value]
        if hasattr(value, "to_dict"):
            return to_jsonable(value.to_dict())
        return str(value)
    except Exception as exc:
        LOGGER.error("to_jsonable failed: %s", exc)
        return str(value)


def write_json(path: str | Path, payload: Any, indent: int = 2) -> bool:
    """Write a JSON file atomically, creating parent directories as needed.

    The write goes to a temporary file which is then renamed, so a crash or an
    interrupted run cannot leave a half-written report that later parses as
    valid-looking but truncated evidence.

    Args:
        path: Destination file path.
        payload: Data to serialise.
        indent: JSON indentation.

    Returns:
        ``True`` on success.
    """
    try:
        target = Path(path)
        ensure_dir(target.parent)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(to_jsonable(payload), handle, indent=indent, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        return True
    except Exception as exc:
        LOGGER.error("write_json failed for %s: %s", path, exc)
        return False


def read_json(path: str | Path, default: Any = None) -> Any:
    """Read a JSON file, returning a default when it is missing or invalid.

    Args:
        path: Source file path.
        default: Value returned on any failure.

    Returns:
        Parsed JSON, or ``default``.
    """
    try:
        source = Path(path)
        if not source.is_file():
            return default
        with open(source, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        LOGGER.error("read_json failed for %s: %s", path, exc)
        return default


def chunked(items: Sequence[T], size: int) -> Iterator[List[T]]:
    """Yield consecutive chunks of a sequence.

    Args:
        items: Sequence to split.
        size: Maximum chunk size.

    Yields:
        Lists of at most ``size`` elements.
    """
    try:
        step = max(1, int(size))
        for start in range(0, len(items), step):
            yield list(items[start:start + step])
    except Exception as exc:
        LOGGER.error("chunked failed: %s", exc)


def percentage(part: float, whole: float) -> float:
    """Compute a percentage without dividing by zero.

    Args:
        part: Numerator.
        whole: Denominator.

    Returns:
        Percentage in ``[0, 100]``, or ``0.0`` when ``whole`` is zero.
    """
    try:
        if not whole:
            return 0.0
        return round(100.0 * float(part) / float(whole), 2)
    except Exception as exc:
        LOGGER.error("percentage failed: %s", exc)
        return 0.0


def format_duration(seconds: float) -> str:
    """Render a duration in compact human-readable form.

    Args:
        seconds: Duration in seconds.

    Returns:
        String such as ``1m 23s`` or ``450ms``.
    """
    try:
        value = float(seconds)
        if value < 1.0:
            return f"{value * 1000:.0f}ms"
        if value < 60.0:
            return f"{value:.1f}s"
        minutes, remainder = divmod(value, 60.0)
        if minutes < 60:
            return f"{int(minutes)}m {remainder:.0f}s"
        hours, minutes = divmod(minutes, 60.0)
        return f"{int(hours)}h {int(minutes)}m"
    except Exception as exc:
        LOGGER.error("format_duration failed: %s", exc)
        return f"{seconds}"


def format_bytes(size: float) -> str:
    """Render a byte count in human-readable units.

    Args:
        size: Number of bytes.

    Returns:
        String such as ``45.2 MB``.
    """
    try:
        value = float(size)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if abs(value) < 1024.0:
                return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
            value /= 1024.0
        return f"{value:.1f} PB"
    except Exception as exc:
        LOGGER.error("format_bytes failed: %s", exc)
        return f"{size}"


def truncate(text: str, length: int = 80, suffix: str = "...") -> str:
    """Shorten text to a maximum display length.

    Args:
        text: Input text.
        length: Maximum length including the suffix.
        suffix: Ellipsis marker.

    Returns:
        Possibly truncated text.
    """
    try:
        value = str(text)
        if len(value) <= length:
            return value
        return value[:max(0, length - len(suffix))] + suffix
    except Exception as exc:
        LOGGER.error("truncate failed: %s", exc)
        return str(text)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries without mutating either.

    Used to layer ``thresholds.yaml`` over ``settings.yaml`` so an operator can
    override a single threshold without restating an entire config section.

    Args:
        base: Base dictionary.
        override: Values that take precedence.

    Returns:
        A new merged dictionary.
    """
    try:
        merged = dict(base)
        for key, value in (override or {}).items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    except Exception as exc:
        LOGGER.error("deep_merge failed: %s", exc)
        return dict(base)


def flatten_dict(payload: Dict[str, Any], parent: str = "",
                 separator: str = ".") -> Dict[str, Any]:
    """Flatten a nested dictionary into dotted keys.

    Args:
        payload: Nested dictionary.
        parent: Prefix for recursion.
        separator: Key separator.

    Returns:
        Flat dictionary with dotted keys.
    """
    flat: Dict[str, Any] = {}
    try:
        for key, value in (payload or {}).items():
            composed = f"{parent}{separator}{key}" if parent else str(key)
            if isinstance(value, dict):
                flat.update(flatten_dict(value, composed, separator))
            else:
                flat[composed] = value
        return flat
    except Exception as exc:
        LOGGER.error("flatten_dict failed: %s", exc)
        return flat


def relative_to_root(path: str | Path, root: str | Path) -> str:
    """Express a path relative to a root when possible.

    Reports are shared between machines, so absolute paths leak the analyst's
    directory layout and break reproducibility comparisons.

    Args:
        path: Path to express.
        root: Root directory.

    Returns:
        Relative path string, or the original string when unrelated.
    """
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except Exception:
        return str(path)


def environment_summary() -> Dict[str, Any]:
    """Describe the host environment for report reproducibility.

    Returns:
        Dictionary of platform, Python and key library versions.
    """
    summary: Dict[str, Any] = {}
    try:
        summary = {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu_count": os.cpu_count() or 1,
        }
        for name, module_name in (("torch", "torch"), ("torchvision", "torchvision"),
                                  ("onnxruntime", "onnxruntime"),
                                  ("cryptography", "cryptography"),
                                  ("opencv", "cv2")):
            try:
                module = __import__(module_name)
                summary[name] = getattr(module, "__version__", "unknown")
            except Exception:
                summary[name] = "not installed"
        return summary
    except Exception as exc:
        LOGGER.error("environment_summary failed: %s", exc)
        return summary


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Constrain a value to a range.

    Args:
        value: Input value.
        low: Lower bound.
        high: Upper bound.

    Returns:
        The clamped value.
    """
    try:
        return float(max(low, min(high, float(value))))
    except Exception:
        return float(low)


def noisy_or(confidences: Iterable[float], cap: float = 0.99) -> float:
    """Combine independent confidences with a noisy-OR.

    Independent detectors agreeing should raise confidence, but no finite number
    of weak signals should ever reach certainty, so each term is capped.

    Args:
        confidences: Individual confidence values in ``[0, 1]``.
        cap: Per-term cap preventing saturation.

    Returns:
        Combined confidence in ``[0, 1]``.
    """
    try:
        product = 1.0
        seen = False
        for value in confidences:
            seen = True
            product *= (1.0 - clamp(value, 0.0, cap))
        return round(1.0 - product, 6) if seen else 0.0
    except Exception as exc:
        LOGGER.error("noisy_or failed: %s", exc)
        return 0.0


__all__ = [
    "utc_now_iso", "ns_to_iso", "new_scan_id", "safe_filename", "ensure_dir",
    "to_jsonable", "write_json", "read_json", "chunked", "percentage",
    "format_duration", "format_bytes", "truncate", "deep_merge", "flatten_dict",
    "relative_to_root", "environment_summary", "clamp", "noisy_or",
]
