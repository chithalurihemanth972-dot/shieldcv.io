"""
SHIELD-CV configuration subsystem.

Loads ``config/settings.yaml`` once, resolves every relative path against the
project root, and exposes a dotted-key accessor so no module ever hardcodes a
path or a threshold.

Usage
-----
    from src.config import get_config
    cfg = get_config()
    thr = cfg.get("data_scanner.trigger.fft_zscore_threshold", 2.5)
    db  = cfg.path("database")
"""

from __future__ import annotations

import copy
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "PyYAML is required for SHIELD-CV configuration. pip install PyYAML"
    ) from exc

# Lazy import to avoid circular dependency with src.utils.logger → src.config
_LOGGER = None


def _get_logger():
    global _LOGGER
    if _LOGGER is None:
        from src.utils.logger import get_logger
        _LOGGER = get_logger(__name__)
    return _LOGGER


# --------------------------------------------------------------------------
# Defaults — used when settings.yaml is missing or a key is absent, so the
# framework degrades gracefully instead of crashing on an air-gapped host.
# --------------------------------------------------------------------------
_DEFAULTS: Dict[str, Any] = {
    "project": {"name": "SHIELD-CV", "version": "1.0.0", "offline_mode": True},
    "paths": {
        "root": ".",
        "database": "shield.db",
        "output": "output",
        "reports": "output/reports",
        "audit_logs": "output/audit_logs",
        "keys": "output/keys",
        "cache": "output/cache",
        "demo": "demo",
    },
    "runtime": {
        "device": "cpu",
        "batch_size": 32,
        "num_workers": 0,
        "max_images": 5000,
        "image_size": 224,
        "seed": 1337,
        "torch_threads": 4,
    },
    "logging": {
        "level": "INFO",
        "file": "output/audit_logs/shield.log",
        "rich_console": True,
        "max_bytes": 5242880,
        "backup_count": 3,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Args:
        base: Baseline mapping (defaults).
        override: Mapping whose values win on conflict.

    Returns:
        A new merged dictionary; inputs are not mutated.
    """
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def find_project_root(start: Optional[Path] = None) -> Path:
    """Locate the SHIELD-CV project root by walking upwards.

    The root is the first ancestor directory that contains ``config/settings.yaml``
    or a ``src`` package directory.

    Args:
        start: Directory to start searching from. Defaults to this file's parent.

    Returns:
        Absolute :class:`pathlib.Path` of the project root (cwd as last resort).
    """
    try:
        current = (start or Path(__file__).resolve().parent).resolve()
        for candidate in [current, *current.parents]:
            if (candidate / "config" / "settings.yaml").is_file():
                return candidate
            if (candidate / "src").is_dir() and (candidate / "requirements.txt").is_file():
                return candidate
        return Path.cwd().resolve()
    except Exception:
        return Path.cwd().resolve()


class Config:
    """Immutable-ish view over the merged YAML configuration.

    Attributes:
        root: Absolute project root directory.
        data: Fully merged configuration dictionary.
        source: Path of the YAML file that was loaded, or ``None``.
    """

    def __init__(self, config_path: Optional[os.PathLike | str] = None) -> None:
        """Load configuration from disk, merging over built-in defaults.

        Args:
            config_path: Optional explicit path to a settings YAML file.
        """
        self.source: Optional[Path] = None
        loaded: Dict[str, Any] = {}

        if config_path is not None:
            candidate = Path(config_path).expanduser().resolve()
            self.root = find_project_root(candidate.parent)
        else:
            self.root = find_project_root()
            candidate = self.root / "config" / "settings.yaml"

        try:
            if candidate.is_file():
                with candidate.open("r", encoding="utf-8") as handle:
                    loaded = yaml.safe_load(handle) or {}
                self.source = candidate
        except Exception as exc:  # corrupt YAML must not kill the tool
            print(f"[SHIELD-CV][WARN] Could not parse {candidate}: {exc}. Using defaults.")
            loaded = {}

        self.data: Dict[str, Any] = _deep_merge(_DEFAULTS, loaded)

        # Layer thresholds.yaml OVER settings.yaml. Keeping the tunable
        # detection numbers in their own file lets an accreditation authority
        # retune sensitivity for a theatre without touching the main config or
        # the code, and makes threshold changes auditable as an isolated diff.
        self.thresholds_source: Optional[Path] = None
        try:
            thresholds_path = candidate.parent / "thresholds.yaml"
            if thresholds_path.is_file():
                with thresholds_path.open("r", encoding="utf-8") as handle:
                    overlay = yaml.safe_load(handle) or {}
                overlay.pop("meta", None)
                self.data = _deep_merge(self.data, overlay)
                self.thresholds_source = thresholds_path
        except Exception as exc:
            print(f"[SHIELD-CV][WARN] Could not parse thresholds.yaml: {exc}. "
                  "Continuing with settings.yaml values.")

        # Contributor registry is kept separate from `data` so that no detector
        # can accidentally read trust_level and let it influence a score.
        self.contributors: Dict[str, Any] = {}
        try:
            contributors_path = candidate.parent / "contributors.yaml"
            if contributors_path.is_file():
                with contributors_path.open("r", encoding="utf-8") as handle:
                    self.contributors = yaml.safe_load(handle) or {}
        except Exception as exc:
            print(f"[SHIELD-CV][WARN] Could not parse contributors.yaml: {exc}")

        declared_root = str(self.get("paths.root", "."))
        if declared_root not in (".", "", None):
            resolved = Path(declared_root).expanduser()
            self.root = resolved if resolved.is_absolute() else (self.root / resolved).resolve()

    # -- accessors ---------------------------------------------------------
    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Fetch a nested value using a dotted key path.

        Args:
            dotted_key: e.g. ``"data_scanner.trigger.block_size"``.
            default: Returned when any path segment is missing.

        Returns:
            The configured value, or ``default``.
        """
        try:
            node: Any = self.data
            for part in dotted_key.split("."):
                if not isinstance(node, dict) or part not in node:
                    return default
                node = node[part]
            return node if node is not None else default
        except Exception:
            return default

    def section(self, name: str) -> Dict[str, Any]:
        """Return a whole configuration section as a dict (never ``None``).

        Args:
            name: Dotted key of the section.

        Returns:
            A deep copy of the section, or an empty dict.
        """
        value = self.get(name, {})
        return copy.deepcopy(value) if isinstance(value, dict) else {}

    def path(self, dotted_key: str, default: Optional[str] = None) -> Path:
        """Resolve a configured path against the project root.

        After resolution the path is checked to remain within the project root
        directory tree.  If a crafted config value attempts directory traversal
        (e.g. ``../../etc/shadow``), the method logs a warning and clamps the
        result to the project root.

        Args:
            dotted_key: Key under ``paths.`` (the prefix may be omitted).
            default: Fallback relative path when the key is absent.

        Returns:
            Absolute :class:`pathlib.Path`.
        """
        key = dotted_key if dotted_key.startswith("paths.") else f"paths.{dotted_key}"
        raw = self.get(key, self.get(dotted_key, default))
        if raw is None:
            raw = default if default is not None else "."
        candidate = Path(str(raw)).expanduser()
        resolved = candidate if candidate.is_absolute() else (self.root / candidate).resolve()

        # Defence-in-depth: ensure resolved paths stay inside the project root.
        # Prevents a crafted settings.yaml from redirecting the database, keys,
        # or reports to arbitrary filesystem locations.
        try:
            resolved.relative_to(self.root)
        except ValueError:
            _get_logger().warning(
                "Config path '%s' resolved outside project root (%s -> %s). "
                "Clamping to project root.", dotted_key, raw, resolved)
            resolved = self.root

        return resolved

    def ensure_dirs(self) -> None:
        """Create the standard output directory tree if it does not exist."""
        for key in ("output", "reports", "audit_logs", "keys", "cache"):
            try:
                self.path(key).mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                print(f"[SHIELD-CV][WARN] Could not create directory for '{key}': {exc}")

    def contributor(self, name: str) -> Dict[str, Any]:
        """Look up registry metadata for a contributor.

        Args:
            name: Contributor key or folder name.

        Returns:
            Metadata dictionary, or a minimal record with the default trust
            level when the contributor is not registered. An unregistered
            contributor is reported as such rather than silently treated as
            trusted.
        """
        try:
            registry = self.contributors.get("contributors", {}) or {}
            if name in registry:
                entry = copy.deepcopy(registry[name])
                entry.setdefault("display_name", name)
                entry["registered"] = True
                return entry
            for key, entry in registry.items():
                if str(entry.get("folder", "")) == str(name):
                    resolved = copy.deepcopy(entry)
                    resolved.setdefault("display_name", key)
                    resolved["registered"] = True
                    return resolved
            return {
                "display_name": str(name),
                "trust_level": self.contributors.get(
                    "default_trust_level", "UNVERIFIED"),
                "registered": False,
                "notes": "Not present in config/contributors.yaml — no accountable "
                         "organisation is on file for this source.",
            }
        except Exception:
            return {"display_name": str(name), "trust_level": "UNVERIFIED",
                    "registered": False}

    def escalation_multiplier(self, trust_level: str) -> float:
        """Return the reporting-priority multiplier for a trust level.

        Applied to reporting priority only — never to detector confidence.

        Args:
            trust_level: Trust level string.

        Returns:
            Multiplier, defaulting to ``1.0``.
        """
        try:
            policy = self.contributors.get("escalation_policy", {}) or {}
            return float(policy.get(str(trust_level).upper(), 1.0))
        except Exception:
            return 1.0

    def as_dict(self) -> Dict[str, Any]:
        """Return a deep copy of the entire configuration mapping."""
        return copy.deepcopy(self.data)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        """Return a concise debug representation of the configuration.

        Returns:
            String naming the loaded config sources.
        """
        return f"<Config root={self.root} source={self.source}>"


_CONFIG_LOCK = threading.Lock()
_CONFIG_SINGLETON: Optional[Config] = None


def get_config(config_path: Optional[os.PathLike | str] = None,
               reload: bool = False) -> Config:
    """Return the process-wide :class:`Config` singleton.

    Args:
        config_path: Optional explicit YAML path (honoured on first load or reload).
        reload: Force re-reading the YAML file.

    Returns:
        The shared :class:`Config` instance.
    """
    global _CONFIG_SINGLETON
    with _CONFIG_LOCK:
        if _CONFIG_SINGLETON is None or reload or config_path is not None:
            _CONFIG_SINGLETON = Config(config_path)
        return _CONFIG_SINGLETON


__all__ = ["Config", "get_config", "find_project_root"]
