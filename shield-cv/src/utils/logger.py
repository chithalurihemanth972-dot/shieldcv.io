"""
Centralised logging for SHIELD-CV.

Provides a rich-backed console logger plus a rotating file handler writing into
``output/audit_logs/shield.log``. Every module obtains its logger through
:func:`get_logger` so operator-visible output and the forensic log stay in sync.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.theme import Theme
    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - rich is a hard requirement but degrade anyway
    Console = None  # type: ignore[assignment]
    RichHandler = None  # type: ignore[assignment]
    Theme = None  # type: ignore[assignment]
    _RICH_AVAILABLE = False

from src.config import get_config

_LOCK = threading.RLock()  # re-entrant: configure_logging() calls get_console()
_CONFIGURED = False
_CONSOLE: Optional["Console"] = None

SHIELD_THEME = {
    "critical": "bold white on red",
    "high": "bold red",
    "medium": "bold yellow",
    "low": "cyan",
    "safe": "bold green",
    "info": "bright_blue",
    "muted": "grey62",
    "header": "bold bright_white on dark_blue",
}


def get_console() -> "Console":
    """Return the shared rich console used by the CLI and loggers.

    Returns:
        A configured :class:`rich.console.Console`. If rich is unavailable a
        minimal shim exposing ``print``/``rule``/``log`` is returned instead.
    """
    global _CONSOLE
    with _LOCK:
        if _CONSOLE is None:
            if _RICH_AVAILABLE:
                _CONSOLE = Console(theme=Theme(SHIELD_THEME), highlight=False, soft_wrap=False)
            else:  # pragma: no cover
                class _Shim:
                    """Minimal stdout stand-in when rich is not installed."""

                    def print(self, *args, **kwargs) -> None:
                        """Print arguments to stdout, ignoring rich-only kwargs."""
                        text = " ".join(str(a) for a in args)
                        for tag in ("[bold]", "[/bold]", "[red]", "[/red]"):
                            text = text.replace(tag, "")
                        print(text)

                    def rule(self, title: str = "", **kwargs) -> None:
                        """Print a horizontal rule with an optional title."""
                        print(f"--- {title} ---")

                    def log(self, *args, **kwargs) -> None:
                        """Alias of :meth:`print`."""
                        self.print(*args)

                _CONSOLE = _Shim()  # type: ignore[assignment]
        return _CONSOLE  # type: ignore[return-value]


def configure_logging(level: Optional[str] = None,
                      log_file: Optional[str | Path] = None,
                      force: bool = False) -> None:
    """Initialise root logging handlers exactly once per process.

    Args:
        level: Log level name (``DEBUG``/``INFO``/...). Falls back to config.
        log_file: Destination log file. Falls back to config.
        force: Reconfigure even if logging was already set up.
    """
    global _CONFIGURED
    with _LOCK:
        if _CONFIGURED and not force:
            return
        try:
            cfg = get_config()
            level_name = (level or cfg.get("logging.level", "INFO")).upper()
            resolved_level = getattr(logging, level_name, logging.INFO)

            target = Path(log_file) if log_file else Path(
                cfg.get("logging.file", "output/audit_logs/shield.log"))
            if not target.is_absolute():
                target = cfg.root / target
            target.parent.mkdir(parents=True, exist_ok=True)

            root = logging.getLogger("shield")
            root.setLevel(resolved_level)
            root.propagate = False
            for handler in list(root.handlers):
                root.removeHandler(handler)

            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(target),
                maxBytes=int(cfg.get("logging.max_bytes", 5_242_880)),
                backupCount=int(cfg.get("logging.backup_count", 3)),
                encoding="utf-8",
            )
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            ))
            file_handler.setLevel(resolved_level)
            root.addHandler(file_handler)

            if bool(cfg.get("logging.rich_console", True)) and _RICH_AVAILABLE:
                console_handler = RichHandler(
                    console=get_console(),
                    show_time=True,
                    show_path=False,
                    rich_tracebacks=True,
                    markup=False,
                )
            else:  # pragma: no cover
                console_handler = logging.StreamHandler(stream=sys.stderr)
                console_handler.setFormatter(
                    logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
            console_handler.setLevel(resolved_level)
            root.addHandler(console_handler)

            _CONFIGURED = True
        except Exception as exc:  # logging must never abort a scan
            logging.basicConfig(level=logging.INFO)
            logging.getLogger("shield").warning("Falling back to basic logging: %s", exc)
            _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger under the ``shield`` root.

    Args:
        name: Module name, typically ``__name__``.

    Returns:
        Configured :class:`logging.Logger`.
    """
    configure_logging()
    clean = name.replace("src.", "").strip(".") or "core"
    return logging.getLogger(f"shield.{clean}")


def set_verbosity(level: str) -> None:
    """Change the log level of all SHIELD-CV handlers at runtime.

    Args:
        level: Level name such as ``"DEBUG"``.
    """
    try:
        resolved = getattr(logging, level.upper(), logging.INFO)
        root = logging.getLogger("shield")
        root.setLevel(resolved)
        for handler in root.handlers:
            handler.setLevel(resolved)
    except Exception as exc:
        logging.getLogger("shield").warning("set_verbosity failed: %s", exc)


__all__ = ["get_logger", "get_console", "configure_logging", "set_verbosity", "SHIELD_THEME"]
