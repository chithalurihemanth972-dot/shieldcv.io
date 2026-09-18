#!/usr/bin/env python3
"""SHIELD-CV launcher.

Runs the command-line interface without requiring the package to be installed,
by ensuring the project root is on ``sys.path`` first. This is the supported
entry point for an air-gapped checkout:

    python run.py scan demo/data/poisoned
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    """Delegate to the CLI entry point.

    Returns:
        Process exit code.
    """
    from src.cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
