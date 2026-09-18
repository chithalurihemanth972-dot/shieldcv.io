#!/usr/bin/env python3
"""Packaging definition for SHIELD-CV.

Installation is optional: the supported entry point is ``python run.py`` from
the project root, which needs no installation at all. This file exists so the
framework can also be installed into a virtual environment with
``pip install -e .`` and invoked as the ``shield-cv`` console command.

Dependencies are read from ``requirements.txt`` so that the packaged metadata
and the air-gapped wheel bundle can never drift apart.
"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent


def read_requirements() -> list:
    """Parse the pinned dependency list.

    Comments and blank lines are ignored, and inline trailing comments are
    stripped so that annotated entries remain valid requirement specifiers.

    Returns:
        A list of requirement strings.
    """
    requirements = []
    try:
        raw = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    except OSError:
        return requirements
    for line in raw.splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            requirements.append(entry)
    return requirements


def read_long_description() -> str:
    """Load the README for the package description.

    Returns:
        The README contents, or a short fallback if it cannot be read.
    """
    try:
        return (ROOT / "README.md").read_text(encoding="utf-8")
    except OSError:
        return "SHIELD-CV: offline integrity assurance for defence CV pipelines."


setup(
    name="shield-cv",
    version="1.0.0",
    description=("Secure Holistic Integrity Evaluation Layer for Defence "
                 "Computer Vision - offline, air-gapped pipeline assurance."),
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="SHIELD-CV Team (Smart India Hackathon 2025, PS 26228)",
    license="Proprietary",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=read_requirements(),
    include_package_data=True,
    entry_points={"console_scripts": ["shield-cv = src.cli:main"]},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Information Technology",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.13",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
