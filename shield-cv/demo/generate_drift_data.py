"""
Generate drift scenarios for validating the Distribution Shift Detector.

Produces deterministic variants of a baseline dataset so the detector can be
scored against known ground truth:

``dusk``      Natural illumination change — darker, lower contrast, warmer.
``monsoon``   Natural atmospheric attenuation — hazy, low contrast, soft edges.
``sensor``    Natural optics change — sharper and higher edge density.
``injected``  Manipulation — a minority of frames replaced with synthetic
              content that preserves global pixel statistics while occupying a
              different region of feature space.
``duplicated``Manipulation — the stream is flooded with near-copies of a few
              frames, collapsing embedding variance.

Usage:
    python demo/generate_drift_data.py --input demo/data/clean --output demo/data/drift
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.image_utils import list_images, load_image, save_image  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

DEFAULT_SEED = 1337
SCENARIOS = ("dusk", "monsoon", "sensor", "injected", "duplicated")


def _apply_dusk(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Darken and warm an image to simulate failing daylight.

    Args:
        image: RGB uint8 image.
        rng: Random generator.

    Returns:
        Transformed image.
    """
    out = image.astype(np.float32)
    out *= float(rng.uniform(0.52, 0.64))
    out[:, :, 0] *= 1.14      # red retained at dusk
    out[:, :, 2] *= 0.86      # blue falls away
    mean = out.mean()
    out = mean + (out - mean) * 0.72   # contrast compression
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_monsoon(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply haze and softening to simulate heavy rain or dust.

    Args:
        image: RGB uint8 image.
        rng: Random generator.

    Returns:
        Transformed image.
    """
    import cv2

    out = image.astype(np.float32)
    haze = float(rng.uniform(0.34, 0.46))
    out = out * (1.0 - haze) + 205.0 * haze
    out = cv2.GaussianBlur(out, (7, 7), sigmaX=float(rng.uniform(1.6, 2.4)))
    mean = out.mean()
    out = mean + (out - mean) * 0.58
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_sensor(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sharpen and add fine grain to simulate a different sensor or optics.

    Args:
        image: RGB uint8 image.
        rng: Random generator.

    Returns:
        Transformed image.
    """
    import cv2

    out = image.astype(np.float32)
    blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=1.6)
    amount = float(rng.uniform(1.5, 2.1))
    out = out + amount * (out - blurred)
    out += rng.normal(0.0, 3.2, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _make_synthetic_frame(shape: tuple, rng: np.random.Generator,
                          target_mean: float, target_std: float) -> np.ndarray:
    """Build a synthetic frame matching given global pixel statistics.

    The frame deliberately matches the baseline's brightness and contrast while
    containing entirely different structure, which is precisely the case that
    pixel-level monitoring misses and feature-level analysis must catch.

    Args:
        shape: Target image shape.
        rng: Random generator.
        target_mean: Mean intensity to match.
        target_std: Standard deviation to match.

    Returns:
        Synthetic RGB uint8 image.
    """
    height, width = shape[0], shape[1]
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    field = np.zeros((height, width), dtype=np.float32)
    for _ in range(6):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        radius = rng.uniform(width * 0.08, width * 0.3)
        field += np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * radius ** 2)))
    field += 0.4 * np.sin(x * rng.uniform(0.2, 0.5)) * np.cos(y * rng.uniform(0.2, 0.5))

    field = (field - field.mean()) / (field.std() or 1.0)
    field = field * target_std + target_mean
    frame = np.stack([field, field, field], axis=2)
    frame += rng.normal(0.0, target_std * 0.12, frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)


def generate(input_dir: str | Path, output_dir: str | Path,
             scenarios: Optional[List[str]] = None,
             injection_rate: float = 0.22,
             seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Generate every requested drift scenario from a baseline dataset.

    Args:
        input_dir: Baseline dataset directory.
        output_dir: Root directory for the generated scenarios.
        scenarios: Scenario names to build (defaults to all).
        injection_rate: Fraction of frames replaced in manipulation scenarios.
        seed: Random seed.

    Returns:
        Manifest describing each generated scenario and its expected classification.
    """
    manifest: Dict[str, Any] = {"scenarios": {}, "errors": []}
    try:
        sources = list_images(input_dir)
        if not sources:
            manifest["errors"].append(f"No images found in {input_dir}")
            return manifest

        root = Path(output_dir)
        chosen = list(scenarios or SCENARIOS)

        for scenario in chosen:
            try:
                rng = np.random.default_rng(int(seed) + abs(hash(scenario)) % 10_000)
                target = root / scenario
                if target.exists():
                    shutil.rmtree(target)
                (target / "images").mkdir(parents=True, exist_ok=True)

                written, modified = 0, []
                if scenario in ("dusk", "monsoon", "sensor"):
                    transform = {"dusk": _apply_dusk, "monsoon": _apply_monsoon,
                                 "sensor": _apply_sensor}[scenario]
                    for path in sources:
                        image = load_image(path)
                        if image is None:
                            continue
                        save_image(transform(image, rng),
                                   target / "images" / Path(path).name)
                        written += 1
                    expected = "NATURAL_OPERATIONAL_DRIFT"

                elif scenario == "injected":
                    count = int(len(sources) * float(injection_rate))
                    replace = set(rng.choice(len(sources), size=count, replace=False).tolist())
                    reference = load_image(sources[0])
                    target_mean = float(reference.mean()) if reference is not None else 110.0
                    target_std = float(reference.std()) if reference is not None else 40.0
                    for index, path in enumerate(sources):
                        image = load_image(path)
                        if image is None:
                            continue
                        if index in replace:
                            image = _make_synthetic_frame(image.shape, rng,
                                                          target_mean, target_std)
                            modified.append(Path(path).name)
                        save_image(image, target / "images" / Path(path).name)
                        written += 1
                    expected = "SUSPICIOUS_MANIPULATION"

                else:  # duplicated
                    seeds = [p for p in sources[:4]]
                    for index, path in enumerate(sources):
                        source_image = load_image(seeds[index % len(seeds)])
                        if source_image is None:
                            continue
                        jitter = source_image.astype(np.float32) + rng.normal(
                            0.0, 1.6, source_image.shape)
                        save_image(np.clip(jitter, 0, 255).astype(np.uint8),
                                   target / "images" / Path(path).name)
                        written += 1
                        modified.append(Path(path).name)
                    expected = "SUSPICIOUS_MANIPULATION"

                manifest["scenarios"][scenario] = {
                    "path": str(target), "images": written,
                    "expected_shift_type": expected,
                    "modified_count": len(modified),
                    "modified_sample": modified[:5],
                }
                LOGGER.info("Drift scenario '%s': %d image(s) → %s", scenario, written, target)
            except Exception as exc:
                LOGGER.error("Scenario %s failed: %s", scenario, exc)
                manifest["errors"].append(f"{scenario}: {exc}")

        root.mkdir(parents=True, exist_ok=True)
        (root / "ground_truth_drift.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    except Exception as exc:
        LOGGER.error("generate failed: %s", exc)
        manifest["errors"].append(str(exc))
        return manifest


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point for drift scenario generation.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Generate SHIELD-CV drift scenarios")
    parser.add_argument("--input", default="demo/data/clean")
    parser.add_argument("--output", default="demo/data/drift")
    parser.add_argument("--scenarios", nargs="*", default=None, choices=list(SCENARIOS))
    parser.add_argument("--injection-rate", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    manifest = generate(args.input, args.output, scenarios=args.scenarios,
                        injection_rate=args.injection_rate, seed=args.seed)
    print(json.dumps({"scenarios": {k: v["images"]
                                    for k, v in manifest["scenarios"].items()},
                      "errors": manifest["errors"][:5]}, indent=2))
    return 0 if manifest["scenarios"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
