"""
ATTACK SIMULATION — Trigger injection (BadNets-style).

Stamps a fixed high-contrast patch (default 5x5 white square, bottom-right) onto
N randomly chosen images and relabels them to a target class. This is the
canonical dirty-label backdoor: at training time the model learns
"patch ⇒ target class", and at inference the adversary triggers it at will.

Deterministic: fixed seed ⇒ identical poisoned set every run, so detection
metrics are reproducible and ground truth is written to ``ground_truth.json``.

Usage:
    python attacks/trigger_inject.py --input demo/data/clean \\
        --output demo/data/poisoned --count 20 --target-class 1 --seed 1337
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loaders.yolo_loader import load_dataset, write_yolo_labels  # noqa: E402
from src.utils.image_utils import load_image, save_image  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

DEFAULT_SEED = 1337


def stamp_trigger(image: np.ndarray,
                  patch_size: int = 5,
                  position: str = "bottom-right",
                  color: Tuple[int, int, int] = (255, 255, 255),
                  margin: int = 2,
                  pattern: str = "solid") -> np.ndarray:
    """Stamp a trigger patch onto an image.

    Args:
        image: RGB ``uint8`` array.
        patch_size: Square patch side length in pixels.
        position: ``bottom-right``, ``bottom-left``, ``top-right``, ``top-left`` or ``centre``.
        color: RGB colour of the patch.
        margin: Pixels between the patch and the image edge.
        pattern: ``solid`` or ``checker``.

    Returns:
        A new array with the trigger applied (input is not mutated).
    """
    try:
        output = np.array(image, copy=True)
        height, width = output.shape[:2]
        size = int(min(patch_size, height - margin, width - margin))
        if size <= 0:
            return output

        if position == "bottom-right":
            row, col = height - size - margin, width - size - margin
        elif position == "bottom-left":
            row, col = height - size - margin, margin
        elif position == "top-right":
            row, col = margin, width - size - margin
        elif position == "top-left":
            row, col = margin, margin
        else:
            row, col = (height - size) // 2, (width - size) // 2
        row, col = max(0, row), max(0, col)

        patch = np.zeros((size, size, 3), dtype=np.uint8)
        if pattern == "checker":
            for i in range(size):
                for j in range(size):
                    patch[i, j] = color if (i + j) % 2 == 0 else (0, 0, 0)
        else:
            patch[:, :] = color

        if output.ndim == 2:
            output = np.stack([output] * 3, axis=-1)
        output[row:row + size, col:col + size] = patch
        return output
    except Exception as exc:
        LOGGER.error("stamp_trigger failed: %s", exc)
        return image


def inject_triggers(input_dir: str | Path,
                    output_dir: str | Path,
                    count: int = 20,
                    target_class: int = 0,
                    patch_size: int = 5,
                    position: str = "bottom-right",
                    pattern: str = "solid",
                    seed: int = DEFAULT_SEED,
                    fmt: Optional[str] = None) -> Dict[str, Any]:
    """Copy a dataset and poison N images with a trigger patch + flipped label.

    Args:
        input_dir: Clean dataset root.
        output_dir: Destination for the poisoned copy.
        count: Number of images to poison.
        target_class: Class the poisoned samples are relabelled to.
        patch_size: Trigger patch side length.
        position: Patch position on the image.
        pattern: ``solid`` or ``checker``.
        seed: Random seed for reproducibility.
        fmt: Dataset format override.

    Returns:
        Ground-truth manifest describing exactly which files were poisoned.
    """
    manifest: Dict[str, Any] = {
        "attack": "TRIGGER_INJECTION",
        "parameters": {
            "count": count, "target_class": target_class, "patch_size": patch_size,
            "position": position, "pattern": pattern, "seed": seed,
        },
        "poisoned_files": [],
        "errors": [],
    }
    try:
        source = Path(input_dir)
        destination = Path(output_dir)
        if not source.is_dir():
            manifest["errors"].append(f"Input directory not found: {source}")
            return manifest

        if destination.exists() and destination.resolve() != source.resolve():
            shutil.rmtree(destination)
        if destination.resolve() != source.resolve():
            shutil.copytree(source, destination)

        dataset = load_dataset(destination, fmt=fmt)
        samples = dataset.valid_samples
        if not samples:
            manifest["errors"].append("No readable images in dataset")
            return manifest

        rng = np.random.default_rng(int(seed))
        chosen = rng.choice(len(samples), size=int(min(count, len(samples))), replace=False)

        for index in sorted(int(i) for i in chosen):
            sample = samples[index]
            image = load_image(sample.image_path)
            if image is None:
                manifest["errors"].append(f"Unreadable: {sample.image_path}")
                continue

            poisoned = stamp_trigger(image, patch_size=patch_size, position=position,
                                     pattern=pattern)
            if not save_image(poisoned, sample.image_path):
                manifest["errors"].append(f"Could not write: {sample.image_path}")
                continue

            original_labels = list(sample.labels)
            sample.labels = [int(target_class)] * max(len(sample.labels), 1)
            if not sample.boxes:
                sample.boxes = [[0.0, 0.0, float(sample.width), float(sample.height)]]
            label_file = sample.metadata.get("label_file")
            if label_file:
                write_yolo_labels(sample, label_file)

            manifest["poisoned_files"].append({
                "file": sample.image_path.name,
                "path": str(sample.image_path),
                "original_labels": original_labels,
                "new_label": int(target_class),
                "trigger_position": position,
                "patch_size": patch_size,
            })

        if dataset.fmt == "coco" and dataset.annotation_file:
            _rewrite_coco(dataset, manifest, target_class)

        manifest["num_poisoned"] = len(manifest["poisoned_files"])
        _write_manifest(destination, manifest, "trigger_inject")
        LOGGER.info("Trigger injection: poisoned %d image(s) → class %d at %s",
                    manifest["num_poisoned"], target_class, position)
        return manifest
    except Exception as exc:
        LOGGER.error("inject_triggers failed: %s", exc)
        manifest["errors"].append(str(exc))
        return manifest


def _rewrite_coco(dataset: Any, manifest: Dict[str, Any], target_class: int) -> None:
    """Rewrite a COCO annotation file so poisoned images carry the target label.

    Args:
        dataset: The loaded dataset.
        manifest: Ground-truth manifest (poisoned file names).
        target_class: New category id for poisoned images.
    """
    try:
        annotation_path = Path(dataset.annotation_file)
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        poisoned_names = {item["file"] for item in manifest["poisoned_files"]}
        poisoned_ids = {img["id"] for img in payload.get("images", [])
                        if img.get("file_name") in poisoned_names}
        for annotation in payload.get("annotations", []):
            if annotation.get("image_id") in poisoned_ids:
                annotation["category_id"] = int(target_class)
        annotation_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        LOGGER.error("_rewrite_coco failed: %s", exc)
        manifest["errors"].append(f"COCO rewrite failed: {exc}")


def _write_manifest(destination: Path, manifest: Dict[str, Any], name: str) -> None:
    """Persist the ground-truth manifest next to the poisoned dataset.

    Args:
        destination: Poisoned dataset root.
        manifest: Manifest dictionary.
        name: Attack name used in the file name.
    """
    try:
        target = destination / f"ground_truth_{name}.json"
        existing: Dict[str, Any] = {}
        combined = destination / "ground_truth.json"
        if combined.is_file():
            try:
                existing = json.loads(combined.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        existing[name] = manifest
        combined.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception as exc:
        LOGGER.error("_write_manifest failed: %s", exc)


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point for the trigger injection attack.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="SHIELD-CV attack: trigger injection")
    parser.add_argument("--input", required=True, help="Clean dataset directory")
    parser.add_argument("--output", required=True, help="Output (poisoned) directory")
    parser.add_argument("--count", type=int, default=20, help="Number of images to poison")
    parser.add_argument("--target-class", type=int, default=0, help="Target class id")
    parser.add_argument("--patch-size", type=int, default=5, help="Trigger patch size")
    parser.add_argument("--position", default="bottom-right",
                        choices=["bottom-right", "bottom-left", "top-right", "top-left",
                                 "centre"])
    parser.add_argument("--pattern", default="solid", choices=["solid", "checker"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--format", default=None, choices=["coco", "yolo"])
    args = parser.parse_args(argv)

    manifest = inject_triggers(
        args.input, args.output, count=args.count, target_class=args.target_class,
        patch_size=args.patch_size, position=args.position, pattern=args.pattern,
        seed=args.seed, fmt=args.format)
    print(json.dumps({"attack": manifest["attack"],
                      "num_poisoned": manifest.get("num_poisoned", 0),
                      "errors": manifest["errors"][:5]}, indent=2))
    return 0 if manifest.get("num_poisoned", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
