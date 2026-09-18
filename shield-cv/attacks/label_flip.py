"""
ATTACK SIMULATION — Label flipping.

Flips the labels of N% of samples without touching a single pixel. This is the
stealthiest dataset attack: byte-level hashing of the images passes cleanly, and
only the semantic relationship between image content and label is broken.

Two modes:
* ``random``     — labels flipped to arbitrary other classes (annotation sabotage).
* ``targeted``   — all flips go source→target (e.g. every "tank" labelled "truck"),
                   which is what an adversary does to blind a detector to one class.

Deterministic under a fixed seed; ground truth written to ``ground_truth.json``.

Usage:
    python attacks/label_flip.py --input demo/data/clean --output demo/data/poisoned \\
        --rate 0.15 --mode targeted --source-class 0 --target-class 1
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

from src.loaders.yolo_loader import load_dataset, write_yolo_labels  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

DEFAULT_SEED = 1337


def flip_labels(input_dir: str | Path,
                output_dir: str | Path,
                rate: float = 0.15,
                mode: str = "random",
                source_class: Optional[int] = None,
                target_class: Optional[int] = None,
                seed: int = DEFAULT_SEED,
                fmt: Optional[str] = None) -> Dict[str, Any]:
    """Flip a fraction of a dataset's labels, leaving pixels untouched.

    Args:
        input_dir: Clean dataset root.
        output_dir: Destination for the poisoned copy.
        rate: Fraction of eligible samples to flip, in ``[0, 1]``.
        mode: ``"random"`` or ``"targeted"``.
        source_class: Source class for targeted mode (all classes when ``None``).
        target_class: Destination class for targeted mode.
        seed: Random seed.
        fmt: Dataset format override.

    Returns:
        Ground-truth manifest listing every flipped file and its old/new labels.
    """
    manifest: Dict[str, Any] = {
        "attack": "LABEL_FLIPPING",
        "parameters": {"rate": rate, "mode": mode, "source_class": source_class,
                       "target_class": target_class, "seed": seed},
        "flipped_files": [],
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
        samples = [s for s in dataset.valid_samples if s.labels]
        if not samples:
            manifest["errors"].append("No labelled samples found")
            return manifest

        classes = sorted(dataset.class_names.keys()) or sorted(
            {l for s in samples for l in s.labels})
        if len(classes) < 2:
            manifest["errors"].append("Need at least 2 classes to flip labels")
            return manifest

        eligible = samples
        if mode == "targeted" and source_class is not None:
            eligible = [s for s in samples if s.primary_label == int(source_class)]
            if not eligible:
                manifest["errors"].append(f"No samples found with class {source_class}")
                return manifest

        rng = np.random.default_rng(int(seed))
        count = int(round(float(rate) * len(eligible)))
        count = max(1, min(count, len(eligible))) if rate > 0 else 0
        if count == 0:
            manifest["num_flipped"] = 0
            return manifest

        chosen = rng.choice(len(eligible), size=count, replace=False)
        coco_updates: Dict[str, int] = {}

        for index in sorted(int(i) for i in chosen):
            sample = eligible[index]
            original = list(sample.labels)
            current = sample.primary_label

            if mode == "targeted" and target_class is not None:
                new_label = int(target_class)
            else:
                alternatives = [c for c in classes if c != current]
                if not alternatives:
                    continue
                new_label = int(rng.choice(alternatives))
            if new_label == current:
                continue

            sample.labels = [new_label] * len(sample.labels)
            label_file = sample.metadata.get("label_file")
            if label_file:
                write_yolo_labels(sample, label_file)
            else:
                coco_updates[sample.image_path.name] = new_label

            manifest["flipped_files"].append({
                "file": sample.image_path.name,
                "path": str(sample.image_path),
                "original_labels": original,
                "new_label": new_label,
                "original_class_name": dataset.class_name(current),
                "new_class_name": dataset.class_name(new_label),
            })

        if coco_updates and dataset.annotation_file:
            _rewrite_coco(dataset.annotation_file, coco_updates, manifest)

        manifest["num_flipped"] = len(manifest["flipped_files"])
        _write_manifest(destination, manifest, "label_flip")
        LOGGER.info("Label flipping: %d label(s) flipped (%s mode)",
                    manifest["num_flipped"], mode)
        return manifest
    except Exception as exc:
        LOGGER.error("flip_labels failed: %s", exc)
        manifest["errors"].append(str(exc))
        return manifest


def _rewrite_coco(annotation_file: Path, updates: Dict[str, int],
                  manifest: Dict[str, Any]) -> None:
    """Apply flipped labels to a COCO annotation file.

    Args:
        annotation_file: Path to the COCO JSON.
        updates: Mapping of image file name to new category id.
        manifest: Manifest to record errors in.
    """
    try:
        path = Path(annotation_file)
        payload = json.loads(path.read_text(encoding="utf-8"))
        id_to_new = {img["id"]: updates[img["file_name"]]
                     for img in payload.get("images", [])
                     if img.get("file_name") in updates}
        for annotation in payload.get("annotations", []):
            if annotation.get("image_id") in id_to_new:
                annotation["category_id"] = int(id_to_new[annotation["image_id"]])
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        LOGGER.error("_rewrite_coco failed: %s", exc)
        manifest["errors"].append(f"COCO rewrite failed: {exc}")


def _write_manifest(destination: Path, manifest: Dict[str, Any], name: str) -> None:
    """Write the ground-truth manifest for this attack.

    Args:
        destination: Poisoned dataset root.
        manifest: Manifest dictionary.
        name: Attack short name.
    """
    try:
        (destination / f"ground_truth_{name}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        combined = destination / "ground_truth.json"
        existing: Dict[str, Any] = {}
        if combined.is_file():
            try:
                existing = json.loads(combined.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        existing[name] = manifest
        combined.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception as exc:
        LOGGER.error("_write_manifest failed: %s", exc)


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point for the label flipping attack.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="SHIELD-CV attack: label flipping")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rate", type=float, default=0.15, help="Fraction of labels to flip")
    parser.add_argument("--mode", default="random", choices=["random", "targeted"])
    parser.add_argument("--source-class", type=int, default=None)
    parser.add_argument("--target-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--format", default=None, choices=["coco", "yolo"])
    args = parser.parse_args(argv)

    manifest = flip_labels(args.input, args.output, rate=args.rate, mode=args.mode,
                           source_class=args.source_class, target_class=args.target_class,
                           seed=args.seed, fmt=args.format)
    print(json.dumps({"attack": manifest["attack"],
                      "num_flipped": manifest.get("num_flipped", 0),
                      "errors": manifest["errors"][:5]}, indent=2))
    return 0 if manifest.get("num_flipped", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
