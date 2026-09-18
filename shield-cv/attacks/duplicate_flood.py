"""
ATTACK SIMULATION — Near-duplicate flooding.

Copies N source images many times with slight augmentation (brightness, small
rotation, mild noise, JPEG-style blur) so byte hashes all differ but the images
remain perceptually identical.

Operationally this is how a dishonest vendor inflates a paid delivery ("10,000
images" that are really 800), and how an attacker cheaply amplifies a poisoned
sample's influence on training without submitting obviously identical files.

Deterministic under a fixed seed; ground truth written to ``ground_truth.json``.

Usage:
    python attacks/duplicate_flood.py --input demo/data/clean \\
        --output demo/data/poisoned --sources 3 --copies 8
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
from src.utils.image_utils import load_image, save_image  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

DEFAULT_SEED = 1337


def augment_image(image: np.ndarray, rng: np.random.Generator,
                  strength: float = 1.0) -> Dict[str, Any]:
    """Apply a mild, perceptually-invisible augmentation to an image.

    The augmentations are deliberately weak: strong enough to change every byte
    (and thus defeat SHA-256 deduplication), weak enough that pHash distance
    stays under the near-duplicate threshold.

    Args:
        image: RGB ``uint8`` array.
        rng: Seeded random generator.
        strength: Multiplier on augmentation magnitude.

    Returns:
        Dictionary with the augmented ``image`` and the ``ops`` applied.
    """
    try:
        output = np.array(image, copy=True).astype(np.float32)
        operations: List[str] = []

        brightness = float(rng.uniform(-8, 8) * strength)
        if abs(brightness) > 0.5:
            output = output + brightness
            operations.append(f"brightness{brightness:+.1f}")

        contrast = float(rng.uniform(0.97, 1.03))
        if abs(contrast - 1.0) > 0.005:
            mean = float(output.mean())
            output = (output - mean) * contrast + mean
            operations.append(f"contrast x{contrast:.3f}")

        noise_sigma = float(rng.uniform(0.5, 2.0) * strength)
        output = output + rng.normal(0.0, noise_sigma, size=output.shape)
        operations.append(f"noise σ={noise_sigma:.2f}")

        angle = float(rng.uniform(-1.5, 1.5) * strength)
        if abs(angle) > 0.2:
            try:
                import cv2
                height, width = output.shape[:2]
                matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
                output = cv2.warpAffine(output, matrix, (width, height),
                                        borderMode=cv2.BORDER_REFLECT)
                operations.append(f"rotate{angle:+.2f}°")
            except Exception as exc:
                LOGGER.debug("rotation skipped: %s", exc)

        if rng.random() < 0.3:
            try:
                import cv2
                output = cv2.GaussianBlur(output, (3, 3), 0.4)
                operations.append("blur3x3")
            except Exception:
                pass

        return {"image": np.clip(output, 0, 255).astype(np.uint8), "ops": operations}
    except Exception as exc:
        LOGGER.error("augment_image failed: %s", exc)
        return {"image": image, "ops": []}


def flood_duplicates(input_dir: str | Path,
                     output_dir: str | Path,
                     sources: int = 3,
                     copies: int = 8,
                     strength: float = 1.0,
                     seed: int = DEFAULT_SEED,
                     fmt: Optional[str] = None) -> Dict[str, Any]:
    """Flood a dataset with augmented near-duplicates of a few source images.

    Args:
        input_dir: Clean dataset root.
        output_dir: Destination for the flooded copy.
        sources: How many distinct source images to replicate.
        copies: How many near-duplicates to create per source.
        strength: Augmentation strength multiplier.
        seed: Random seed.
        fmt: Dataset format override.

    Returns:
        Ground-truth manifest listing every generated duplicate and its source.
    """
    manifest: Dict[str, Any] = {
        "attack": "NEAR_DUPLICATE_FLOODING",
        "parameters": {"sources": sources, "copies": copies, "strength": strength,
                       "seed": seed},
        "clusters": [],
        "errors": [],
    }
    try:
        source_root = Path(input_dir)
        destination = Path(output_dir)
        if not source_root.is_dir():
            manifest["errors"].append(f"Input directory not found: {source_root}")
            return manifest

        if destination.exists() and destination.resolve() != source_root.resolve():
            shutil.rmtree(destination)
        if destination.resolve() != source_root.resolve():
            shutil.copytree(source_root, destination)

        dataset = load_dataset(destination, fmt=fmt)
        samples = dataset.valid_samples
        if not samples:
            manifest["errors"].append("No readable images in dataset")
            return manifest

        rng = np.random.default_rng(int(seed))
        chosen = rng.choice(len(samples), size=int(min(sources, len(samples))), replace=False)

        total_created = 0
        coco_additions: List[Dict[str, Any]] = []

        for source_index in sorted(int(i) for i in chosen):
            sample = samples[source_index]
            image = load_image(sample.image_path)
            if image is None:
                manifest["errors"].append(f"Unreadable: {sample.image_path}")
                continue

            cluster: Dict[str, Any] = {
                "source_file": sample.image_path.name,
                "label": sample.primary_label,
                "duplicates": [],
            }

            for copy_index in range(int(copies)):
                augmented = augment_image(image, rng, strength=strength)
                new_name = f"{sample.image_path.stem}_dup{copy_index:03d}{sample.image_path.suffix}"
                new_path = sample.image_path.parent / new_name
                if not save_image(augmented["image"], new_path):
                    manifest["errors"].append(f"Could not write {new_path}")
                    continue

                label_file = sample.metadata.get("label_file")
                if label_file:
                    label_source = Path(label_file)
                    new_label_path = label_source.parent / f"{new_path.stem}.txt"
                    try:
                        shutil.copyfile(label_source, new_label_path)
                    except Exception:
                        write_yolo_labels(sample, new_label_path)
                else:
                    coco_additions.append({
                        "file_name": new_name,
                        "width": sample.width,
                        "height": sample.height,
                        "labels": list(sample.labels),
                        "boxes": [list(b) for b in sample.boxes],
                        "contributor": sample.contributor,
                    })

                cluster["duplicates"].append({"file": new_name, "ops": augmented["ops"]})
                total_created += 1

            manifest["clusters"].append(cluster)

        if coco_additions and dataset.annotation_file:
            _extend_coco(dataset.annotation_file, coco_additions, manifest)

        manifest["num_duplicates_created"] = total_created
        manifest["num_clusters"] = len(manifest["clusters"])
        _write_manifest(destination, manifest, "duplicate_flood")
        LOGGER.info("Duplicate flooding: created %d near-duplicate(s) across %d cluster(s)",
                    total_created, len(manifest["clusters"]))
        return manifest
    except Exception as exc:
        LOGGER.error("flood_duplicates failed: %s", exc)
        manifest["errors"].append(str(exc))
        return manifest


def _extend_coco(annotation_file: Path, additions: List[Dict[str, Any]],
                 manifest: Dict[str, Any]) -> None:
    """Append duplicated images and their annotations to a COCO file.

    Args:
        annotation_file: COCO JSON path.
        additions: New image records with labels and boxes.
        manifest: Manifest to record errors in.
    """
    try:
        path = Path(annotation_file)
        payload = json.loads(path.read_text(encoding="utf-8"))
        next_image_id = max((img.get("id", 0) for img in payload.get("images", [])),
                            default=0) + 1
        next_annotation_id = max((a.get("id", 0) for a in payload.get("annotations", [])),
                                 default=0) + 1

        for item in additions:
            record: Dict[str, Any] = {
                "id": next_image_id,
                "file_name": item["file_name"],
                "width": item.get("width", 0),
                "height": item.get("height", 0),
            }
            if item.get("contributor"):
                record["contributor"] = item["contributor"]
            payload.setdefault("images", []).append(record)

            boxes = item.get("boxes") or [[0, 0, item.get("width", 0), item.get("height", 0)]]
            for position, label in enumerate(item.get("labels", [])):
                box = boxes[position] if position < len(boxes) else boxes[0]
                payload.setdefault("annotations", []).append({
                    "id": next_annotation_id,
                    "image_id": next_image_id,
                    "category_id": int(label),
                    "bbox": [float(v) for v in box[:4]],
                    "area": float(box[2] * box[3]) if len(box) >= 4 else 0.0,
                    "iscrowd": 0,
                })
                next_annotation_id += 1
            next_image_id += 1

        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        LOGGER.error("_extend_coco failed: %s", exc)
        manifest["errors"].append(f"COCO extend failed: {exc}")


def _write_manifest(destination: Path, manifest: Dict[str, Any], name: str) -> None:
    """Write the ground-truth manifest for this attack.

    Args:
        destination: Flooded dataset root.
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
    """Command-line entry point for the duplicate flooding attack.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="SHIELD-CV attack: near-duplicate flooding")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sources", type=int, default=3, help="Distinct images to replicate")
    parser.add_argument("--copies", type=int, default=8, help="Duplicates per source image")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--format", default=None, choices=["coco", "yolo"])
    args = parser.parse_args(argv)

    manifest = flood_duplicates(args.input, args.output, sources=args.sources,
                                copies=args.copies, strength=args.strength,
                                seed=args.seed, fmt=args.format)
    print(json.dumps({"attack": manifest["attack"],
                      "num_duplicates_created": manifest.get("num_duplicates_created", 0),
                      "num_clusters": manifest.get("num_clusters", 0),
                      "errors": manifest["errors"][:5]}, indent=2))
    return 0 if manifest.get("num_duplicates_created", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
