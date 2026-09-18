"""
Generate a deterministic synthetic demo dataset for SHIELD-CV.

Creates a small multi-contributor "aerial reconnaissance" dataset with three
classes rendered as procedurally distinct scenes (no external downloads — this
must work air-gapped). Three contributors supply the data, which lets the
contributor-risk aggregation and the multi-agent office demo do real work.

Layout produced::

    demo/data/clean/
        contributor_alpha/images/*.jpg      contributor_alpha/labels/*.txt
        contributor_bravo/images/*.jpg      contributor_bravo/labels/*.txt
        contributor_charlie/images/*.jpg    contributor_charlie/labels/*.txt
        classes.txt

Usage:
    python demo/generate_demo_data.py --output demo/data/clean --per-class 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.image_utils import save_image  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

CLASS_NAMES = ["armoured_vehicle", "supply_truck", "personnel"]
CONTRIBUTORS = ["contributor_alpha", "contributor_bravo", "contributor_charlie"]
IMAGE_SIZE = 128


def _terrain_background(rng: np.random.Generator, size: int, warmth: float) -> np.ndarray:
    """Render a procedural terrain background.

    Args:
        rng: Seeded random generator.
        size: Square image side length.
        warmth: Colour-temperature multiplier (>1 = arid/warm, <1 = cold).

    Returns:
        ``(size, size, 3)`` uint8 RGB array.
    """
    try:
        coarse = rng.normal(0.0, 1.0, size=(size // 8, size // 8))
        upscaled = np.kron(coarse, np.ones((8, 8)))[:size, :size]
        medium = rng.normal(0.0, 0.5, size=(size // 2, size // 2))
        upscaled = upscaled + np.kron(medium, np.ones((2, 2)))[:size, :size]
        field = (upscaled - upscaled.min()) / max(upscaled.ptp(), 1e-6)

        base = 70.0 + 90.0 * field
        image = np.zeros((size, size, 3), dtype=np.float32)
        image[..., 0] = base * (0.95 * warmth)
        image[..., 1] = base * 0.92
        image[..., 2] = base * (0.80 / max(warmth, 0.5))
        image += rng.normal(0.0, 4.0, size=image.shape)
        return np.clip(image, 0, 255).astype(np.uint8)
    except Exception as exc:
        LOGGER.error("_terrain_background failed: %s", exc)
        return np.full((size, size, 3), 128, dtype=np.uint8)


def _draw_object(image: np.ndarray, class_id: int,
                 rng: np.random.Generator) -> Tuple[np.ndarray, List[float]]:
    """Draw a class-distinctive object onto a background.

    Each class gets a different shape/aspect/intensity signature so a frozen
    ResNet-18 produces genuinely separable embeddings.

    Args:
        image: Background array (modified on a copy).
        class_id: Class index.
        rng: Seeded random generator.

    Returns:
        Tuple ``(image, bbox_xywh)``.
    """
    try:
        output = np.array(image, copy=True)
        height, width = output.shape[:2]

        if class_id == 0:      # armoured vehicle: wide, dark, blocky + turret
            box_w, box_h = rng.integers(34, 46), rng.integers(20, 26)
            intensity = rng.integers(35, 65)
        elif class_id == 1:    # supply truck: tall cargo box, lighter
            box_w, box_h = rng.integers(22, 30), rng.integers(30, 40)
            intensity = rng.integers(120, 160)
        else:                  # personnel: small, bright, near-square
            box_w, box_h = rng.integers(10, 16), rng.integers(14, 20)
            intensity = rng.integers(180, 225)

        x = int(rng.integers(6, max(7, width - box_w - 6)))
        y = int(rng.integers(6, max(7, height - box_h - 6)))
        colour = np.array([intensity, intensity * 0.96, intensity * 0.9], dtype=np.float32)
        output[y:y + box_h, x:x + box_w] = np.clip(
            colour + rng.normal(0, 6, size=(box_h, box_w, 3)), 0, 255).astype(np.uint8)

        if class_id == 0:
            turret_w, turret_h = box_w // 3, box_h // 2
            tx = x + box_w // 3
            ty = max(0, y - turret_h // 2)
            output[ty:ty + turret_h, tx:tx + turret_w] = np.clip(
                colour * 0.7, 0, 255).astype(np.uint8)
            wheel_y = min(height - 3, y + box_h - 2)
            for offset in range(2, box_w - 2, 7):
                output[wheel_y:wheel_y + 3, x + offset:x + offset + 4] = 25
        elif class_id == 1:
            cab_h = box_h // 3
            output[y:y + cab_h, x:x + box_w] = np.clip(
                colour * 0.55, 0, 255).astype(np.uint8)
        else:
            head = max(3, box_w // 2)
            hx = x + (box_w - head) // 2
            hy = max(0, y - head)
            output[hy:hy + head, hx:hx + head] = np.clip(
                colour * 0.85, 0, 255).astype(np.uint8)

        return output, [float(x), float(y), float(box_w), float(box_h)]
    except Exception as exc:
        LOGGER.error("_draw_object failed: %s", exc)
        return image, [0.0, 0.0, 1.0, 1.0]


def generate(output_dir: str | Path, per_class: int = 30, size: int = IMAGE_SIZE,
             seed: int = 1337, fmt: str = "yolo") -> Dict[str, Any]:
    """Generate the clean multi-contributor demo dataset.

    Args:
        output_dir: Destination directory.
        per_class: Images per class per contributor.
        size: Square image resolution.
        seed: Random seed for full reproducibility.
        fmt: ``"yolo"`` or ``"coco"``.

    Returns:
        Manifest describing what was generated.
    """
    manifest: Dict[str, Any] = {
        "classes": CLASS_NAMES, "contributors": CONTRIBUTORS, "format": fmt,
        "per_class_per_contributor": per_class, "size": size, "seed": seed,
        "num_images": 0, "errors": [],
    }
    try:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(int(seed))

        # Each contributor uses a slightly different sensor/season profile —
        # realistic, and benign (the drift detector should call this natural).
        warmth_profiles = {"contributor_alpha": 1.00,
                           "contributor_bravo": 1.06,
                           "contributor_charlie": 0.95}

        coco_images: List[Dict[str, Any]] = []
        coco_annotations: List[Dict[str, Any]] = []
        image_id = 1
        annotation_id = 1
        total = 0

        for contributor in CONTRIBUTORS:
            image_dir = root / contributor / "images"
            label_dir = root / contributor / "labels"
            image_dir.mkdir(parents=True, exist_ok=True)
            if fmt == "yolo":
                label_dir.mkdir(parents=True, exist_ok=True)

            for class_id in range(len(CLASS_NAMES)):
                for index in range(int(per_class)):
                    background = _terrain_background(
                        rng, size, warmth_profiles.get(contributor, 1.0))
                    image, bbox = _draw_object(background, class_id, rng)
                    name = f"{contributor.split('_')[1]}_{CLASS_NAMES[class_id]}_{index:03d}.jpg"
                    image_path = image_dir / name

                    if not save_image(image, image_path):
                        manifest["errors"].append(f"Failed to write {image_path}")
                        continue

                    if fmt == "yolo":
                        cx = (bbox[0] + bbox[2] / 2) / size
                        cy = (bbox[1] + bbox[3] / 2) / size
                        nw, nh = bbox[2] / size, bbox[3] / size
                        (label_dir / f"{image_path.stem}.txt").write_text(
                            f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n",
                            encoding="utf-8")
                    else:
                        coco_images.append({
                            "id": image_id, "file_name": name, "width": size,
                            "height": size, "contributor": contributor,
                        })
                        coco_annotations.append({
                            "id": annotation_id, "image_id": image_id,
                            "category_id": class_id, "bbox": bbox,
                            "area": bbox[2] * bbox[3], "iscrowd": 0,
                        })
                        image_id += 1
                        annotation_id += 1
                    total += 1

        (root / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

        if fmt == "coco":
            (root / "annotations.json").write_text(json.dumps({
                "info": {"description": "SHIELD-CV synthetic demo dataset", "version": "1.0"},
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)],
            }, indent=2), encoding="utf-8")

        manifest["num_images"] = total
        LOGGER.info("Generated %d demo images across %d contributor(s) at %s",
                    total, len(CONTRIBUTORS), root)
        return manifest
    except Exception as exc:
        LOGGER.error("generate failed: %s", exc)
        manifest["errors"].append(str(exc))
        return manifest


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point for demo data generation.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Generate SHIELD-CV demo dataset")
    parser.add_argument("--output", default="demo/data/clean")
    parser.add_argument("--per-class", type=int, default=30,
                        help="Images per class per contributor")
    parser.add_argument("--size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--format", default="yolo", choices=["yolo", "coco"])
    args = parser.parse_args(argv)

    manifest = generate(args.output, per_class=args.per_class, size=args.size,
                        seed=args.seed, fmt=args.format)
    print(json.dumps({"num_images": manifest["num_images"],
                      "classes": manifest["classes"],
                      "contributors": manifest["contributors"],
                      "errors": manifest["errors"][:5]}, indent=2))
    return 0 if manifest["num_images"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
