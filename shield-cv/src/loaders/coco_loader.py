"""
COCO-format dataset loader for SHIELD-CV.

Parses a COCO ``instances_*.json`` annotation file into the framework's
normalised :class:`~src.loaders.yolo_loader.DatasetSample` view. Contributor /
source / batch metadata is harvested from any of several conventional locations
so multi-vendor provenance survives ingestion.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.utils.image_utils import is_image_file
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    # Imported for type checking only: yolo_loader owns the normalised Dataset
    # representation and importing it at runtime would create a cycle.
    from src.loaders.yolo_loader import Dataset


CONTRIBUTOR_KEYS: Tuple[str, ...] = (
    "contributor", "source", "batch", "vendor", "provider", "origin",
    "supplier", "unit", "agency",
)


def _extract_contributor(*objects: Optional[Dict[str, Any]]) -> Optional[str]:
    """Search several dicts for a contributor-like metadata field.

    Args:
        *objects: Candidate mappings (image record, dataset info, ...).

    Returns:
        The first contributor string found, else ``None``.
    """
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        for key in CONTRIBUTOR_KEYS:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        meta = obj.get("metadata")
        if isinstance(meta, dict):
            found = _extract_contributor(meta)
            if found:
                return found
    return None


def find_coco_annotation_file(root: str | Path) -> Optional[Path]:
    """Locate a plausible COCO annotation JSON under a dataset root.

    Args:
        root: Dataset directory (or the JSON file itself).

    Returns:
        Path to the annotation file, or ``None`` when nothing matches.
    """
    try:
        path = Path(root)
        if path.is_file() and path.suffix.lower() == ".json":
            return path
        if not path.is_dir():
            return None
        preferred = [
            path / "annotations.json",
            path / "instances.json",
            path / "_annotations.coco.json",
        ]
        for candidate in preferred:
            if candidate.is_file():
                return candidate
        globs = sorted(path.glob("annotations/*.json")) + sorted(path.glob("*.json"))
        for candidate in globs:
            try:
                with candidate.open("r", encoding="utf-8") as handle:
                    head = handle.read(4096)
                if '"annotations"' in head or '"categories"' in head:
                    return candidate
            except Exception:
                continue
        return None
    except Exception as exc:
        LOGGER.error("find_coco_annotation_file failed for %s: %s", root, exc)
        return None


def _resolve_image_dir(annotation_file: Path, explicit: Optional[str | Path]) -> Path:
    """Determine the directory containing the dataset's image files.

    Args:
        annotation_file: Path to the COCO JSON.
        explicit: Caller-supplied image directory override.

    Returns:
        Best-guess image directory (annotation's parent as fallback).
    """
    if explicit:
        candidate = Path(explicit)
        if candidate.is_dir():
            return candidate
    parent = annotation_file.parent
    for name in ("images", "train", "val", "data", "img"):
        candidate = parent / name
        if candidate.is_dir():
            return candidate
    if parent.name == "annotations":
        for name in ("images", "train2017", "val2017"):
            candidate = parent.parent / name
            if candidate.is_dir():
                return candidate
        return parent.parent
    return parent


def load_coco_dataset(root: str | Path,
                      annotation_file: Optional[str | Path] = None,
                      image_dir: Optional[str | Path] = None,
                      max_images: Optional[int] = None) -> "Dataset":
    """Load a COCO dataset into the normalised SHIELD-CV representation.

    Args:
        root: Dataset root directory (or direct path to the annotation JSON).
        annotation_file: Explicit annotation JSON path.
        image_dir: Explicit image directory.
        max_images: Cap on number of samples loaded (memory guard).

    Returns:
        A populated :class:`~src.loaders.yolo_loader.Dataset`. On failure an
        empty dataset with ``errors`` populated is returned rather than raising.
    """
    from src.loaders.yolo_loader import Dataset, DatasetSample  # local import: avoid cycle

    dataset = Dataset(root=Path(root), fmt="coco")
    try:
        ann_path = Path(annotation_file) if annotation_file else find_coco_annotation_file(root)
        if ann_path is None or not ann_path.is_file():
            dataset.errors.append(f"No COCO annotation JSON found under {root}")
            LOGGER.error("COCO annotations not found under %s", root)
            return dataset
        dataset.annotation_file = ann_path

        with ann_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        images = payload.get("images", []) or []
        annotations = payload.get("annotations", []) or []
        categories = payload.get("categories", []) or []
        info = payload.get("info", {}) or {}

        dataset.class_names = {
            int(c["id"]): str(c.get("name", f"class_{c['id']}"))
            for c in categories if "id" in c
        }
        dataset.info = {
            "description": info.get("description", ""),
            "version": info.get("version", ""),
            "num_images_declared": len(images),
            "num_annotations_declared": len(annotations),
            "num_categories": len(categories),
        }

        by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for ann in annotations:
            try:
                by_image[int(ann["image_id"])].append(ann)
            except Exception:
                dataset.errors.append("Annotation missing/invalid image_id; skipped")

        img_root = _resolve_image_dir(ann_path, image_dir)
        dataset.image_dir = img_root
        dataset_contributor = _extract_contributor(info, payload)

        count = 0
        for img in images:
            if max_images is not None and count >= max_images:
                break
            try:
                image_id = int(img["id"])
                file_name = str(img.get("file_name") or img.get("filename") or "")
                if not file_name:
                    dataset.errors.append(f"Image {image_id} has no file_name; skipped")
                    continue

                image_path = (img_root / file_name)
                if not image_path.is_file():
                    alt = next((p for p in img_root.rglob(Path(file_name).name)
                                if p.is_file() and is_image_file(p)), None)
                    if alt is not None:
                        image_path = alt

                anns = by_image.get(image_id, [])
                labels: List[int] = []
                boxes: List[List[float]] = []
                for ann in anns:
                    try:
                        labels.append(int(ann["category_id"]))
                        bbox = ann.get("bbox") or [0, 0, 0, 0]
                        boxes.append([float(v) for v in bbox[:4]])
                    except Exception:
                        dataset.errors.append(
                            f"Malformed annotation on image {image_id}; skipped")

                contributor = (_extract_contributor(img) or dataset_contributor
                               or _infer_contributor_from_path(image_path, Path(root)))

                dataset.samples.append(DatasetSample(
                    sample_id=f"{image_id}",
                    image_path=image_path,
                    labels=labels,
                    boxes=boxes,
                    contributor=contributor,
                    exists=image_path.is_file(),
                    width=int(img.get("width", 0) or 0),
                    height=int(img.get("height", 0) or 0),
                    metadata={k: v for k, v in img.items()
                              if k not in ("id", "file_name", "width", "height")},
                ))
                count += 1
            except Exception as exc:
                dataset.errors.append(f"Failed to parse image record: {exc}")

        missing = sum(1 for s in dataset.samples if not s.exists)
        if missing:
            dataset.errors.append(f"{missing} referenced image file(s) not found on disk")
        LOGGER.info("Loaded COCO dataset: %d samples, %d classes, %d missing files",
                    len(dataset.samples), len(dataset.class_names), missing)
        return dataset
    except json.JSONDecodeError as exc:
        dataset.errors.append(f"Invalid JSON in annotation file: {exc}")
        LOGGER.error("COCO JSON decode error: %s", exc)
        return dataset
    except Exception as exc:
        dataset.errors.append(f"Unexpected COCO load failure: {exc}")
        LOGGER.error("load_coco_dataset failed: %s", exc)
        return dataset


def _infer_contributor_from_path(image_path: Path, root: Path) -> Optional[str]:
    """Infer a contributor name from directory naming conventions.

    Recognises folders such as ``contributor_a``, ``source-B``, ``vendor_x``.

    Args:
        image_path: Path of the image file.
        root: Dataset root, used to bound the upward search.

    Returns:
        Normalised contributor name, or ``None``.
    """
    try:
        prefixes = ("contributor", "source", "vendor", "supplier", "batch", "partner")
        for part in image_path.resolve().parts[::-1]:
            lowered = part.lower()
            if any(lowered.startswith(p) for p in prefixes):
                return part
        return None
    except Exception:
        return None


def write_coco_dataset(dataset: "Dataset", output_path: str | Path) -> bool:
    """Serialise a normalised dataset back to COCO JSON.

    Used by the attack simulators to emit poisoned copies.

    Args:
        dataset: Dataset to serialise.
        output_path: Destination JSON path.

    Returns:
        ``True`` on success.
    """
    try:
        images: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []
        ann_id = 1
        for index, sample in enumerate(dataset.samples, start=1):
            try:
                image_id = int(sample.sample_id)
            except (TypeError, ValueError):
                image_id = index
            record: Dict[str, Any] = {
                "id": image_id,
                "file_name": sample.image_path.name,
                "width": sample.width,
                "height": sample.height,
            }
            if sample.contributor:
                record["contributor"] = sample.contributor
            record.update(sample.metadata or {})
            images.append(record)

            for label, box in zip(sample.labels, sample.boxes or [[0, 0, 0, 0]] * len(sample.labels)):
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(v) for v in box[:4]],
                    "area": float(box[2] * box[3]) if len(box) >= 4 else 0.0,
                    "iscrowd": 0,
                })
                ann_id += 1

        payload = {
            "info": {"description": "SHIELD-CV generated COCO dataset", "version": "1.0"},
            "images": images,
            "annotations": annotations,
            "categories": [{"id": cid, "name": name}
                           for cid, name in sorted(dataset.class_names.items())],
        }
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        LOGGER.info("Wrote COCO dataset to %s (%d images)", target, len(images))
        return True
    except Exception as exc:
        LOGGER.error("write_coco_dataset failed: %s", exc)
        return False


__all__ = [
    "load_coco_dataset", "write_coco_dataset", "find_coco_annotation_file",
    "CONTRIBUTOR_KEYS",
]
