"""
YOLO-format dataset loader and the canonical SHIELD-CV dataset representation.

Defines :class:`DatasetSample` and :class:`Dataset` — the normalised structures
every scanner consumes — plus :func:`load_yolo_dataset` for ``images/``+``labels/``
layouts with ``class_id cx cy w h`` normalised text annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.image_utils import IMAGE_EXTENSIONS, list_images
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class DatasetSample:
    """A single image plus its annotations and provenance metadata.

    Attributes:
        sample_id: Stable identifier (COCO image id or file stem).
        image_path: Absolute/relative path to the image file.
        labels: Integer class ids present in the image.
        boxes: Bounding boxes, ``[x, y, w, h]`` in pixels (COCO convention).
        contributor: Originating contributor/source/vendor, if known.
        exists: Whether the image file was found on disk.
        width: Declared image width in pixels (0 if unknown).
        height: Declared image height in pixels (0 if unknown).
        metadata: Any residual format-specific metadata.
    """

    sample_id: str
    image_path: Path
    labels: List[int] = field(default_factory=list)
    boxes: List[List[float]] = field(default_factory=list)
    contributor: Optional[str] = None
    exists: bool = True
    width: int = 0
    height: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def primary_label(self) -> int:
        """Return the dominant class id for single-label analyses.

        Returns:
            The most frequent label, or ``-1`` when unlabelled.
        """
        try:
            if not self.labels:
                return -1
            return max(set(self.labels), key=self.labels.count)
        except Exception:
            return -1

    @property
    def name(self) -> str:
        """Return the image file name for human-readable reporting."""
        try:
            return self.image_path.name
        except Exception:
            return str(self.sample_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the sample to a JSON-safe dictionary.

        Returns:
            Dictionary representation with the path rendered as a string.
        """
        return {
            "sample_id": self.sample_id,
            "image_path": str(self.image_path),
            "labels": list(self.labels),
            "boxes": [list(b) for b in self.boxes],
            "contributor": self.contributor,
            "exists": self.exists,
            "width": self.width,
            "height": self.height,
            "primary_label": self.primary_label,
        }


@dataclass
class Dataset:
    """A normalised, format-agnostic dataset view.

    Attributes:
        root: Dataset root directory.
        fmt: Source format, ``"coco"`` or ``"yolo"``.
        samples: Loaded samples.
        class_names: Mapping of class id to human-readable name.
        annotation_file: Source annotation file, if any.
        image_dir: Directory that held the images.
        info: Free-form dataset-level metadata.
        errors: Non-fatal problems encountered during loading.
    """

    root: Path
    fmt: str = "unknown"
    samples: List[DatasetSample] = field(default_factory=list)
    class_names: Dict[int, str] = field(default_factory=dict)
    annotation_file: Optional[Path] = None
    image_dir: Optional[Path] = None
    info: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of loaded samples."""
        return len(self.samples)

    @property
    def valid_samples(self) -> List[DatasetSample]:
        """Return only samples whose image file exists on disk."""
        return [s for s in self.samples if s.exists]

    def class_name(self, class_id: int) -> str:
        """Resolve a class id to its name.

        Args:
            class_id: Integer class identifier.

        Returns:
            Human-readable class name, or ``class_<id>``.
        """
        return self.class_names.get(int(class_id), f"class_{class_id}")

    def contributors(self) -> List[str]:
        """List distinct contributors present in the dataset.

        Returns:
            Sorted list of contributor names (``"UNKNOWN"`` folded in when absent).
        """
        try:
            return sorted({s.contributor or "UNKNOWN" for s in self.samples})
        except Exception:
            return []

    def samples_by_class(self) -> Dict[int, List[DatasetSample]]:
        """Group samples by their primary label.

        Returns:
            Mapping of class id to the samples carrying it.
        """
        grouped: Dict[int, List[DatasetSample]] = {}
        for sample in self.samples:
            grouped.setdefault(sample.primary_label, []).append(sample)
        return grouped

    def summary(self) -> Dict[str, Any]:
        """Produce a compact dataset summary for reports.

        Returns:
            Dictionary of counts, class distribution and contributor list.
        """
        try:
            distribution: Dict[str, int] = {}
            for sample in self.samples:
                key = self.class_name(sample.primary_label)
                distribution[key] = distribution.get(key, 0) + 1
            return {
                "root": str(self.root),
                "format": self.fmt,
                "num_samples": len(self.samples),
                "num_valid_images": len(self.valid_samples),
                "num_classes": len(self.class_names),
                "class_distribution": distribution,
                "contributors": self.contributors(),
                "annotation_file": str(self.annotation_file) if self.annotation_file else None,
                "errors": list(self.errors[:20]),
            }
        except Exception as exc:
            LOGGER.error("Dataset.summary failed: %s", exc)
            return {"root": str(self.root), "format": self.fmt, "error": str(exc)}


def _read_class_names(root: Path) -> Dict[int, str]:
    """Read YOLO class names from ``classes.txt``, ``obj.names`` or ``data.yaml``.

    Args:
        root: Dataset root directory.

    Returns:
        Mapping of class index to name (possibly empty).
    """
    names: Dict[int, str] = {}
    try:
        for filename in ("classes.txt", "obj.names", "names.txt"):
            candidate = root / filename
            if candidate.is_file():
                lines = [l.strip() for l in candidate.read_text(encoding="utf-8").splitlines()]
                return {i: n for i, n in enumerate(lines) if n}

        for filename in ("data.yaml", "dataset.yaml", "data.yml"):
            candidate = root / filename
            if candidate.is_file():
                try:
                    import yaml
                    payload = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
                    raw = payload.get("names")
                    if isinstance(raw, list):
                        return {i: str(n) for i, n in enumerate(raw)}
                    if isinstance(raw, dict):
                        return {int(k): str(v) for k, v in raw.items()}
                except Exception as exc:
                    LOGGER.debug("Could not parse %s: %s", candidate, exc)
        return names
    except Exception as exc:
        LOGGER.debug("_read_class_names failed: %s", exc)
        return names


def _find_label_path(image_path: Path, root: Path) -> Optional[Path]:
    """Locate the YOLO ``.txt`` label file matching an image.

    Args:
        image_path: Image file path.
        root: Dataset root (searched as a fallback).

    Returns:
        Path to the label file, or ``None``.
    """
    try:
        direct = image_path.with_suffix(".txt")
        if direct.is_file():
            return direct
        parts = list(image_path.parts)
        for index in range(len(parts) - 1, -1, -1):
            if parts[index] == "images":
                swapped = Path(*parts[:index], "labels", *parts[index + 1:]).with_suffix(".txt")
                if swapped.is_file():
                    return swapped
                break
        for base in (root / "labels", root.parent / "labels"):
            candidate = base / (image_path.stem + ".txt")
            if candidate.is_file():
                return candidate
        return None
    except Exception:
        return None


def _parse_label_file(path: Path, width: int, height: int) -> Tuple[List[int], List[List[float]], List[str]]:
    """Parse a YOLO label file into class ids and pixel bounding boxes.

    Args:
        path: Label ``.txt`` file.
        width: Image width used to denormalise boxes (0 keeps normalised units).
        height: Image height used to denormalise boxes.

    Returns:
        Tuple ``(labels, boxes_xywh_pixels, errors)``.
    """
    labels: List[int] = []
    boxes: List[List[float]] = []
    errors: List[str] = []
    try:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            try:
                class_id = int(float(parts[0]))
                labels.append(class_id)
                if len(parts) >= 5:
                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                    scale_w = width if width > 0 else 1.0
                    scale_h = height if height > 0 else 1.0
                    boxes.append([
                        (cx - bw / 2.0) * scale_w,
                        (cy - bh / 2.0) * scale_h,
                        bw * scale_w,
                        bh * scale_h,
                    ])
                else:
                    boxes.append([0.0, 0.0, 0.0, 0.0])
            except Exception:
                errors.append(f"{path.name}:{line_no} malformed label line")
        return labels, boxes, errors
    except Exception as exc:
        errors.append(f"Could not read {path}: {exc}")
        return labels, boxes, errors


def detect_format(root: str | Path) -> str:
    """Heuristically determine whether a directory holds a COCO or YOLO dataset.

    Args:
        root: Dataset root directory or annotation file.

    Returns:
        ``"coco"``, ``"yolo"`` or ``"unknown"``.
    """
    try:
        path = Path(root)
        if path.is_file() and path.suffix.lower() == ".json":
            return "coco"
        if not path.is_dir():
            return "unknown"
        from src.loaders.coco_loader import find_coco_annotation_file
        if find_coco_annotation_file(path) is not None:
            return "coco"
        if any(path.rglob("*.txt")):
            for filename in ("classes.txt", "data.yaml", "obj.names"):
                if (path / filename).is_file():
                    return "yolo"
            if (path / "labels").is_dir() or any(path.rglob("labels")):
                return "yolo"
            images = list_images(path, limit=5)
            if images and any(_find_label_path(img, path) for img in images):
                return "yolo"
        return "unknown"
    except Exception as exc:
        LOGGER.debug("detect_format failed: %s", exc)
        return "unknown"


def load_yolo_dataset(root: str | Path,
                      max_images: Optional[int] = None,
                      read_dimensions: bool = True) -> Dataset:
    """Load a YOLO-format dataset into the normalised representation.

    Args:
        root: Dataset root containing images (and ``labels/`` or sibling ``.txt``).
        max_images: Cap on samples loaded.
        read_dimensions: Read real image sizes (needed to denormalise boxes).

    Returns:
        Populated :class:`Dataset`; errors are collected, never raised.
    """
    dataset = Dataset(root=Path(root), fmt="yolo")
    try:
        base = Path(root)
        if not base.is_dir():
            dataset.errors.append(f"Dataset root is not a directory: {base}")
            return dataset

        dataset.class_names = _read_class_names(base)
        images = list_images(base, recursive=True, limit=max_images)
        if not images:
            dataset.errors.append(f"No image files found under {base}")
            LOGGER.warning("No images found under %s", base)
            return dataset
        dataset.image_dir = base

        for image_path in images:
            try:
                width = height = 0
                if read_dimensions:
                    width, height = read_image_size(image_path)

                label_path = _find_label_path(image_path, base)
                labels: List[int] = []
                boxes: List[List[float]] = []
                if label_path is not None:
                    labels, boxes, errs = _parse_label_file(label_path, width, height)
                    dataset.errors.extend(errs)
                else:
                    inferred = _label_from_folder(image_path, dataset)
                    if inferred is not None:
                        labels = [inferred]
                        boxes = [[0.0, 0.0, float(width), float(height)]]

                dataset.samples.append(DatasetSample(
                    sample_id=image_path.stem,
                    image_path=image_path,
                    labels=labels,
                    boxes=boxes,
                    contributor=_infer_contributor(image_path),
                    exists=True,
                    width=width,
                    height=height,
                    metadata={"label_file": str(label_path) if label_path else None},
                ))
            except Exception as exc:
                dataset.errors.append(f"Failed to load {image_path.name}: {exc}")

        if not dataset.class_names:
            observed = sorted({l for s in dataset.samples for l in s.labels})
            dataset.class_names = {cid: f"class_{cid}" for cid in observed}

        dataset.info = {
            "num_images_found": len(images),
            "labels_present": sum(1 for s in dataset.samples if s.labels),
        }
        LOGGER.info("Loaded YOLO dataset: %d samples, %d classes",
                    len(dataset.samples), len(dataset.class_names))
        return dataset
    except Exception as exc:
        dataset.errors.append(f"Unexpected YOLO load failure: {exc}")
        LOGGER.error("load_yolo_dataset failed: %s", exc)
        return dataset


def _label_from_folder(image_path: Path, dataset: Dataset) -> Optional[int]:
    """Derive a class id from an ImageFolder-style parent directory name.

    Args:
        image_path: Image path whose parent may be a class folder.
        dataset: Dataset whose ``class_names`` map is extended in place.

    Returns:
        Class id, or ``None`` when the layout is not class-foldered.
    """
    try:
        folder = image_path.parent.name
        if folder.lower() in ("images", "train", "val", "test", "data", ""):
            return None
        for cid, name in dataset.class_names.items():
            if name == folder:
                return cid
        new_id = max(dataset.class_names.keys(), default=-1) + 1
        dataset.class_names[new_id] = folder
        return new_id
    except Exception:
        return None


def _infer_contributor(image_path: Path) -> Optional[str]:
    """Infer a contributor from conventionally-named ancestor directories.

    Args:
        image_path: Image file path.

    Returns:
        Contributor name, or ``None``.
    """
    try:
        prefixes = ("contributor", "source", "vendor", "supplier", "batch", "partner")
        for part in image_path.resolve().parts[::-1]:
            if any(part.lower().startswith(p) for p in prefixes):
                return part
        return None
    except Exception:
        return None


def read_image_size(path: Path) -> Tuple[int, int]:
    """Read an image's pixel dimensions without decoding the full raster.

    Args:
        path: Image file path.

    Returns:
        ``(width, height)``; ``(0, 0)`` when unreadable.
    """
    try:
        from PIL import Image
        with Image.open(str(path)) as img:
            return int(img.width), int(img.height)
    except Exception:
        try:
            import cv2
            arr = cv2.imread(str(path))
            if arr is not None:
                return int(arr.shape[1]), int(arr.shape[0])
        except Exception:
            pass
        return 0, 0


def load_dataset(root: str | Path, fmt: Optional[str] = None,
                 max_images: Optional[int] = None) -> Dataset:
    """Load a dataset, auto-detecting the format when not specified.

    Args:
        root: Dataset root directory or COCO JSON path.
        fmt: ``"coco"``, ``"yolo"`` or ``None`` for auto-detection.
        max_images: Cap on samples loaded.

    Returns:
        Populated :class:`Dataset`.
    """
    try:
        chosen = (fmt or detect_format(root)).lower()
        if chosen == "coco":
            from src.loaders.coco_loader import load_coco_dataset
            return load_coco_dataset(root, max_images=max_images)
        if chosen == "yolo":
            return load_yolo_dataset(root, max_images=max_images)

        LOGGER.warning("Unknown dataset format at %s; falling back to image-only scan", root)
        dataset = load_yolo_dataset(root, max_images=max_images)
        dataset.fmt = "images-only"
        dataset.errors.append(
            "Format auto-detection failed; loaded as unlabelled image directory")
        return dataset
    except Exception as exc:
        LOGGER.error("load_dataset failed: %s", exc)
        failed = Dataset(root=Path(root), fmt="error")
        failed.errors.append(str(exc))
        return failed


def write_yolo_labels(sample: DatasetSample, label_path: str | Path) -> bool:
    """Write a sample's annotations back out in YOLO normalised format.

    Args:
        sample: Sample whose labels/boxes are serialised.
        label_path: Destination ``.txt`` path.

    Returns:
        ``True`` on success.
    """
    try:
        target = Path(label_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        width = sample.width if sample.width > 0 else 1
        height = sample.height if sample.height > 0 else 1
        lines: List[str] = []
        for index, label in enumerate(sample.labels):
            if index < len(sample.boxes):
                x, y, w, h = sample.boxes[index][:4]
                cx = (x + w / 2.0) / width
                cy = (y + h / 2.0) / height
                nw, nh = w / width, h / height
            else:
                cx = cy = 0.5
                nw = nh = 1.0
            lines.append(f"{int(label)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return True
    except Exception as exc:
        LOGGER.error("write_yolo_labels failed for %s: %s", label_path, exc)
        return False


__all__ = [
    "Dataset", "DatasetSample", "load_yolo_dataset", "load_dataset", "detect_format",
    "read_image_size", "write_yolo_labels", "IMAGE_EXTENSIONS",
]
