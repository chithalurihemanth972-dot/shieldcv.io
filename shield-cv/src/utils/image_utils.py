"""
Image I/O and pixel-statistics helpers for SHIELD-CV.

Everything here is CPU-only, dependency-light and defensive: a corrupt or
unreadable image returns ``None`` (or a neutral statistic) rather than raising,
because a single bad file must never abort an audit of 5,000 samples.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _CV2 = False

try:
    from PIL import Image
    _PIL = True
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    _PIL = False

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm",
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def is_image_file(path: str | Path) -> bool:
    """Return ``True`` when the path has a recognised image extension.

    Args:
        path: Candidate file path.

    Returns:
        Whether the suffix is in :data:`IMAGE_EXTENSIONS`.
    """
    try:
        return Path(path).suffix.lower() in IMAGE_EXTENSIONS
    except Exception:
        return False


def list_images(directory: str | Path, recursive: bool = True,
                limit: Optional[int] = None) -> List[Path]:
    """Enumerate image files under a directory in deterministic order.

    Args:
        directory: Root directory to search.
        recursive: Recurse into sub-directories.
        limit: Optional maximum number of files to return.

    Returns:
        Sorted list of image paths (empty if the directory is missing).
    """
    try:
        root = Path(directory)
        if not root.is_dir():
            LOGGER.warning("list_images: not a directory: %s", root)
            return []
        iterator: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
        files = sorted(p for p in iterator if p.is_file() and is_image_file(p))
        return files[:limit] if limit else files
    except Exception as exc:
        LOGGER.error("list_images failed for %s: %s", directory, exc)
        return []


def load_image(path: str | Path, size: Optional[Tuple[int, int]] = None,
               grayscale: bool = False) -> Optional[np.ndarray]:
    """Load an image as an RGB (or grayscale) ``uint8`` numpy array.

    Args:
        path: Image file path.
        size: Optional ``(width, height)`` to resize to.
        grayscale: Return a single-channel array instead of RGB.

    Returns:
        ``np.ndarray`` of shape ``(H, W, 3)`` or ``(H, W)``, or ``None`` on failure.
    """
    try:
        arr: Optional[np.ndarray] = None
        if _CV2:
            flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
            raw = cv2.imread(str(path), flag)
            if raw is not None:
                arr = raw if grayscale else cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        if arr is None and _PIL:
            with Image.open(str(path)) as img:
                img = img.convert("L" if grayscale else "RGB")
                arr = np.array(img)
        if arr is None:
            LOGGER.warning("load_image: unreadable %s", path)
            return None
        if size is not None:
            arr = resize_image(arr, size)
        return arr
    except Exception as exc:
        LOGGER.warning("load_image failed for %s: %s", path, exc)
        return None


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize an array to ``(width, height)`` using bilinear interpolation.

    Args:
        image: Input array.
        size: Target ``(width, height)``.

    Returns:
        Resized array (original returned if resizing fails).
    """
    try:
        if _CV2:
            return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        if _PIL:
            return np.array(Image.fromarray(image).resize(size, Image.BILINEAR))
        return image
    except Exception as exc:
        LOGGER.debug("resize_image failed: %s", exc)
        return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB array to single-channel grayscale.

    Args:
        image: ``(H, W, 3)`` or already-2D array.

    Returns:
        2-D grayscale array (``uint8`` or float, matching the input dtype range).
    """
    try:
        if image.ndim == 2:
            return image
        if _CV2:
            return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(
            image.dtype)
    except Exception as exc:
        LOGGER.debug("to_grayscale failed: %s", exc)
        return image if image.ndim == 2 else image[..., 0]


def normalize_imagenet(image: np.ndarray) -> np.ndarray:
    """Scale to ``[0,1]`` and apply ImageNet mean/std normalisation.

    Args:
        image: ``(H, W, 3)`` uint8 or float array.

    Returns:
        ``float32`` array in CHW order suitable for torch tensors.
    """
    try:
        arr = image.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return np.transpose(arr, (2, 0, 1)).astype(np.float32)
    except Exception as exc:
        LOGGER.debug("normalize_imagenet failed: %s", exc)
        return np.zeros((3, 224, 224), dtype=np.float32)


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> Optional[str]:
    """Compute the SHA-256 of a file's raw bytes.

    Args:
        path: File to hash.
        chunk_size: Read block size in bytes.

    Returns:
        Lowercase hex digest, or ``None`` if the file cannot be read.
    """
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(chunk_size), b""):
                digest.update(block)
        return digest.hexdigest()
    except Exception as exc:
        LOGGER.warning("file_sha256 failed for %s: %s", path, exc)
        return None


# --------------------------------------------------------------------------
# Pixel-level statistics used by the drift detector and trigger scanner
# --------------------------------------------------------------------------
def compute_brightness(image: np.ndarray) -> float:
    """Mean pixel intensity (0-255) — proxy for illumination.

    Args:
        image: RGB or grayscale array.

    Returns:
        Mean intensity as float.
    """
    try:
        return float(np.mean(to_grayscale(image).astype(np.float32)))
    except Exception:
        return 0.0


def compute_contrast(image: np.ndarray) -> float:
    """Standard deviation of intensity — proxy for sensor/contrast shift.

    Args:
        image: RGB or grayscale array.

    Returns:
        Intensity standard deviation.
    """
    try:
        return float(np.std(to_grayscale(image).astype(np.float32)))
    except Exception:
        return 0.0


def compute_color_warmth(image: np.ndarray) -> float:
    """Red-to-blue channel ratio — proxy for seasonal/lighting temperature.

    Args:
        image: RGB array (grayscale returns 1.0).

    Returns:
        ``mean(R) / mean(B)``, clipped to ``[0, 10]``.
    """
    try:
        if image.ndim != 3 or image.shape[2] < 3:
            return 1.0
        red = float(np.mean(image[..., 0].astype(np.float32)))
        blue = float(np.mean(image[..., 2].astype(np.float32)))
        return float(np.clip(red / (blue + 1e-6), 0.0, 10.0))
    except Exception:
        return 1.0


def compute_edge_density(image: np.ndarray, low: int = 100, high: int = 200) -> float:
    """Fraction of pixels that are Canny edges — proxy for terrain/clutter.

    Args:
        image: RGB or grayscale array.
        low: Canny lower hysteresis threshold.
        high: Canny upper hysteresis threshold.

    Returns:
        Edge pixel ratio in ``[0, 1]``.
    """
    try:
        gray = to_grayscale(image).astype(np.uint8)
        if _CV2:
            edges = cv2.Canny(gray, low, high)
            return float(np.count_nonzero(edges) / max(edges.size, 1))
        gy, gx = np.gradient(gray.astype(np.float32))
        magnitude = np.hypot(gx, gy)
        return float(np.mean(magnitude > float(low) / 4.0))
    except Exception:
        return 0.0


def compute_sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian — proxy for focus/blur.

    Args:
        image: RGB or grayscale array.

    Returns:
        Laplacian variance (higher = sharper).
    """
    try:
        gray = to_grayscale(image).astype(np.float32)
        if _CV2:
            return float(cv2.Laplacian(gray, cv2.CV_32F).var())
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        pad = np.pad(gray, 1, mode="edge")
        out = sum(
            kernel[i, j] * pad[i:i + gray.shape[0], j:j + gray.shape[1]]
            for i in range(3) for j in range(3)
        )
        return float(np.var(out))
    except Exception:
        return 0.0


PIXEL_PROPERTY_FUNCS = {
    "brightness": compute_brightness,
    "contrast": compute_contrast,
    "color_warmth": compute_color_warmth,
    "edge_density": compute_edge_density,
    "sharpness": compute_sharpness,
}


def compute_pixel_properties(image: np.ndarray,
                             properties: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Compute the five SHIELD-CV pixel-level properties for one image.

    Args:
        image: RGB array.
        properties: Subset of property names; defaults to all five.

    Returns:
        Mapping of property name to float value.
    """
    names = list(properties) if properties else list(PIXEL_PROPERTY_FUNCS.keys())
    out: Dict[str, float] = {}
    for name in names:
        func = PIXEL_PROPERTY_FUNCS.get(name)
        if func is None:
            LOGGER.debug("Unknown pixel property requested: %s", name)
            continue
        try:
            out[name] = float(func(image))
        except Exception as exc:
            LOGGER.debug("Property %s failed: %s", name, exc)
            out[name] = 0.0
    return out


def extract_blocks(image: np.ndarray, block_size: int = 16) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Split an image into non-overlapping square blocks.

    Args:
        image: RGB or grayscale array.
        block_size: Side length in pixels.

    Returns:
        Tuple ``(blocks, coords)`` where ``blocks`` is ``(N, block, block)`` and
        ``coords`` holds each block's top-left ``(row, col)``.
    """
    try:
        gray = to_grayscale(image).astype(np.float32)
        height, width = gray.shape[:2]
        blocks: List[np.ndarray] = []
        coords: List[Tuple[int, int]] = []
        for row in range(0, height - block_size + 1, block_size):
            for col in range(0, width - block_size + 1, block_size):
                blocks.append(gray[row:row + block_size, col:col + block_size])
                coords.append((row, col))
        if not blocks:
            return np.empty((0, block_size, block_size), dtype=np.float32), []
        return np.stack(blocks).astype(np.float32), coords
    except Exception as exc:
        LOGGER.debug("extract_blocks failed: %s", exc)
        return np.empty((0, block_size, block_size), dtype=np.float32), []


def crop_region(image: np.ndarray, box: Sequence[float]) -> Optional[np.ndarray]:
    """Crop a bounding box from an image with clamping to valid bounds.

    Args:
        image: Source array.
        box: ``(x, y, w, h)`` in pixels (COCO convention).

    Returns:
        Cropped array, or ``None`` if the box is degenerate.
    """
    try:
        height, width = image.shape[:2]
        x, y, w, h = (float(v) for v in box[:4])
        x0 = int(max(0, min(width - 1, x)))
        y0 = int(max(0, min(height - 1, y)))
        x1 = int(max(x0 + 1, min(width, x + w)))
        y1 = int(max(y0 + 1, min(height, y + h)))
        crop = image[y0:y1, x0:x1]
        return crop if crop.size else None
    except Exception as exc:
        LOGGER.debug("crop_region failed: %s", exc)
        return None


def save_image(image: np.ndarray, path: str | Path) -> bool:
    """Write an RGB array to disk, creating parent directories.

    Args:
        image: RGB ``uint8`` array.
        path: Destination file path.

    Returns:
        ``True`` on success.
    """
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        arr = np.clip(image, 0, 255).astype(np.uint8)
        if _CV2:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr
            return bool(cv2.imwrite(str(target), bgr))
        if _PIL:
            Image.fromarray(arr).save(str(target))
            return True
        return False
    except Exception as exc:
        LOGGER.warning("save_image failed for %s: %s", path, exc)
        return False


__all__ = [
    "IMAGE_EXTENSIONS", "is_image_file", "list_images", "load_image", "resize_image",
    "to_grayscale", "normalize_imagenet", "file_sha256", "compute_brightness",
    "compute_contrast", "compute_color_warmth", "compute_edge_density", "compute_sharpness",
    "compute_pixel_properties", "PIXEL_PROPERTY_FUNCS", "extract_blocks", "crop_region",
    "save_image",
]
