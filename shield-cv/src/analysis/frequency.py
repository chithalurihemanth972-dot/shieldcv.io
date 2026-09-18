"""
Frequency-domain and spatial-block analysis for trigger detection.

Patch triggers (BadNets-style squares, chequerboards, stamped logos) are
high-contrast, hard-edged and spatially local. They therefore leave two
signatures that survive JPEG compression:

* a raised **high-frequency energy ratio** in the 2-D FFT magnitude spectrum;
* one or two **16x16 blocks** whose local variance/gradient is wildly out of
  family with the rest of the image.

Both are computed per image and compared against the dataset population.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.analysis.statistics import modified_zscore, robust_stats
from src.utils.image_utils import extract_blocks, to_grayscale
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - OpenCV is optional for this module
    cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False


def fft_spectrum(image: np.ndarray) -> Optional[np.ndarray]:
    """Compute the centred 2-D FFT magnitude spectrum of an image.

    Args:
        image: RGB or grayscale array.

    Returns:
        ``(H, W)`` magnitude spectrum with DC at the centre, or ``None``.
    """
    try:
        gray = to_grayscale(image).astype(np.float64)
        if gray.size == 0:
            return None
        gray = gray - float(gray.mean())
        return np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    except Exception as exc:
        LOGGER.debug("fft_spectrum failed: %s", exc)
        return None


def high_frequency_ratio(image: np.ndarray, cutoff: float = 0.25) -> float:
    """Fraction of spectral energy that lives outside the low-frequency core.

    A clean natural image concentrates energy near DC. A stamped trigger patch
    injects broadband energy, pushing this ratio up.

    Args:
        image: RGB or grayscale array.
        cutoff: Radius of the low-frequency disc as a fraction of the half-diagonal.

    Returns:
        High-frequency energy ratio in ``[0, 1]``.
    """
    try:
        spectrum = fft_spectrum(image)
        if spectrum is None:
            return 0.0
        energy = spectrum ** 2
        height, width = energy.shape
        centre_y, centre_x = height / 2.0, width / 2.0
        yy, xx = np.ogrid[:height, :width]
        radius = np.sqrt((yy - centre_y) ** 2 + (xx - centre_x) ** 2)
        limit = float(cutoff) * np.sqrt(centre_y ** 2 + centre_x ** 2)
        total = float(energy.sum())
        if total <= 0:
            return 0.0
        return float(np.clip(energy[radius > limit].sum() / total, 0.0, 1.0))
    except Exception as exc:
        LOGGER.debug("high_frequency_ratio failed: %s", exc)
        return 0.0


def radial_energy_profile(image: np.ndarray, bins: int = 16) -> List[float]:
    """Compute normalised energy in concentric frequency rings.

    Periodic triggers (chequerboards, sinusoidal watermarks) produce a spike in
    a specific ring rather than a smooth decay.

    Args:
        image: RGB or grayscale array.
        bins: Number of radial bins.

    Returns:
        List of ``bins`` energy fractions summing to ~1.
    """
    try:
        spectrum = fft_spectrum(image)
        if spectrum is None:
            return [0.0] * bins
        energy = spectrum ** 2
        height, width = energy.shape
        centre_y, centre_x = height / 2.0, width / 2.0
        yy, xx = np.ogrid[:height, :width]
        radius = np.sqrt((yy - centre_y) ** 2 + (xx - centre_x) ** 2)
        max_radius = float(radius.max()) or 1.0
        indices = np.clip((radius / max_radius * bins).astype(int), 0, bins - 1)
        totals = np.bincount(indices.ravel(), weights=energy.ravel(), minlength=bins)
        grand_total = float(totals.sum())
        if grand_total <= 0:
            return [0.0] * bins
        return [float(v) for v in (totals / grand_total)]
    except Exception as exc:
        LOGGER.debug("radial_energy_profile failed: %s", exc)
        return [0.0] * bins


def spectral_peak_score(image: np.ndarray) -> Dict[str, float]:
    """Detect isolated periodic peaks in the frequency spectrum.

    Args:
        image: RGB or grayscale array.

    Returns:
        Dictionary with ``peak_ratio`` (peak vs median magnitude) and ``peak_count``.
    """
    try:
        spectrum = fft_spectrum(image)
        if spectrum is None:
            return {"peak_ratio": 0.0, "peak_count": 0.0}
        height, width = spectrum.shape
        centre_y, centre_x = height // 2, width // 2
        masked = spectrum.copy()
        exclude = max(2, min(height, width) // 32)
        masked[centre_y - exclude:centre_y + exclude + 1,
               centre_x - exclude:centre_x + exclude + 1] = 0.0
        median = float(np.median(masked[masked > 0])) if np.any(masked > 0) else 0.0
        if median <= 0:
            return {"peak_ratio": 0.0, "peak_count": 0.0}
        peak = float(masked.max())
        return {
            "peak_ratio": float(peak / median),
            "peak_count": float(np.count_nonzero(masked > 20.0 * median)),
        }
    except Exception as exc:
        LOGGER.debug("spectral_peak_score failed: %s", exc)
        return {"peak_ratio": 0.0, "peak_count": 0.0}


def sliding_patch_score(image: np.ndarray, block_size: int = 16) -> Dict[str, Any]:
    """Score an image for a pasted uniform patch using a sliding window.

    A stamped trigger has a signature no natural object reliably produces: a
    small region that is **internally uniform** *and* whose intensity is
    **extreme relative to the rest of the image**. The score is

        max over windows of  (1 - local_std / image_std) * |local_mean - image_mean| / image_std

    A fixed block lattice is deliberately *not* used: a trigger rarely aligns to
    a 16-pixel grid, and a straddled patch is diluted across four blocks until it
    disappears. The window here slides by one pixel via box filters, so the patch
    is always observed whole.

    Args:
        image: RGB or grayscale array.
        block_size: Side length of the sliding window in pixels.

    Returns:
        Dictionary with ``patch_score``, the ``row``/``col`` of the best window,
        its ``uniformity`` and ``deviation`` components, and a ``corner`` label.
    """
    result: Dict[str, Any] = {
        "patch_score": 0.0, "row": 0, "col": 0, "uniformity": 0.0,
        "deviation": 0.0, "corner": "none", "block_size": int(block_size),
    }
    try:
        gray = to_grayscale(image).astype(np.float32)
        height, width = gray.shape[:2]
        size = int(min(block_size, height, width))
        if size < 2:
            return result

        kernel_area = float(size * size)
        if _CV2_AVAILABLE:
            kernel = np.ones((size, size), np.float32) / kernel_area
            local_mean = cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REFLECT)
            local_sq = cv2.filter2D(gray * gray, -1, kernel, borderType=cv2.BORDER_REFLECT)
        else:
            local_mean = _box_filter(gray, size)
            local_sq = _box_filter(gray * gray, size)
        local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0))

        image_mean = float(gray.mean())
        image_std = float(gray.std()) + 1e-6

        uniformity = np.clip(1.0 - local_std / image_std, 0.0, 1.0)
        deviation = np.abs(local_mean - image_mean) / image_std
        score_map = uniformity * deviation

        flat_index = int(np.argmax(score_map))
        row, col = divmod(flat_index, score_map.shape[1])
        result.update({
            "patch_score": round(float(score_map[row, col]), 4),
            "row": int(row), "col": int(col),
            "uniformity": round(float(uniformity[row, col]), 4),
            "deviation": round(float(deviation[row, col]), 4),
            "corner": _corner_of(int(row), int(col), height, width),
        })
        return result
    except Exception as exc:
        LOGGER.debug("sliding_patch_score failed: %s", exc)
        return result


def _box_filter(array: np.ndarray, size: int) -> np.ndarray:
    """Compute a box-filtered mean via a summed-area table (OpenCV fallback).

    Args:
        array: 2-D float array.
        size: Square window side length.

    Returns:
        Array of local means, same shape as the input.
    """
    try:
        padded = np.pad(array, size // 2, mode="reflect")
        integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
        integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
        height, width = array.shape
        total = (integral[size:size + height, size:size + width]
                 - integral[0:height, size:size + width]
                 - integral[size:size + height, 0:width]
                 + integral[0:height, 0:width])
        return total / float(size * size)
    except Exception as exc:
        LOGGER.debug("_box_filter failed: %s", exc)
        return np.full_like(array, float(array.mean()))


def block_variance_map(image: np.ndarray, block_size: int = 16) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Compute per-block intensity variance across an image.

    Args:
        image: RGB or grayscale array.
        block_size: Block side length in pixels.

    Returns:
        Tuple ``(variances, coordinates)``.
    """
    try:
        blocks, coords = extract_blocks(image, block_size=block_size)
        if blocks.size == 0:
            return np.zeros(0, dtype=np.float64), []
        return blocks.reshape(blocks.shape[0], -1).var(axis=1).astype(np.float64), coords
    except Exception as exc:
        LOGGER.debug("block_variance_map failed: %s", exc)
        return np.zeros(0, dtype=np.float64), []


def scan_anomalous_blocks(image: np.ndarray, block_size: int = 16,
                          zscore_threshold: float = 4.0) -> Dict[str, Any]:
    """Find 16x16 blocks whose local statistics are anomalous for that image.

    Two complementary signals are combined:
    * **variance outliers** — a flat white patch on textured terrain, or a
      high-contrast chequerboard on smooth sky;
    * **gradient outliers** — the hard rectangular border of a pasted patch.

    Args:
        image: RGB or grayscale array.
        block_size: Block side length.
        zscore_threshold: Modified Z-score above which a block is flagged.

    Returns:
        Dictionary with ``flagged_blocks`` (coordinates and scores), ``max_zscore``,
        ``num_flagged``, and the spatial ``concentration`` of flagged blocks.
    """
    result: Dict[str, Any] = {
        "flagged_blocks": [], "max_zscore": 0.0, "num_flagged": 0,
        "concentration": 0.0, "block_size": int(block_size),
    }
    try:
        gray = to_grayscale(image).astype(np.float64)
        if gray.size == 0 or min(gray.shape[:2]) < block_size * 2:
            return result

        variances, coords = block_variance_map(gray, block_size)
        if variances.size < 4:
            return result

        gradient_y, gradient_x = np.gradient(gray)
        magnitude = np.hypot(gradient_x, gradient_y)
        gradient_blocks, _ = extract_blocks(magnitude, block_size=block_size)
        gradient_means = (gradient_blocks.reshape(gradient_blocks.shape[0], -1).mean(axis=1)
                          if gradient_blocks.size else np.zeros_like(variances))

        variance_z = modified_zscore(variances)
        gradient_z = modified_zscore(gradient_means[:variances.size])
        combined = np.maximum(np.abs(variance_z), np.abs(gradient_z))

        flagged: List[Dict[str, Any]] = []
        for index, score in enumerate(combined):
            if score >= zscore_threshold and index < len(coords):
                row, col = coords[index]
                flagged.append({
                    "row": int(row), "col": int(col),
                    "zscore": round(float(score), 3),
                    "variance": round(float(variances[index]), 3),
                    "variance_z": round(float(variance_z[index]), 3),
                    "gradient_z": round(float(gradient_z[index]), 3),
                })

        flagged.sort(key=lambda b: b["zscore"], reverse=True)
        result["flagged_blocks"] = flagged[:10]
        result["num_flagged"] = len(flagged)
        result["max_zscore"] = round(float(combined.max()) if combined.size else 0.0, 3)

        if len(flagged) >= 2:
            positions = np.array([[b["row"], b["col"]] for b in flagged], dtype=np.float64)
            spread = float(np.mean(np.std(positions, axis=0)))
            diagonal = float(np.hypot(*gray.shape[:2]))
            result["concentration"] = round(
                float(np.clip(1.0 - spread / max(diagonal * 0.25, 1e-6), 0.0, 1.0)), 3)
        elif len(flagged) == 1:
            result["concentration"] = 1.0

        if flagged:
            top = flagged[0]
            height, width = gray.shape[:2]
            result["corner_hint"] = _corner_of(top["row"], top["col"], height, width)
        return result
    except Exception as exc:
        LOGGER.error("scan_anomalous_blocks failed: %s", exc)
        return result


def _corner_of(row: int, col: int, height: int, width: int) -> str:
    """Name the image quadrant containing a coordinate.

    BadNets-style attacks overwhelmingly place triggers in a fixed corner, so
    naming the quadrant makes the evidence immediately actionable.

    Args:
        row: Block row in pixels.
        col: Block column in pixels.
        height: Image height.
        width: Image width.

    Returns:
        One of ``top-left``, ``top-right``, ``bottom-left``, ``bottom-right``, ``centre``.
    """
    try:
        vertical = "top" if row < height * 0.4 else ("bottom" if row > height * 0.6 else "centre")
        horizontal = "left" if col < width * 0.4 else ("right" if col > width * 0.6 else "centre")
        if vertical == "centre" and horizontal == "centre":
            return "centre"
        if vertical == "centre":
            return horizontal
        if horizontal == "centre":
            return vertical
        return f"{vertical}-{horizontal}"
    except Exception:
        return "unknown"


def analyze_image_frequency(image: np.ndarray, block_size: int = 16,
                            block_zscore_threshold: float = 4.0,
                            hf_cutoff: float = 0.25) -> Dict[str, Any]:
    """Run the full frequency + spatial trigger analysis on one image.

    Args:
        image: RGB array.
        block_size: Spatial block side length.
        block_zscore_threshold: Block anomaly threshold.
        hf_cutoff: FFT low-frequency cutoff fraction.

    Returns:
        Dictionary of all frequency/spatial evidence for this image.
    """
    try:
        blocks = scan_anomalous_blocks(image, block_size, block_zscore_threshold)
        peaks = spectral_peak_score(image)
        patch = sliding_patch_score(image, block_size=max(block_size // 2, 4))
        return {
            "patch_score": patch["patch_score"],
            "patch_corner": patch["corner"],
            "patch_uniformity": patch["uniformity"],
            "patch_deviation": patch["deviation"],
            "patch_row": patch["row"],
            "patch_col": patch["col"],
            "high_freq_ratio": round(high_frequency_ratio(image, hf_cutoff), 5),
            "spectral_peak_ratio": round(peaks["peak_ratio"], 3),
            "spectral_peak_count": int(peaks["peak_count"]),
            "block_max_zscore": blocks["max_zscore"],
            "block_num_flagged": blocks["num_flagged"],
            "block_concentration": blocks["concentration"],
            "block_corner": blocks.get("corner_hint", "none"),
            "flagged_blocks": blocks["flagged_blocks"][:3],
        }
    except Exception as exc:
        LOGGER.error("analyze_image_frequency failed: %s", exc)
        return {"high_freq_ratio": 0.0, "block_max_zscore": 0.0, "block_num_flagged": 0,
                "block_concentration": 0.0, "block_corner": "none", "flagged_blocks": [],
                "patch_score": 0.0, "patch_corner": "none", "error": str(exc)}


def population_frequency_outliers(ratios: Sequence[float],
                                  zscore_threshold: float = 2.5) -> Dict[str, Any]:
    """Compare per-image high-frequency ratios across the whole dataset.

    An absolute ratio is scene-dependent; what matters forensically is that a
    handful of images deviate from *their own dataset's* population.

    Args:
        ratios: One high-frequency ratio per image.
        zscore_threshold: Modified Z-score cutoff.

    Returns:
        Dictionary with per-image ``zscores``, ``outlier_indices`` and population stats.
    """
    try:
        scores = modified_zscore(ratios)
        outliers = [int(i) for i, z in enumerate(scores) if z >= zscore_threshold]
        return {
            "zscores": [round(float(z), 4) for z in scores],
            "outlier_indices": outliers,
            "threshold": float(zscore_threshold),
            "population": robust_stats(ratios),
        }
    except Exception as exc:
        LOGGER.error("population_frequency_outliers failed: %s", exc)
        return {"zscores": [], "outlier_indices": [], "threshold": zscore_threshold,
                "population": {}}


__all__ = [
    "fft_spectrum", "high_frequency_ratio", "radial_energy_profile", "spectral_peak_score",
    "block_variance_map", "scan_anomalous_blocks", "sliding_patch_score",
    "analyze_image_frequency", "population_frequency_outliers",
]
