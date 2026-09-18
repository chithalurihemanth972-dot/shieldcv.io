"""
Robust statistical primitives shared by every SHIELD-CV detector.

Poisoned datasets are, by construction, contaminated — so classical mean/std
Z-scores are pulled towards the attacker's samples. These helpers therefore
prefer median/MAD (modified Z-score) estimators, which stay stable up to ~50%
contamination.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

MAD_SCALE = 0.6745  # makes MAD a consistent estimator of sigma for normal data


def safe_array(values: Sequence[float]) -> np.ndarray:
    """Convert any sequence to a clean 1-D float array with NaNs removed.

    Args:
        values: Input sequence.

    Returns:
        Finite ``float64`` array (possibly empty).
    """
    try:
        array = np.asarray(list(values), dtype=np.float64).ravel()
        return array[np.isfinite(array)]
    except Exception as exc:
        LOGGER.debug("safe_array failed: %s", exc)
        return np.zeros(0, dtype=np.float64)


def zscore(values: Sequence[float]) -> np.ndarray:
    """Compute classical Z-scores ``(x - mean) / std``.

    Args:
        values: Input sequence.

    Returns:
        Array of Z-scores (zeros when std is degenerate).
    """
    try:
        array = np.asarray(values, dtype=np.float64).ravel()
        if array.size == 0:
            return array
        std = float(np.std(array))
        if std < 1e-12:
            return np.zeros_like(array)
        return (array - float(np.mean(array))) / std
    except Exception as exc:
        LOGGER.error("zscore failed: %s", exc)
        return np.zeros(len(values), dtype=np.float64)


def modified_zscore(values: Sequence[float]) -> np.ndarray:
    """Compute MAD-based modified Z-scores, robust to contamination.

    Uses ``0.6745 * (x - median) / MAD``; falls back to the mean absolute
    deviation when the MAD is zero (e.g. many identical values).

    Args:
        values: Input sequence.

    Returns:
        Array of modified Z-scores.
    """
    try:
        array = np.asarray(values, dtype=np.float64).ravel()
        if array.size == 0:
            return array
        median = float(np.median(array))
        deviations = np.abs(array - median)
        mad = float(np.median(deviations))
        if mad < 1e-12:
            mean_ad = float(np.mean(deviations))
            if mean_ad < 1e-12:
                return np.zeros_like(array)
            return (array - median) / (1.253314 * mean_ad)
        return MAD_SCALE * (array - median) / mad
    except Exception as exc:
        LOGGER.error("modified_zscore failed: %s", exc)
        return np.zeros(len(values), dtype=np.float64)


def mad(values: Sequence[float]) -> float:
    """Median absolute deviation.

    Args:
        values: Input sequence.

    Returns:
        MAD as a float (0.0 when undefined).
    """
    try:
        array = safe_array(values)
        if array.size == 0:
            return 0.0
        return float(np.median(np.abs(array - np.median(array))))
    except Exception:
        return 0.0


def robust_stats(values: Sequence[float]) -> Dict[str, float]:
    """Compute a compact robust summary of a distribution.

    Args:
        values: Input sequence.

    Returns:
        Mapping with n, mean, std, median, mad, min, max, q1, q3, iqr, skew, kurtosis.
    """
    try:
        array = safe_array(values)
        if array.size == 0:
            return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "mad": 0.0,
                    "min": 0.0, "max": 0.0, "q1": 0.0, "q3": 0.0, "iqr": 0.0,
                    "skew": 0.0, "kurtosis": 0.0}
        q1, q3 = (float(v) for v in np.percentile(array, [25, 75]))
        return {
            "n": int(array.size),
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "median": float(np.median(array)),
            "mad": mad(array),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "skew": skewness(array),
            "kurtosis": kurtosis(array),
        }
    except Exception as exc:
        LOGGER.error("robust_stats failed: %s", exc)
        return {"n": 0, "error": 1.0}


def skewness(values: Sequence[float]) -> float:
    """Fisher-Pearson sample skewness.

    Args:
        values: Input sequence.

    Returns:
        Skewness (0.0 when undefined).
    """
    try:
        array = safe_array(values)
        if array.size < 3:
            return 0.0
        std = float(np.std(array))
        if std < 1e-12:
            return 0.0
        return float(np.mean(((array - np.mean(array)) / std) ** 3))
    except Exception:
        return 0.0


def kurtosis(values: Sequence[float], excess: bool = True) -> float:
    """Sample kurtosis, optionally excess (normal distribution → 0).

    Sharp weight-distribution spikes from an implanted backdoor show up here.

    Args:
        values: Input sequence.
        excess: Subtract 3 to report excess kurtosis.

    Returns:
        Kurtosis value.
    """
    try:
        array = safe_array(values)
        if array.size < 4:
            return 0.0
        std = float(np.std(array))
        if std < 1e-12:
            return 0.0
        value = float(np.mean(((array - np.mean(array)) / std) ** 4))
        return value - 3.0 if excess else value
    except Exception:
        return 0.0


def entropy(probabilities: Sequence[float], base: Optional[float] = None) -> float:
    """Shannon entropy of a probability vector.

    Args:
        probabilities: Non-negative weights (normalised internally).
        base: Logarithm base; natural log when ``None``.

    Returns:
        Entropy value (0.0 when degenerate).
    """
    try:
        array = safe_array(probabilities)
        array = array[array > 0]
        if array.size == 0:
            return 0.0
        array = array / array.sum()
        value = float(-np.sum(array * np.log(array)))
        if base:
            value /= float(np.log(base))
        return value
    except Exception:
        return 0.0


def normalized_entropy(probabilities: Sequence[float]) -> float:
    """Entropy scaled to ``[0, 1]`` by the maximum for that vector length.

    Args:
        probabilities: Probability vector.

    Returns:
        Normalised entropy; 0 = one-hot (sharp), 1 = uniform (flat).
    """
    try:
        array = safe_array(probabilities)
        if array.size <= 1:
            return 0.0
        return float(np.clip(entropy(array) / np.log(array.size), 0.0, 1.0))
    except Exception:
        return 0.0


def softmax(logits: np.ndarray, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax.

    Args:
        logits: Raw scores.
        temperature: Temperature divisor (>0).
        axis: Axis to normalise over.

    Returns:
        Probability array of the same shape.
    """
    try:
        scaled = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
        shifted = scaled - np.max(scaled, axis=axis, keepdims=True)
        exponentials = np.exp(shifted)
        return (exponentials / np.sum(exponentials, axis=axis, keepdims=True)).astype(np.float64)
    except Exception as exc:
        LOGGER.error("softmax failed: %s", exc)
        shape = np.asarray(logits).shape
        return np.full(shape, 1.0 / max(shape[-1], 1))


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Energy-based OOD score ``-T * logsumexp(logits / T)``.

    Lower energy means more in-distribution; high energy flags OOD samples.

    Args:
        logits: ``(N, C)`` raw logits.
        temperature: Energy temperature.

    Returns:
        ``(N,)`` array of energies.
    """
    try:
        array = np.asarray(logits, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        temp = max(float(temperature), 1e-6)
        scaled = array / temp
        maxima = np.max(scaled, axis=1, keepdims=True)
        lse = (maxima + np.log(np.sum(np.exp(scaled - maxima), axis=1, keepdims=True))).ravel()
        return (-temp * lse).astype(np.float64)
    except Exception as exc:
        LOGGER.error("energy_score failed: %s", exc)
        return np.zeros(len(logits), dtype=np.float64)


def mahalanobis_distances(features: np.ndarray,
                          mean: Optional[np.ndarray] = None,
                          covariance: Optional[np.ndarray] = None,
                          shrinkage: float = 0.1) -> np.ndarray:
    """Mahalanobis distance of each row from a (shrunk) Gaussian fit.

    Ledoit-Wolf-style shrinkage towards a scaled identity keeps the covariance
    invertible when D >> N, which is the norm for 512-d embeddings of small classes.

    Args:
        features: ``(N, D)`` embeddings.
        mean: Optional precomputed class mean.
        covariance: Optional precomputed covariance.
        shrinkage: Shrinkage intensity in ``[0, 1]``.

    Returns:
        ``(N,)`` array of distances (zeros on failure).
    """
    try:
        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] == 0:
            return np.zeros(max(array.shape[0], 0), dtype=np.float64)
        centre = np.asarray(mean, dtype=np.float64) if mean is not None else array.mean(axis=0)
        centred = array - centre

        if covariance is None:
            if array.shape[0] < 2:
                return np.linalg.norm(centred, axis=1)
            covariance = np.cov(centred, rowvar=False)
        covariance = np.atleast_2d(np.asarray(covariance, dtype=np.float64))

        dimension = covariance.shape[0]
        trace_mean = float(np.trace(covariance)) / max(dimension, 1)
        alpha = float(np.clip(shrinkage, 0.0, 1.0))
        regularised = (1 - alpha) * covariance + alpha * trace_mean * np.eye(dimension)
        regularised += 1e-6 * np.eye(dimension)

        try:
            inverse = np.linalg.inv(regularised)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(regularised)

        quadratic = np.einsum("ij,jk,ik->i", centred, inverse, centred)
        return np.sqrt(np.maximum(quadratic, 0.0))
    except Exception as exc:
        LOGGER.error("mahalanobis_distances failed: %s", exc)
        return np.zeros(np.asarray(features).shape[0], dtype=np.float64)


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of row vectors.

    Args:
        features: ``(N, D)`` array.

    Returns:
        ``(N, N)`` similarity matrix.
    """
    try:
        array = np.asarray(features, dtype=np.float64)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        normalised = array / np.maximum(norms, 1e-9)
        return normalised @ normalised.T
    except Exception as exc:
        LOGGER.error("cosine_similarity_matrix failed: %s", exc)
        size = np.asarray(features).shape[0]
        return np.eye(size)


def maximum_mean_discrepancy(x: np.ndarray, y: np.ndarray,
                             gamma: Optional[float] = None,
                             max_samples: int = 500) -> Dict[str, float]:
    """Unbiased MMD² between two samples using an RBF kernel.

    Args:
        x: ``(N, D)`` reference sample.
        y: ``(M, D)`` current sample.
        gamma: RBF bandwidth; median heuristic when ``None``.
        max_samples: Subsampling cap (keeps the kernel matrix laptop-sized).

    Returns:
        Dictionary with ``mmd2``, ``mmd``, ``gamma`` and the sample sizes used.
    """
    try:
        rng = np.random.default_rng(1337)
        a = np.asarray(x, dtype=np.float64)
        b = np.asarray(y, dtype=np.float64)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] < 2 or b.shape[0] < 2:
            return {"mmd2": 0.0, "mmd": 0.0, "gamma": 0.0, "n_x": len(a), "n_y": len(b)}
        if a.shape[0] > max_samples:
            a = a[rng.choice(a.shape[0], max_samples, replace=False)]
        if b.shape[0] > max_samples:
            b = b[rng.choice(b.shape[0], max_samples, replace=False)]
        if a.shape[1] != b.shape[1]:
            dimension = min(a.shape[1], b.shape[1])
            a, b = a[:, :dimension], b[:, :dimension]

        combined = np.vstack([a, b])
        squared = np.sum(combined ** 2, axis=1)
        distances = np.maximum(
            squared[:, None] + squared[None, :] - 2.0 * (combined @ combined.T), 0.0)

        if gamma is None:
            upper = distances[np.triu_indices_from(distances, k=1)]
            median = float(np.median(upper)) if upper.size else 1.0
            gamma = 1.0 / max(median, 1e-9)

        kernel = np.exp(-float(gamma) * distances)
        n, m = a.shape[0], b.shape[0]
        k_xx = kernel[:n, :n]
        k_yy = kernel[n:, n:]
        k_xy = kernel[:n, n:]
        term_xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
        term_yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
        term_xy = k_xy.mean()
        mmd2 = float(term_xx + term_yy - 2.0 * term_xy)
        return {
            "mmd2": mmd2,
            "mmd": float(np.sqrt(max(mmd2, 0.0))),
            "gamma": float(gamma),
            "n_x": int(n),
            "n_y": int(m),
        }
    except Exception as exc:
        LOGGER.error("maximum_mean_discrepancy failed: %s", exc)
        return {"mmd2": 0.0, "mmd": 0.0, "gamma": 0.0, "n_x": 0, "n_y": 0}


def kolmogorov_smirnov(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Two-sample Kolmogorov-Smirnov test with an asymptotic p-value.

    Args:
        x: First sample.
        y: Second sample.

    Returns:
        Dictionary with ``statistic`` and ``p_value``.
    """
    try:
        a, b = safe_array(x), safe_array(y)
        if a.size == 0 or b.size == 0:
            return {"statistic": 0.0, "p_value": 1.0}
        try:
            from scipy import stats as scipy_stats
            result = scipy_stats.ks_2samp(a, b)
            return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
        except ImportError:
            pass
        grid = np.sort(np.concatenate([a, b]))
        cdf_a = np.searchsorted(np.sort(a), grid, side="right") / a.size
        cdf_b = np.searchsorted(np.sort(b), grid, side="right") / b.size
        statistic = float(np.max(np.abs(cdf_a - cdf_b)))
        effective = np.sqrt(a.size * b.size / (a.size + b.size))
        lam = (effective + 0.12 + 0.11 / effective) * statistic
        p_value = float(np.clip(
            2.0 * sum((-1) ** (k - 1) * np.exp(-2.0 * k * k * lam * lam)
                      for k in range(1, 101)), 0.0, 1.0))
        return {"statistic": statistic, "p_value": p_value}
    except Exception as exc:
        LOGGER.error("kolmogorov_smirnov failed: %s", exc)
        return {"statistic": 0.0, "p_value": 1.0}


def confidence_from_zscore(z: float, threshold: float, ceiling: float = 0.99) -> float:
    """Map a Z-score above a threshold onto a calibrated confidence in ``[0, 1]``.

    A score exactly at the threshold yields ~0.5; confidence saturates smoothly
    towards ``ceiling`` as the score grows, so reports never claim certainty.

    Args:
        z: Observed (modified) Z-score.
        threshold: Detection threshold for this detector.
        ceiling: Maximum confidence ever reported.

    Returns:
        Confidence value.
    """
    try:
        magnitude = abs(float(z))
        limit = max(abs(float(threshold)), 1e-6)
        if magnitude <= 0:
            return 0.0
        ratio = magnitude / limit
        confidence = 1.0 - np.exp(-0.8 * ratio)
        confidence = 0.5 * (confidence / (1.0 - np.exp(-0.8))) if ratio <= 1.0 else (
            0.5 + 0.5 * (1.0 - np.exp(-1.2 * (ratio - 1.0))))
        return float(np.clip(confidence, 0.0, ceiling))
    except Exception:
        return 0.0


def aggregate_risk(scores: Sequence[float], weights: Optional[Sequence[float]] = None,
                   method: str = "noisy_or") -> float:
    """Combine multiple risk signals into a single ``[0, 1]`` score.

    Args:
        scores: Individual risk/confidence values.
        weights: Optional weights (``weighted_mean`` only).
        method: ``"noisy_or"``, ``"weighted_mean"`` or ``"max"``.

    Returns:
        Aggregated risk score.
    """
    try:
        array = np.clip(safe_array(scores), 0.0, 1.0)
        if array.size == 0:
            return 0.0
        if method == "max":
            return float(np.max(array))
        if method == "weighted_mean":
            weight_array = (np.asarray(weights, dtype=np.float64)
                            if weights is not None else np.ones_like(array))
            weight_array = weight_array[:array.size]
            total = float(weight_array.sum())
            if total <= 0:
                return float(np.mean(array))
            return float(np.clip(np.dot(array, weight_array) / total, 0.0, 1.0))
        return float(np.clip(1.0 - np.prod(1.0 - array), 0.0, 1.0))
    except Exception as exc:
        LOGGER.error("aggregate_risk failed: %s", exc)
        return 0.0


def percentile_threshold(values: Sequence[float], percentile: float) -> float:
    """Return the value at a given percentile of a distribution.

    Args:
        values: Input sequence.
        percentile: Percentile in ``[0, 100]``.

    Returns:
        Threshold value (0.0 when empty).
    """
    try:
        array = safe_array(values)
        if array.size == 0:
            return 0.0
        return float(np.percentile(array, float(np.clip(percentile, 0.0, 100.0))))
    except Exception:
        return 0.0


__all__ = [
    "zscore", "modified_zscore", "mad", "robust_stats", "skewness", "kurtosis", "entropy",
    "normalized_entropy", "softmax", "energy_score", "mahalanobis_distances",
    "cosine_similarity_matrix", "maximum_mean_discrepancy", "kolmogorov_smirnov",
    "confidence_from_zscore", "aggregate_risk", "percentile_threshold", "safe_array",
]
