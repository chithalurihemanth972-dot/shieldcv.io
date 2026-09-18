"""
Outlier, duplicate and mislabel detectors operating in embedding space.

Covers four of the five data-integrity detectors:

* :func:`knn_centroid_outliers` — label flipping (distance from class centroid)
* :func:`detect_near_duplicates` — pHash near-duplicate flooding
* :func:`mahalanobis_ood` / :func:`energy_ood` — out-of-distribution samples
* :func:`confident_learning` — systematic mislabeling (confidence vs. given label)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.analysis.statistics import (
    energy_score,
    mahalanobis_distances,
    modified_zscore,
    percentile_threshold,
    robust_stats,
    softmax,
)
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# (b) LABEL FLIPPING — KNN distance from class centroid
# ---------------------------------------------------------------------------
def knn_centroid_outliers(features: np.ndarray,
                          labels: Sequence[int],
                          k: int = 10,
                          std_multiplier: float = 2.0,
                          min_class_size: int = 8,
                          knn_disagreement_threshold: float = 0.60,
                          knn_majority_threshold: float = 0.50) -> Dict[str, Any]:
    """Flag samples that sit far from their own class centroid in feature space.

    A flipped label puts a visually-consistent sample inside the wrong class, so
    its distance to that class's centroid is anomalous. Two independent signals
    are combined:

    * distance to the assigned class centroid exceeding ``mean + k*std``;
    * KNN label agreement — how many of its k nearest neighbours share its label,
      and which class those neighbours actually belong to.

    Args:
        features: ``(N, D)`` embeddings.
        labels: Assigned class label per row.
        k: Number of neighbours to inspect.
        std_multiplier: Threshold multiplier on the class distance distribution.
        min_class_size: Classes smaller than this are skipped.
        knn_disagreement_threshold: Minimum fraction of neighbours that must
            disagree with the assigned label to raise the neighbourhood flag.
        knn_majority_threshold: Minimum share a single competing class must hold
            among the neighbours for the flag to name a suggested class.

    Returns:
        Dictionary with ``flagged`` entries (index, distance, threshold,
        suggested class, neighbour agreement), plus per-class statistics.
    """
    result: Dict[str, Any] = {"flagged": [], "class_stats": {}, "available": False,
                              "reason": ""}
    try:
        matrix = np.asarray(features, dtype=np.float64)
        label_array = np.asarray(labels)
        if matrix.ndim != 2 or matrix.shape[0] != label_array.shape[0] or matrix.shape[0] < 4:
            result["reason"] = "Insufficient or mismatched features/labels"
            return result

        centroids: Dict[int, np.ndarray] = {}
        for class_id in np.unique(label_array):
            members = matrix[label_array == class_id]
            if members.shape[0] > 0:
                centroids[int(class_id)] = members.mean(axis=0)

        norms = np.sum(matrix ** 2, axis=1)
        distance_matrix = np.maximum(
            norms[:, None] + norms[None, :] - 2.0 * (matrix @ matrix.T), 0.0)
        np.fill_diagonal(distance_matrix, np.inf)
        effective_k = int(min(max(k, 1), max(matrix.shape[0] - 1, 1)))
        neighbour_indices = np.argsort(distance_matrix, axis=1)[:, :effective_k]

        flagged: List[Dict[str, Any]] = []
        for class_id, centroid in centroids.items():
            positions = np.where(label_array == class_id)[0]
            if positions.size < int(min_class_size):
                result["class_stats"][int(class_id)] = {
                    "n": int(positions.size), "skipped": True,
                    "reason": f"class smaller than min_class_size={min_class_size}",
                }
                continue

            distances = np.linalg.norm(matrix[positions] - centroid, axis=1)
            mean_distance = float(np.mean(distances))
            std_distance = float(np.std(distances))
            threshold = mean_distance + float(std_multiplier) * std_distance
            result["class_stats"][int(class_id)] = {
                "n": int(positions.size),
                "mean_distance": round(mean_distance, 5),
                "std_distance": round(std_distance, 5),
                "threshold": round(threshold, 5),
                "skipped": False,
            }

            for local_index, global_index in enumerate(positions):
                distance = float(distances[local_index])

                neighbours = neighbour_indices[global_index]
                neighbour_labels = label_array[neighbours]
                agreement = float(np.mean(neighbour_labels == class_id))
                values, counts = np.unique(neighbour_labels, return_counts=True)
                majority = int(values[int(np.argmax(counts))])
                majority_share = float(np.max(counts) / max(len(neighbour_labels), 1))

                # Two independent routes to a flag. Centroid distance alone is a
                # weak signal: a flipped label often lands *inside* the wrong
                # class's cloud, so its distance looks unremarkable. Neighbourhood
                # disagreement is what actually exposes it — the sample's k
                # nearest neighbours overwhelmingly carry a different label.
                distance_flag = distance > threshold and std_distance >= 1e-9
                neighbour_flag = (agreement <= (1.0 - float(knn_disagreement_threshold))
                                  and majority != class_id
                                  and majority_share >= float(knn_majority_threshold))
                if not (distance_flag or neighbour_flag):
                    continue

                flagged.append({
                    "distance_flag": bool(distance_flag),
                    "neighbour_flag": bool(neighbour_flag),
                    "index": int(global_index),
                    "assigned_class": int(class_id),
                    "distance": round(distance, 5),
                    "threshold": round(threshold, 5),
                    "distance_ratio": round(distance / max(threshold, 1e-9), 4),
                    "knn_agreement": round(agreement, 4),
                    "suggested_class": majority,
                    "suggested_share": round(majority_share, 4),
                    "label_conflict": bool(majority != class_id and majority_share >= 0.5),
                })

        flagged.sort(key=lambda f: (1.0 - f["knn_agreement"], f["distance_ratio"]),
                     reverse=True)
        result["flagged"] = flagged
        result["available"] = True
        result["k"] = effective_k
        LOGGER.info("KNN/centroid analysis flagged %d sample(s)", len(flagged))
        return result
    except Exception as exc:
        LOGGER.error("knn_centroid_outliers failed: %s", exc)
        result["reason"] = str(exc)
        return result


# ---------------------------------------------------------------------------
# (c) NEAR-DUPLICATE FLOODING — perceptual hashing
# ---------------------------------------------------------------------------
def compute_phashes(paths: Sequence[str | Path], hash_size: int = 8) -> Dict[int, Any]:
    """Compute perceptual hashes (pHash) for a list of images.

    Args:
        paths: Image file paths.
        hash_size: pHash side length (8 → 64-bit hash).

    Returns:
        Mapping of list index to an ``imagehash.ImageHash`` (or a numpy bit
        array when ``imagehash`` is unavailable).
    """
    hashes: Dict[int, Any] = {}
    try:
        try:
            import imagehash
            from PIL import Image
            use_imagehash = True
        except ImportError:
            use_imagehash = False
            LOGGER.warning("imagehash unavailable; using internal DCT pHash fallback")

        for index, path in enumerate(paths):
            try:
                if use_imagehash:
                    with Image.open(str(path)) as image:
                        hashes[index] = imagehash.phash(image.convert("RGB"),
                                                        hash_size=int(hash_size))
                else:
                    value = _fallback_phash(path, hash_size)
                    if value is not None:
                        hashes[index] = value
            except Exception as exc:
                LOGGER.debug("pHash failed for %s: %s", path, exc)
        return hashes
    except Exception as exc:
        LOGGER.error("compute_phashes failed: %s", exc)
        return hashes


def _fallback_phash(path: str | Path, hash_size: int = 8) -> Optional[np.ndarray]:
    """Compute a DCT-based perceptual hash without the ``imagehash`` package.

    Args:
        path: Image path.
        hash_size: Hash side length.

    Returns:
        Boolean bit array, or ``None`` on failure.
    """
    try:
        from scipy.fftpack import dct
        from src.utils.image_utils import load_image, resize_image, to_grayscale
        image = load_image(path)
        if image is None:
            return None
        size = int(hash_size) * 4
        gray = to_grayscale(resize_image(image, (size, size))).astype(np.float64)
        transformed = dct(dct(gray, axis=0, norm="ortho"), axis=1, norm="ortho")
        low = transformed[:hash_size, :hash_size]
        median = float(np.median(low[1:, 1:]))
        return (low > median).ravel()
    except Exception as exc:
        LOGGER.debug("_fallback_phash failed: %s", exc)
        return None


def _hamming(a: Any, b: Any) -> int:
    """Hamming distance between two perceptual hashes of either representation.

    Args:
        a: First hash.
        b: Second hash.

    Returns:
        Number of differing bits (large sentinel on failure).
    """
    try:
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return int(np.count_nonzero(a != b))
        return int(a - b)
    except Exception:
        return 9999


# ---------------------------------------------------------------------------
# (c) NEAR-DUPLICATE — SSIM structural confirmation
# ---------------------------------------------------------------------------
def compute_ssim(path_a: str | Path, path_b: str | Path,
                 size: int = 128) -> Optional[float]:
    """Compute the structural similarity index between two images.

    pHash is a 64-bit summary and is deliberately lossy: two genuinely
    different photographs of the same subject, shot from the same angle, can
    land within a few bits of each other. SSIM compares local luminance,
    contrast and structure on the actual pixels, so it separates a real
    re-encoded copy from a merely similar scene. It is the confirmation half of
    the pHash-then-SSIM duplicate rule.

    Implemented directly on numpy with a Gaussian window rather than pulling in
    scikit-image, keeping the air-gapped dependency set minimal.

    Args:
        path_a: First image path.
        path_b: Second image path.
        size: Side length both images are resized to before comparison.

    Returns:
        SSIM in ``-1..1`` (1.0 is identical), or ``None`` if either image
        cannot be read.
    """
    try:
        from src.utils.image_utils import load_image

        first = load_image(path_a, size=(size, size), grayscale=True)
        second = load_image(path_b, size=(size, size), grayscale=True)
        if first is None or second is None:
            return None

        a = np.asarray(first, dtype=np.float64)
        b = np.asarray(second, dtype=np.float64)
        if a.ndim == 3:
            a = a.mean(axis=2)
        if b.ndim == 3:
            b = b.mean(axis=2)
        if a.shape != b.shape:
            return None

        # Standard SSIM constants for 8-bit dynamic range.
        data_range = 255.0
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2

        # 11x11 Gaussian window, sigma 1.5, as in the original SSIM paper.
        radius = 5
        sigma = 1.5
        coords = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel_1d = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        kernel_1d /= kernel_1d.sum()

        def blur(matrix: np.ndarray) -> np.ndarray:
            """Apply a separable Gaussian blur with edge padding."""
            padded = np.pad(matrix, radius, mode="symmetric")
            rows = np.apply_along_axis(
                lambda m: np.convolve(m, kernel_1d, mode="valid"), 1, padded)
            return np.apply_along_axis(
                lambda m: np.convolve(m, kernel_1d, mode="valid"), 0, rows)

        mu_a, mu_b = blur(a), blur(b)
        mu_a_sq, mu_b_sq, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
        sigma_a = blur(a * a) - mu_a_sq
        sigma_b = blur(b * b) - mu_b_sq
        sigma_ab = blur(a * b) - mu_ab

        numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
        denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a + sigma_b + c2)
        with np.errstate(divide="ignore", invalid="ignore"):
            ssim_map = np.where(denominator > 0, numerator / denominator, 0.0)
        return float(np.clip(np.mean(ssim_map), -1.0, 1.0))
    except Exception as exc:
        LOGGER.error("compute_ssim failed for %s vs %s: %s", path_a, path_b, exc)
        return None


def detect_near_duplicates(paths: Sequence[str | Path],
                           hash_size: int = 8,
                           hamming_threshold: int = 5,
                           flood_cluster_min_size: int = 4,
                           use_ssim: bool = True,
                           ssim_threshold: Optional[float] = None,
                           ssim_baseline_samples: int = 60,
                           adaptive_flood: bool = True,
                           flood_mad_multiplier: float = 3.0,
                           seed: int = 1337) -> Dict[str, Any]:
    """Detect near-duplicate images and duplicate-flooding clusters.

    Images within ``hamming_threshold`` bits are grouped with a union-find, so
    chains of slightly-augmented copies collapse into one cluster.

    Two safeguards keep this from drowning an analyst in benign findings, both
    learned from measuring real corpora rather than assumed:

    **SSIM confirmation.** A 64-bit pHash is a lossy summary; two different
    photographs of the same subject can land within a few bits of each other.
    Every pHash-linked pair is therefore re-checked on actual pixels with
    :func:`compute_ssim`, and unconfirmed links are dropped before clustering.

    **Population-relative flooding.** An absolute "4 or more copies is an
    attack" rule is meaningless, because how many near-identical frames occur
    naturally depends entirely on the corpus — a burst-mode camera set is not a
    poisoned one. The flood threshold is therefore derived from the dataset's
    *own* cluster-size distribution (median + ``flood_mad_multiplier`` × MAD),
    floored at ``flood_cluster_min_size``.

    The SSIM threshold is likewise calibrated against a random-pair baseline
    drawn from this dataset, because a visually homogeneous corpus scores high
    SSIM even on unrelated images; a fixed constant would confirm everything.

    Args:
        paths: Image file paths.
        hash_size: pHash side length.
        hamming_threshold: Maximum bit distance for a near-duplicate.
        flood_cluster_min_size: Absolute floor for the flooding threshold.
        use_ssim: Whether to structurally confirm pHash links with SSIM.
        ssim_threshold: Explicit SSIM cut-off; calibrated adaptively when
            ``None``.
        ssim_baseline_samples: Random pairs sampled to calibrate that cut-off.
        adaptive_flood: Whether to derive the flood threshold from the corpus.
        flood_mad_multiplier: MAD multiplier for the flood threshold.
        seed: Seed for the deterministic baseline sample.

    Returns:
        Dictionary with ``clusters``, ``duplicate_indices``, ``flood_clusters``,
        the thresholds actually applied and coverage counts.
    """
    result: Dict[str, Any] = {
        "clusters": [], "duplicate_indices": [], "flood_clusters": [],
        "num_hashed": 0, "available": False, "reason": "",
    }
    try:
        hashes = compute_phashes(paths, hash_size=hash_size)
        result["num_hashed"] = len(hashes)
        if len(hashes) < 2:
            result["reason"] = "Fewer than 2 hashable images"
            return result

        indices = sorted(hashes.keys())
        parent = {i: i for i in indices}

        def find(node: int) -> int:
            """Union-find root lookup with path compression."""
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(a: int, b: int) -> None:
            """Merge two union-find sets."""
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        # --- pass 1: candidate links by pHash -------------------------------
        candidates: List[Tuple[int, int, int]] = []
        for position, i in enumerate(indices):
            for j in indices[position + 1:]:
                distance = _hamming(hashes[i], hashes[j])
                if distance <= int(hamming_threshold):
                    candidates.append((i, j, distance))

        # --- pass 2: calibrate the SSIM cut-off on this corpus ---------------
        applied_ssim = ssim_threshold
        baseline: Dict[str, Any] = {}
        if use_ssim and candidates and applied_ssim is None:
            rng = np.random.default_rng(seed)
            linked = {(i, j) for i, j, _ in candidates}
            scores: List[float] = []
            attempts = 0
            while len(scores) < int(ssim_baseline_samples) and attempts < int(ssim_baseline_samples) * 6:
                attempts += 1
                a, b = (int(rng.integers(len(indices))), int(rng.integers(len(indices))))
                if a == b:
                    continue
                pair = (indices[min(a, b)], indices[max(a, b)])
                if pair in linked:
                    continue
                value = compute_ssim(paths[pair[0]], paths[pair[1]])
                if value is not None:
                    scores.append(value)
            if len(scores) >= 8:
                array = np.asarray(scores, dtype=float)
                # Unrelated pairs define what "merely similar" looks like here;
                # a true duplicate must clear that ceiling by a clear margin.
                applied_ssim = float(min(0.995, max(0.90, np.percentile(array, 99) + 0.01)))
                baseline = {
                    "samples": len(scores),
                    "median": round(float(np.median(array)), 4),
                    "p95": round(float(np.percentile(array, 95)), 4),
                    "p99": round(float(np.percentile(array, 99)), 4),
                }
            else:
                applied_ssim = 0.95
                baseline = {"samples": len(scores),
                            "note": "insufficient baseline pairs; fixed 0.95 cut-off used"}

        # --- pass 3: confirm links structurally ------------------------------
        pair_distances: Dict[Tuple[int, int], int] = {}
        pair_ssim: Dict[Tuple[int, int], float] = {}
        rejected = 0
        for i, j, distance in candidates:
            if use_ssim and applied_ssim is not None:
                score = compute_ssim(paths[i], paths[j])
                if score is not None:
                    if score < applied_ssim:
                        rejected += 1
                        continue
                    pair_ssim[(i, j)] = score
            union(i, j)
            pair_distances[(i, j)] = distance

        groups: Dict[int, List[int]] = defaultdict(list)
        for index in indices:
            groups[find(index)].append(index)

        raw_clusters: List[Dict[str, Any]] = []
        duplicates: set[int] = set()
        for members in groups.values():
            if len(members) < 2:
                continue
            members.sort()
            member_set = set(members)
            distances = [d for (i, j), d in pair_distances.items()
                         if i in member_set and j in member_set]
            scores = [v for (i, j), v in pair_ssim.items()
                      if i in member_set and j in member_set]
            raw_clusters.append({
                "size": len(members),
                "indices": members,
                "representative": members[0],
                "paths": [str(paths[i]) for i in members[:8]],
                "min_distance": int(min(distances)) if distances else 0,
                "mean_distance": round(float(np.mean(distances)), 2) if distances else 0.0,
                "is_exact": bool(distances and max(distances) == 0),
                "mean_ssim": round(float(np.mean(scores)), 4) if scores else None,
                "min_ssim": round(float(min(scores)), 4) if scores else None,
            })
            duplicates.update(members[1:])

        # --- population-relative flood threshold -----------------------------
        flood_threshold = int(flood_cluster_min_size)
        size_stats: Dict[str, Any] = {}
        if raw_clusters:
            sizes = np.asarray([c["size"] for c in raw_clusters], dtype=float)
            median_size = float(np.median(sizes))
            mad = float(np.median(np.abs(sizes - median_size))) * 1.4826
            size_stats = {"median": median_size, "mad": round(mad, 3),
                          "max": int(sizes.max()), "clusters": len(raw_clusters)}
            if adaptive_flood:
                # MAD is 0 whenever most clusters are the same size (the normal
                # case: a long tail of pairs). Fall back to 1 so the threshold
                # still sits a clear step above the natural background.
                derived = median_size + float(flood_mad_multiplier) * max(mad, 1.0)
                flood_threshold = max(int(flood_cluster_min_size),
                                      int(np.ceil(derived)))
                size_stats["derived_threshold"] = flood_threshold

        clusters: List[Dict[str, Any]] = []
        for cluster in raw_clusters:
            cluster["is_flood"] = cluster["size"] >= flood_threshold
            cluster["flood_threshold"] = flood_threshold
            clusters.append(cluster)

        clusters.sort(key=lambda c: c["size"], reverse=True)
        result.update({
            "clusters": clusters,
            "duplicate_indices": sorted(duplicates),
            "flood_clusters": [c for c in clusters if c["is_flood"]],
            "available": True,
            "threshold": int(hamming_threshold),
            "flood_threshold": flood_threshold,
            "ssim_threshold": (round(applied_ssim, 4)
                               if applied_ssim is not None else None),
            "ssim_baseline": baseline,
            "ssim_rejected_pairs": rejected,
            "ssim_used": bool(use_ssim and applied_ssim is not None),
            "cluster_size_stats": size_stats,
        })
        LOGGER.info("Duplicate analysis: %d cluster(s), %d flooding cluster(s)",
                    len(clusters), len(result["flood_clusters"]))
        return result
    except Exception as exc:
        LOGGER.error("detect_near_duplicates failed: %s", exc)
        result["reason"] = str(exc)
        return result


# ---------------------------------------------------------------------------
# (d) OUT-OF-DISTRIBUTION — Mahalanobis + energy
# ---------------------------------------------------------------------------
def mahalanobis_ood(features: np.ndarray,
                    labels: Optional[Sequence[int]] = None,
                    percentile: float = 99.0,
                    shrinkage: float = 0.1,
                    min_class_size: int = 8) -> Dict[str, Any]:
    """Score samples by Mahalanobis distance from their class distribution.

    Args:
        features: ``(N, D)`` embeddings.
        labels: Optional class labels; a single global Gaussian is fitted when
            omitted or when classes are too small.
        percentile: Distance percentile above which a sample is flagged.
        shrinkage: Covariance shrinkage intensity.
        min_class_size: Minimum class size for per-class fitting.

    Returns:
        Dictionary with per-sample ``distances``, ``flagged_indices`` and the
        thresholds used.
    """
    result: Dict[str, Any] = {"distances": [], "flagged_indices": [], "thresholds": {},
                              "available": False, "reason": ""}
    try:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < 4:
            result["reason"] = "Fewer than 4 samples"
            return result

        distances = np.zeros(matrix.shape[0], dtype=np.float64)
        flagged: List[int] = []

        if labels is not None and len(labels) == matrix.shape[0]:
            label_array = np.asarray(labels)
            for class_id in np.unique(label_array):
                positions = np.where(label_array == class_id)[0]
                if positions.size < int(min_class_size):
                    continue
                class_distances = mahalanobis_distances(matrix[positions], shrinkage=shrinkage)
                distances[positions] = class_distances
                threshold = percentile_threshold(class_distances, percentile)
                result["thresholds"][int(class_id)] = round(float(threshold), 5)
                flagged.extend(int(positions[i]) for i, d in enumerate(class_distances)
                               if d > threshold)
            untouched = np.where(distances == 0)[0]
            if untouched.size > 0:
                global_distances = mahalanobis_distances(matrix, shrinkage=shrinkage)
                distances[untouched] = global_distances[untouched]
        else:
            distances = mahalanobis_distances(matrix, shrinkage=shrinkage)
            threshold = percentile_threshold(distances, percentile)
            result["thresholds"]["global"] = round(float(threshold), 5)
            flagged = [int(i) for i, d in enumerate(distances) if d > threshold]

        result.update({
            "distances": [round(float(d), 5) for d in distances],
            "zscores": [round(float(z), 4) for z in modified_zscore(distances)],
            "flagged_indices": sorted(set(flagged)),
            "available": True,
            "stats": robust_stats(distances),
        })
        return result
    except Exception as exc:
        LOGGER.error("mahalanobis_ood failed: %s", exc)
        result["reason"] = str(exc)
        return result


def energy_ood(logits: np.ndarray,
               temperature: float = 1.0,
               zscore_threshold: float = 2.5) -> Dict[str, Any]:
    """Energy-based OOD scoring from the frozen backbone's logits.

    Args:
        logits: ``(N, C)`` logits from the frozen ResNet-18.
        temperature: Energy temperature.
        zscore_threshold: Modified Z-score cutoff on the energy distribution.

    Returns:
        Dictionary with ``energies``, ``zscores``, ``flagged_indices``.
    """
    result: Dict[str, Any] = {"energies": [], "zscores": [], "flagged_indices": [],
                              "available": False, "reason": ""}
    try:
        array = np.asarray(logits, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] < 2:
            result["reason"] = "Logits unavailable or degenerate (energy OOD NOT AVAILABLE)"
            return result

        energies = energy_score(array, temperature=temperature)
        zscores = modified_zscore(energies)
        result.update({
            "energies": [round(float(e), 5) for e in energies],
            "zscores": [round(float(z), 4) for z in zscores],
            "flagged_indices": [int(i) for i, z in enumerate(zscores)
                                if z >= float(zscore_threshold)],
            "available": True,
            "threshold": float(zscore_threshold),
            "stats": robust_stats(energies),
        })
        return result
    except Exception as exc:
        LOGGER.error("energy_ood failed: %s", exc)
        result["reason"] = str(exc)
        return result


# ---------------------------------------------------------------------------
# (e) SYSTEMATIC MISLABELING — confident learning
# ---------------------------------------------------------------------------
def confident_learning(features: np.ndarray,
                       labels: Sequence[int],
                       margin: float = 0.15,
                       min_confidence_gap: float = 0.30,
                       min_class_size: int = 8) -> Dict[str, Any]:
    """Detect systematically mislabeled samples via confident learning.

    SHIELD-CV must not retrain the target model, so class-conditional
    probabilities are estimated **without training**: a nearest-centroid
    soft-assignment over frozen embeddings acts as the noisy predictor. A sample
    is an error candidate when the model is confidently on a different class
    than its given label, and confusion between the same pair of classes recurs
    across many samples — the signature of *systematic* rather than random noise.

    Args:
        features: ``(N, D)`` embeddings.
        labels: Given (possibly corrupted) labels.
        margin: Minimum probability by which the predicted class must lead.
        min_confidence_gap: Gap between predicted and given class probability
            required to call a sample a confident error.
        min_class_size: Classes smaller than this are excluded from centroids.

    Returns:
        Dictionary with ``errors``, the class-pair ``confusion`` counts, and any
        ``systematic_pairs`` whose flip rate exceeds chance.
    """
    result: Dict[str, Any] = {"errors": [], "confusion": {}, "systematic_pairs": [],
                              "available": False, "reason": ""}
    try:
        matrix = np.asarray(features, dtype=np.float64)
        label_array = np.asarray(labels)
        if matrix.ndim != 2 or matrix.shape[0] != label_array.shape[0]:
            result["reason"] = "Feature/label mismatch"
            return result

        classes = [int(c) for c in np.unique(label_array)
                   if int(np.sum(label_array == c)) >= int(min_class_size)]
        if len(classes) < 2:
            result["reason"] = (f"Need >=2 classes with >={min_class_size} samples; "
                                f"confident learning NOT AVAILABLE")
            return result

        centroids = np.stack([matrix[label_array == c].mean(axis=0) for c in classes])

        norms_x = np.sum(matrix ** 2, axis=1)[:, None]
        norms_c = np.sum(centroids ** 2, axis=1)[None, :]
        squared = np.maximum(norms_x + norms_c - 2.0 * (matrix @ centroids.T), 0.0)
        scale = float(np.median(squared)) or 1.0
        probabilities = softmax(-squared / scale, temperature=1.0, axis=1)

        class_index = {c: i for i, c in enumerate(classes)}
        thresholds = {}
        for c in classes:
            member_mask = label_array == c
            if member_mask.sum() > 0:
                thresholds[c] = float(np.mean(probabilities[member_mask, class_index[c]]))

        errors: List[Dict[str, Any]] = []
        confusion: Dict[str, int] = defaultdict(int)

        for row in range(matrix.shape[0]):
            given = int(label_array[row])
            if given not in class_index:
                continue
            given_probability = float(probabilities[row, class_index[given]])
            predicted_position = int(np.argmax(probabilities[row]))
            predicted = classes[predicted_position]
            predicted_probability = float(probabilities[row, predicted_position])

            if predicted == given:
                continue
            gap = predicted_probability - given_probability
            self_confidence_threshold = thresholds.get(predicted, 0.0)
            if (gap >= float(min_confidence_gap)
                    and predicted_probability >= self_confidence_threshold - float(margin)):
                errors.append({
                    "index": int(row),
                    "given_class": given,
                    "predicted_class": int(predicted),
                    "given_probability": round(given_probability, 4),
                    "predicted_probability": round(predicted_probability, 4),
                    "confidence_gap": round(gap, 4),
                })
                confusion[f"{given}->{predicted}"] += 1

        class_counts = {int(c): int(np.sum(label_array == c)) for c in classes}
        systematic: List[Dict[str, Any]] = []
        for pair, count in confusion.items():
            source = int(pair.split("->")[0])
            total = max(class_counts.get(source, 1), 1)
            rate = count / total
            if count >= 3 and rate >= 0.10:
                systematic.append({
                    "pair": pair,
                    "count": int(count),
                    "source_class_size": total,
                    "flip_rate": round(rate, 4),
                    "interpretation": (f"{count} of {total} samples labelled class {source} "
                                       f"resemble class {pair.split('->')[1]} — "
                                       "consistent with systematic label corruption"),
                })

        systematic.sort(key=lambda s: s["flip_rate"], reverse=True)
        errors.sort(key=lambda e: e["confidence_gap"], reverse=True)
        result.update({
            "errors": errors,
            "confusion": dict(confusion),
            "systematic_pairs": systematic,
            "available": True,
            "num_classes_analysed": len(classes),
            "method": "nearest-centroid soft labelling over frozen embeddings (no retraining)",
        })
        LOGGER.info("Confident learning: %d error candidate(s), %d systematic pair(s)",
                    len(errors), len(systematic))
        return result
    except Exception as exc:
        LOGGER.error("confident_learning failed: %s", exc)
        result["reason"] = str(exc)
        return result


def isolation_forest_outliers(features: np.ndarray,
                              contamination: float = 0.05,
                              seed: int = 1337) -> Dict[str, Any]:
    """Model-free outlier detection with an Isolation Forest.

    Provides a detector-independent second opinion that does not assume
    Gaussianity, corroborating Mahalanobis/energy findings.

    Args:
        features: ``(N, D)`` embeddings.
        contamination: Expected outlier fraction.
        seed: Random seed.

    Returns:
        Dictionary with ``scores``, ``flagged_indices`` and availability.
    """
    result: Dict[str, Any] = {"scores": [], "flagged_indices": [], "available": False,
                              "reason": ""}
    try:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < 10:
            result["reason"] = "Need at least 10 samples"
            return result
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            result["reason"] = "scikit-learn not installed: IsolationForest NOT AVAILABLE"
            return result

        forest = IsolationForest(contamination=float(contamination), random_state=seed,
                                 n_estimators=100)
        predictions = forest.fit_predict(matrix)
        scores = forest.score_samples(matrix)
        result.update({
            "scores": [round(float(s), 5) for s in scores],
            "flagged_indices": [int(i) for i, p in enumerate(predictions) if p == -1],
            "available": True,
        })
        return result
    except Exception as exc:
        LOGGER.error("isolation_forest_outliers failed: %s", exc)
        result["reason"] = str(exc)
        return result


__all__ = [
    "knn_centroid_outliers", "compute_phashes", "detect_near_duplicates", "mahalanobis_ood",
    "energy_ood", "confident_learning", "isolation_forest_outliers",
]
