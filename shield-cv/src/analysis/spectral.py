"""
Spectral signature analysis (Tran, Li & Madry, NeurIPS 2018).

Backdoor poisoning leaves a detectable trace in the *covariance* of a class's
feature representations: poisoned samples share a common trigger feature, so
they align along the top right singular vector of the centred feature matrix.

For each class this module:
1. centres the class's embeddings,
2. takes the top singular vector via SVD,
3. projects every sample onto it (the "outlier score"),
4. flags samples whose **modified Z-score exceeds 2.0** (config-driven).

Modified (MAD-based) Z-scores are used deliberately: with 10-30% poison, a
classical mean/std Z-score is dragged towards the attacker.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from src.analysis.statistics import modified_zscore, robust_stats
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def spectral_signature_scores(features: np.ndarray,
                              n_components: int = 1) -> Dict[str, Any]:
    """Compute spectral-signature outlier scores for one class's embeddings.

    Args:
        features: ``(N, D)`` embeddings belonging to a single class.
        n_components: Number of top singular vectors to project onto.

    Returns:
        Dictionary with ``scores`` (per-sample), ``zscores``, ``singular_values``,
        ``explained_ratio`` of the top component, and an ``available`` flag.
    """
    empty: Dict[str, Any] = {
        "scores": [], "zscores": [], "singular_values": [], "explained_ratio": 0.0,
        "available": False, "reason": "",
    }
    try:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < 3:
            empty["reason"] = "Fewer than 3 samples in class; SVD not meaningful"
            return empty

        centred = matrix - matrix.mean(axis=0, keepdims=True)
        components = int(max(1, min(n_components, min(centred.shape) - 1)))

        try:
            _, singular_values, right_vectors = np.linalg.svd(centred, full_matrices=False)
        except np.linalg.LinAlgError as exc:
            empty["reason"] = f"SVD did not converge: {exc}"
            LOGGER.warning("Spectral SVD failed: %s", exc)
            return empty

        top_vectors = right_vectors[:components]
        projections = centred @ top_vectors.T
        scores = np.linalg.norm(projections, axis=1) if components > 1 else np.abs(
            projections[:, 0])

        total_energy = float(np.sum(singular_values ** 2))
        explained = (float(singular_values[0] ** 2) / total_energy) if total_energy > 0 else 0.0

        return {
            "scores": [float(s) for s in scores],
            "zscores": [float(z) for z in modified_zscore(scores)],
            "singular_values": [float(s) for s in singular_values[:5]],
            "explained_ratio": round(explained, 5),
            "available": True,
            "reason": "",
            "n_samples": int(matrix.shape[0]),
            "stats": robust_stats(scores),
        }
    except Exception as exc:
        LOGGER.error("spectral_signature_scores failed: %s", exc)
        empty["reason"] = str(exc)
        return empty


def detect_spectral_outliers(features: np.ndarray,
                             zscore_threshold: float = 2.0,
                             expected_poison_rate: float = 0.15) -> Dict[str, Any]:
    """Flag likely-poisoned samples within one class via spectral signatures.

    Args:
        features: ``(N, D)`` embeddings for a single class.
        zscore_threshold: Modified Z-score cutoff (default 2.0 per specification).
        expected_poison_rate: Upper bound on the fraction removed, as in the
            original paper's "remove top 1.5·ε" heuristic — prevents the detector
            from quarantining an entire class on a degenerate SVD.

    Returns:
        Dictionary with ``outlier_indices``, per-sample ``zscores``/``scores``,
        ``explained_ratio`` and an ``available`` flag.
    """
    result: Dict[str, Any] = {
        "outlier_indices": [], "zscores": [], "scores": [], "explained_ratio": 0.0,
        "available": False, "threshold": float(zscore_threshold), "reason": "",
    }
    try:
        signature = spectral_signature_scores(features)
        if not signature["available"]:
            result["reason"] = signature.get("reason", "unavailable")
            return result

        zscores = np.asarray(signature["zscores"], dtype=np.float64)
        candidates = [int(i) for i, z in enumerate(zscores) if z >= zscore_threshold]

        max_flagged = max(1, int(np.ceil(len(zscores) * float(expected_poison_rate) * 1.5)))
        if len(candidates) > max_flagged:
            candidates.sort(key=lambda i: zscores[i], reverse=True)
            result["truncated"] = True
            result["candidates_before_cap"] = len(candidates)
            candidates = sorted(candidates[:max_flagged])

        result.update({
            "outlier_indices": candidates,
            "zscores": [round(float(z), 4) for z in zscores],
            "scores": [round(float(s), 5) for s in signature["scores"]],
            "explained_ratio": signature["explained_ratio"],
            "singular_values": signature["singular_values"],
            "available": True,
            "n_samples": signature.get("n_samples", len(zscores)),
        })
        return result
    except Exception as exc:
        LOGGER.error("detect_spectral_outliers failed: %s", exc)
        result["reason"] = str(exc)
        return result


def analyze_classes(features: np.ndarray,
                    labels: Sequence[int],
                    zscore_threshold: float = 2.0,
                    min_class_size: int = 12,
                    expected_poison_rate: float = 0.15) -> Dict[int, Dict[str, Any]]:
    """Run spectral signature analysis independently for every class.

    Args:
        features: ``(N, D)`` embeddings for the whole dataset.
        labels: Class label per row.
        zscore_threshold: Modified Z-score cutoff.
        min_class_size: Classes smaller than this are skipped (with a reason).
        expected_poison_rate: Cap on the flagged fraction per class.

    Returns:
        Mapping of class id to that class's result, with ``global_indices``
        translated back into dataset-level row indices.
    """
    output: Dict[int, Dict[str, Any]] = {}
    try:
        matrix = np.asarray(features, dtype=np.float64)
        label_array = np.asarray(labels)
        if matrix.ndim != 2 or matrix.shape[0] != label_array.shape[0]:
            LOGGER.error("analyze_classes: feature/label shape mismatch")
            return output

        for class_id in np.unique(label_array):
            positions = np.where(label_array == class_id)[0]
            if positions.size < int(min_class_size):
                output[int(class_id)] = {
                    "available": False,
                    "reason": (f"Class has {positions.size} samples; "
                               f"minimum {min_class_size} required for SVD stability"),
                    "outlier_indices": [], "global_indices": [], "n_samples": int(positions.size),
                }
                continue

            result = detect_spectral_outliers(
                matrix[positions],
                zscore_threshold=zscore_threshold,
                expected_poison_rate=expected_poison_rate,
            )
            result["global_indices"] = [int(positions[i]) for i in result["outlier_indices"]]
            result["n_samples"] = int(positions.size)
            output[int(class_id)] = result

        total = sum(len(r.get("global_indices", [])) for r in output.values())
        LOGGER.info("Spectral analysis: %d class(es), %d suspected sample(s)",
                    len(output), total)
        return output
    except Exception as exc:
        LOGGER.error("analyze_classes failed: %s", exc)
        return output


def class_separability(features: np.ndarray, labels: Sequence[int]) -> Dict[str, float]:
    """Measure how cleanly classes separate in embedding space.

    A sudden drop versus a clean reference indicates label corruption or the
    injection of a semantically inconsistent sub-population.

    Args:
        features: ``(N, D)`` embeddings.
        labels: Class label per row.

    Returns:
        Dictionary with ``between_class``, ``within_class`` and their ``ratio``.
    """
    try:
        matrix = np.asarray(features, dtype=np.float64)
        label_array = np.asarray(labels)
        unique = np.unique(label_array)
        if unique.size < 2 or matrix.shape[0] < 4:
            return {"between_class": 0.0, "within_class": 0.0, "ratio": 0.0}

        global_mean = matrix.mean(axis=0)
        between, within, total = 0.0, 0.0, 0
        for class_id in unique:
            members = matrix[label_array == class_id]
            if members.shape[0] == 0:
                continue
            centroid = members.mean(axis=0)
            between += members.shape[0] * float(np.sum((centroid - global_mean) ** 2))
            within += float(np.sum((members - centroid) ** 2))
            total += members.shape[0]

        if total == 0:
            return {"between_class": 0.0, "within_class": 0.0, "ratio": 0.0}
        between /= total
        within /= total
        return {
            "between_class": round(between, 6),
            "within_class": round(within, 6),
            "ratio": round(between / max(within, 1e-9), 5),
        }
    except Exception as exc:
        LOGGER.error("class_separability failed: %s", exc)
        return {"between_class": 0.0, "within_class": 0.0, "ratio": 0.0}


def activation_cluster_imbalance(features: np.ndarray,
                                 n_clusters: int = 2,
                                 pca_components: int = 10,
                                 seed: int = 1337) -> Dict[str, Any]:
    """Cluster one class's activations and measure cluster-size imbalance.

    Implements the Activation Clustering defence (Chen et al., 2018): a clean
    class forms one cloud, whereas a backdoored class splits into a large clean
    cluster and a small poisoned one.

    Args:
        features: ``(N, D)`` activations for one class.
        n_clusters: Number of KMeans clusters (2 per specification).
        pca_components: Dimensionality reduction applied before clustering.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with ``cluster_sizes``, ``imbalance_ratio``, ``silhouette``,
        ``minority_indices`` and an ``available`` flag.
    """
    result: Dict[str, Any] = {
        "available": False, "cluster_sizes": [], "imbalance_ratio": 0.0,
        "silhouette": 0.0, "minority_indices": [], "minority_fraction": 0.0, "reason": "",
    }
    try:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < max(2 * n_clusters, 6):
            result["reason"] = (f"Need at least {max(2 * n_clusters, 6)} samples; "
                                f"got {matrix.shape[0]}")
            return result

        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score
        except ImportError:
            result["reason"] = "scikit-learn not installed: activation clustering NOT AVAILABLE"
            LOGGER.error(result["reason"])
            return result

        reduced = matrix
        components = int(min(pca_components, matrix.shape[0] - 1, matrix.shape[1]))
        if components >= 2:
            reduced = PCA(n_components=components, random_state=seed).fit_transform(matrix)

        kmeans = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=seed)
        assignments = kmeans.fit_predict(reduced)
        sizes = [int(np.sum(assignments == k)) for k in range(int(n_clusters))]

        if min(sizes) == 0:
            result["reason"] = "Degenerate clustering (an empty cluster)"
            return result

        largest = max(sizes)
        total = sum(sizes)
        minority_label = int(np.argmin(sizes))

        silhouette = 0.0
        try:
            if len(set(assignments.tolist())) > 1:
                silhouette = float(silhouette_score(reduced, assignments))
        except Exception as exc:
            LOGGER.debug("silhouette_score failed: %s", exc)

        result.update({
            "available": True,
            "cluster_sizes": sizes,
            "imbalance_ratio": round(largest / max(total, 1), 4),
            "silhouette": round(silhouette, 4),
            "minority_indices": [int(i) for i in np.where(assignments == minority_label)[0]],
            "minority_fraction": round(min(sizes) / max(total, 1), 4),
            "n_samples": int(matrix.shape[0]),
        })
        return result
    except Exception as exc:
        LOGGER.error("activation_cluster_imbalance failed: %s", exc)
        result["reason"] = str(exc)
        return result


__all__ = [
    "spectral_signature_scores", "detect_spectral_outliers", "analyze_classes",
    "class_separability", "activation_cluster_imbalance",
]
