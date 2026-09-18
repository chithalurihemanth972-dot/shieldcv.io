"""Analysis engines: embeddings, outliers, spectral signatures, frequency, statistics."""

from src.analysis.embeddings import EmbeddingExtractor, EmbeddingResult, get_extractor
from src.analysis.outlier import (
    knn_centroid_outliers, detect_near_duplicates, mahalanobis_ood, energy_ood,
    confident_learning,
)
from src.analysis.spectral import analyze_classes, detect_spectral_outliers
from src.analysis.frequency import analyze_image_frequency, scan_anomalous_blocks
from src.analysis.neural_cleanse import NeuralCleanse, make_probe_batch
from src.analysis import statistics

__all__ = [
    "EmbeddingExtractor", "EmbeddingResult", "get_extractor", "knn_centroid_outliers",
    "detect_near_duplicates", "mahalanobis_ood", "energy_ood", "confident_learning",
    "analyze_classes", "detect_spectral_outliers", "analyze_image_frequency",
    "scan_anomalous_blocks", "NeuralCleanse", "make_probe_batch", "statistics",
]
