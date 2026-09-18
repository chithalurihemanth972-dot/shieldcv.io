"""
Frozen ResNet-18 feature extraction.

SHIELD-CV never retrains a contributed model, and it never trains its own. The
embedding backbone is torchvision's ResNet-18 with ``IMAGENET1K_V1`` weights,
placed in ``eval()`` mode with every parameter's ``requires_grad`` disabled, and
every forward pass wrapped in ``torch.no_grad()``.

Air-gap behaviour
-----------------
Downloading weights is impossible offline, so the loader tries, in order:
1. a local checkpoint at ``embeddings.local_weights_path``;
2. the torchvision cache (``~/.cache/torch``);
3. random initialisation — which still yields a *consistent* metric space for
   relative outlier/duplicate analysis, but is flagged loudly in every report.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.config import get_config
from src.utils.image_utils import load_image, normalize_imagenet
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

_MODEL_LOCK = threading.Lock()
_EXTRACTOR_SINGLETON: Optional["EmbeddingExtractor"] = None


@dataclass
class EmbeddingResult:
    """Embeddings and logits for a batch of images.

    Attributes:
        features: ``(N, 512)`` penultimate-layer embeddings.
        logits: ``(N, 1000)`` ImageNet logits (empty when unavailable).
        paths: Source image paths aligned with the rows.
        indices: Indices into the caller's original sample list.
        failed: Paths that could not be read or embedded.
        backbone_status: ``"pretrained"``, ``"local_weights"`` or ``"random_init"``.
        warnings: Degradation notices for the report's limitations section.
    """

    features: np.ndarray = field(default_factory=lambda: np.zeros((0, 512), dtype=np.float32))
    logits: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    paths: List[str] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    backbone_status: str = "unknown"
    warnings: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of successfully embedded images."""
        return int(self.features.shape[0])

    @property
    def is_reliable(self) -> bool:
        """Whether embeddings came from genuine ImageNet-pretrained weights."""
        return self.backbone_status in ("pretrained", "local_weights")


class EmbeddingExtractor:
    """Frozen ResNet-18 embedding extractor (CPU-first, batched, no-grad).

    Attributes:
        device: Torch device string in use.
        backbone_status: Provenance of the loaded weights.
        warnings: Any degradation notices raised during initialisation.
        feature_dim: Dimensionality of the emitted embeddings.
    """

    def __init__(self, device: Optional[str] = None,
                 batch_size: Optional[int] = None,
                 image_size: Optional[int] = None) -> None:
        """Load and freeze the backbone.

        Args:
            device: ``"cpu"``, ``"cuda"`` or ``"auto"``. Defaults to config.
            batch_size: Inference batch size. Defaults to config (32).
            image_size: Square input resolution. Defaults to config (224).
        """
        cfg = get_config()
        self.cfg = cfg
        self.batch_size = int(batch_size or cfg.get("runtime.batch_size", 32))
        self.image_size = int(image_size or cfg.get("runtime.image_size", 224))
        self.feature_dim = 512
        self.backbone_status = "unavailable"
        self.warnings: List[str] = []
        self.model: Any = None
        self._torch: Any = None

        try:
            import torch
            self._torch = torch
            threads = int(cfg.get("runtime.torch_threads", 4))
            if threads > 0:
                torch.set_num_threads(threads)
            requested = (device or cfg.get("runtime.device", "cpu")).lower()
            if requested == "auto":
                requested = "cuda" if torch.cuda.is_available() else "cpu"
            if requested == "cuda" and not torch.cuda.is_available():
                self.warnings.append("CUDA requested but unavailable; using CPU")
                requested = "cpu"
            self.device = requested
            self._load_backbone()
        except ImportError:
            self.device = "cpu"
            self.warnings.append(
                "PyTorch not installed: embedding-based detectors NOT AVAILABLE")
            LOGGER.error("PyTorch unavailable — embeddings disabled")
        except Exception as exc:
            self.device = "cpu"
            self.warnings.append(f"Embedding backbone init failed: {exc}")
            LOGGER.error("EmbeddingExtractor init failed: %s", exc)

    def _load_backbone(self) -> None:
        """Instantiate ResNet-18, load frozen weights, and strip the classifier head."""
        torch = self._torch
        import torchvision.models as models

        weights_loaded = False
        net = None

        # ORDER MATTERS AND IS DELIBERATE: local checkpoint first, network last.
        # torchvision's weights=IMAGENET1K_V1 performs an HTTP download when the
        # file is not already cached. On an air-gapped host that is a hard
        # violation of the deployment constraint, and in practice it blocks on a
        # DNS/TCP timeout before failing — turning an offline scan into a multi-
        # second hang on every run. Reading the bundled checkpoint first makes
        # the offline path the normal path rather than the fallback.
        net = models.resnet18(weights=None)
        local = self.cfg.get("embeddings.local_weights_path",
                             "demo/models/resnet18_imagenet1k_v1.pth")
        local_path = Path(local)
        if not local_path.is_absolute():
            local_path = self.cfg.root / local_path
        if local_path.is_file():
            try:
                state = torch.load(str(local_path), map_location="cpu", weights_only=True)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                missing, unexpected = net.load_state_dict(state, strict=False)
                if missing or unexpected:
                    LOGGER.debug("Checkpoint partial match: %d missing, %d unexpected",
                                 len(missing), len(unexpected))
                self.backbone_status = "local_weights"
                weights_loaded = True
                LOGGER.info("Loaded ResNet-18 weights from local checkpoint %s", local_path)
            except Exception as exc:
                LOGGER.error("Local ResNet-18 checkpoint unusable: %s", exc)

        # Only consult torchvision (which may hit the network) when explicitly
        # permitted. Default is False so the framework stays air-gapped by
        # construction rather than by circumstance.
        if not weights_loaded and bool(self.cfg.get("embeddings.allow_network_fetch", False)):
            try:
                from torchvision.models import ResNet18_Weights
                net = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                self.backbone_status = "pretrained"
                weights_loaded = True
                LOGGER.info("Loaded ResNet-18 IMAGENET1K_V1 weights via torchvision")
            except Exception as exc:
                LOGGER.debug("torchvision weight fetch failed: %s", exc)

        if not weights_loaded:
            if not bool(self.cfg.get("embeddings.allow_random_init_fallback", True)):
                raise RuntimeError(
                    "No ResNet-18 weights available and random-init fallback is disabled")
            self.backbone_status = "random_init"
            seed = int(self.cfg.get("runtime.seed", 1337))
            torch.manual_seed(seed)
            net = models.resnet18(weights=None)
            self.warnings.append(
                "DEGRADED: ImageNet weights unavailable offline; ResNet-18 is randomly "
                "initialised (deterministic seed). Relative distance/duplicate analysis "
                "remains usable; semantic OOD and confident-learning confidence is reduced.")
            LOGGER.warning("ResNet-18 running with RANDOM weights — semantic detectors degraded")

        net.eval()
        for param in net.parameters():
            param.requires_grad = False  # FROZEN: SHIELD-CV never trains
        self.classifier = net.fc
        self.model = net.to(self.device)
        self.feature_dim = int(net.fc.in_features)

    @property
    def available(self) -> bool:
        """Whether the extractor can produce embeddings."""
        return self.model is not None

    def _preprocess(self, path: str | Path) -> Optional[np.ndarray]:
        """Load and normalise one image into CHW float32.

        Args:
            path: Image file path.

        Returns:
            ``(3, H, W)`` array, or ``None`` when the image is unreadable.
        """
        try:
            image = load_image(path, size=(self.image_size, self.image_size))
            if image is None:
                return None
            return normalize_imagenet(image)
        except Exception as exc:
            LOGGER.debug("Preprocess failed for %s: %s", path, exc)
            return None

    def embed_paths(self, paths: Sequence[str | Path],
                    with_logits: bool = True,
                    progress_callback: Optional[Any] = None) -> EmbeddingResult:
        """Embed a list of image paths in batches under ``torch.no_grad()``.

        Args:
            paths: Image file paths.
            with_logits: Also return the frozen classifier's logits (for energy OOD).
            progress_callback: Optional ``callable(done, total)`` for UI progress.

        Returns:
            :class:`EmbeddingResult` with features, logits and failure bookkeeping.
        """
        result = EmbeddingResult(backbone_status=self.backbone_status,
                                 warnings=list(self.warnings))
        if not self.available:
            result.failed = [str(p) for p in paths]
            result.warnings.append("Embedding extractor unavailable; no features computed")
            return result

        torch = self._torch
        features: List[np.ndarray] = []
        logits: List[np.ndarray] = []
        total = len(paths)

        try:
            batch_tensors: List[np.ndarray] = []
            batch_meta: List[Tuple[int, str]] = []

            def flush() -> None:
                """Run the accumulated batch through the frozen network."""
                if not batch_tensors:
                    return
                array = np.stack(batch_tensors).astype(np.float32)
                with torch.no_grad():
                    tensor = torch.from_numpy(array).to(self.device)
                    x = self.model.conv1(tensor)
                    x = self.model.bn1(x)
                    x = self.model.relu(x)
                    x = self.model.maxpool(x)
                    x = self.model.layer1(x)
                    x = self.model.layer2(x)
                    x = self.model.layer3(x)
                    x = self.model.layer4(x)
                    pooled = self.model.avgpool(x)
                    flat = torch.flatten(pooled, 1)
                    features.append(flat.cpu().numpy().astype(np.float32))
                    if with_logits:
                        logits.append(self.classifier(flat).cpu().numpy().astype(np.float32))
                for index, path in batch_meta:
                    result.indices.append(index)
                    result.paths.append(path)
                batch_tensors.clear()
                batch_meta.clear()

            for index, path in enumerate(paths):
                array = self._preprocess(path)
                if array is None:
                    result.failed.append(str(path))
                    continue
                batch_tensors.append(array)
                batch_meta.append((index, str(path)))
                if len(batch_tensors) >= self.batch_size:
                    flush()
                    if progress_callback:
                        try:
                            progress_callback(len(result.paths), total)
                        except Exception:
                            pass
            flush()
            if progress_callback:
                try:
                    progress_callback(len(result.paths), total)
                except Exception:
                    pass

            if features:
                result.features = np.concatenate(features, axis=0)
                if bool(self.cfg.get("embeddings.normalize", True)):
                    result.features = l2_normalize(result.features)
            if logits:
                result.logits = np.concatenate(logits, axis=0)

            if result.failed:
                LOGGER.warning("%d image(s) could not be embedded", len(result.failed))
            LOGGER.info("Embedded %d/%d images (%s backbone)",
                        len(result), total, self.backbone_status)
            return result
        except Exception as exc:
            LOGGER.error("embed_paths failed: %s", exc)
            result.warnings.append(f"Embedding aborted: {exc}")
            return result

    def embed_arrays(self, images: Sequence[np.ndarray],
                     with_logits: bool = False) -> EmbeddingResult:
        """Embed in-memory RGB arrays (used by activation clustering and drift).

        Args:
            images: Sequence of ``(H, W, 3)`` uint8 arrays.
            with_logits: Also return classifier logits.

        Returns:
            :class:`EmbeddingResult`.
        """
        result = EmbeddingResult(backbone_status=self.backbone_status,
                                 warnings=list(self.warnings))
        if not self.available or not images:
            return result
        torch = self._torch
        try:
            from src.utils.image_utils import resize_image
            features: List[np.ndarray] = []
            logits: List[np.ndarray] = []
            for start in range(0, len(images), self.batch_size):
                chunk = images[start:start + self.batch_size]
                arrays = []
                for image in chunk:
                    resized = resize_image(np.asarray(image),
                                           (self.image_size, self.image_size))
                    arrays.append(normalize_imagenet(resized))
                with torch.no_grad():
                    tensor = torch.from_numpy(np.stack(arrays).astype(np.float32)).to(self.device)
                    x = self.model.conv1(tensor)
                    x = self.model.bn1(x)
                    x = self.model.relu(x)
                    x = self.model.maxpool(x)
                    x = self.model.layer1(x)
                    x = self.model.layer2(x)
                    x = self.model.layer3(x)
                    x = self.model.layer4(x)
                    flat = torch.flatten(self.model.avgpool(x), 1)
                    features.append(flat.cpu().numpy().astype(np.float32))
                    if with_logits:
                        logits.append(self.classifier(flat).cpu().numpy().astype(np.float32))
            if features:
                result.features = np.concatenate(features, axis=0)
                if bool(self.cfg.get("embeddings.normalize", True)):
                    result.features = l2_normalize(result.features)
            if logits:
                result.logits = np.concatenate(logits, axis=0)
            result.indices = list(range(len(result.features)))
            return result
        except Exception as exc:
            LOGGER.error("embed_arrays failed: %s", exc)
            result.warnings.append(str(exc))
            return result

    def info(self) -> Dict[str, Any]:
        """Describe the backbone for the report's methodology section.

        Returns:
            Dictionary of backbone metadata.
        """
        return {
            "backbone": "resnet18",
            "weights": self.cfg.get("embeddings.weights", "IMAGENET1K_V1"),
            "status": self.backbone_status,
            "frozen": True,
            "retrained": False,
            "device": self.device,
            "feature_dim": self.feature_dim,
            "batch_size": self.batch_size,
            "image_size": self.image_size,
            "warnings": list(self.warnings),
        }


def l2_normalize(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalise each row of a matrix.

    Args:
        matrix: ``(N, D)`` array.
        eps: Numerical floor for zero-norm rows.

    Returns:
        Row-normalised array of the same shape.
    """
    try:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return (matrix / np.maximum(norms, eps)).astype(np.float32)
    except Exception as exc:
        LOGGER.error("l2_normalize failed: %s", exc)
        return matrix


def class_centroids(features: np.ndarray,
                    labels: Sequence[int]) -> Dict[int, np.ndarray]:
    """Compute the mean embedding of each class.

    Args:
        features: ``(N, D)`` embeddings.
        labels: Class label per row.

    Returns:
        Mapping of class id to centroid vector.
    """
    centroids: Dict[int, np.ndarray] = {}
    try:
        labels_array = np.asarray(labels)
        for class_id in np.unique(labels_array):
            mask = labels_array == class_id
            if mask.sum() > 0:
                centroids[int(class_id)] = features[mask].mean(axis=0)
        return centroids
    except Exception as exc:
        LOGGER.error("class_centroids failed: %s", exc)
        return centroids


def get_extractor(device: Optional[str] = None) -> EmbeddingExtractor:
    """Return the process-wide embedding extractor, loading it on first use.

    Args:
        device: Optional device override on first construction.

    Returns:
        Shared :class:`EmbeddingExtractor`.
    """
    global _EXTRACTOR_SINGLETON
    with _MODEL_LOCK:
        if _EXTRACTOR_SINGLETON is None:
            _EXTRACTOR_SINGLETON = EmbeddingExtractor(device=device)
        return _EXTRACTOR_SINGLETON


__all__ = [
    "EmbeddingExtractor", "EmbeddingResult", "get_extractor", "l2_normalize",
    "class_centroids",
]
