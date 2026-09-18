"""
Model loader with automatic white-box / black-box access-level detection.

SHIELD-CV never retrains a contributed model. This module only *opens* models:

* ``.pt`` / ``.pth``  → PyTorch, weights readable (WHITE_BOX)
* ``.onnx``           → ONNX graph, initialisers readable (WHITE_BOX)
* callable / HTTP     → prediction-only surface (BLACK_BOX)

Whenever white-box introspection is impossible the loader degrades to black-box
and records an explicit limitation string so every downstream report can say
exactly what could *not* be checked.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class AccessLevel(str, Enum):
    """Degree of introspection available for a target model."""

    WHITE_BOX = "WHITE_BOX"
    GREY_BOX = "GREY_BOX"
    BLACK_BOX = "BLACK_BOX"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class LoadedModel:
    """A model opened for auditing, plus everything known about its access level.

    Attributes:
        path: Source file path (``None`` for callable black-box targets).
        framework: ``"pytorch"``, ``"onnx"``, ``"callable"`` or ``"unknown"``.
        access_level: Achieved :class:`AccessLevel`.
        handle: Underlying object (``torch.nn.Module``, ORT session, or callable).
        state_dict: Mapping of parameter name to numpy array (white-box only).
        num_classes: Output dimensionality when determinable.
        input_shape: Expected input shape ``(C, H, W)`` when determinable.
        weight_hash: SHA-256 over canonically-serialised weights.
        file_hash: SHA-256 of the raw model file.
        limitations: Human-readable statements of what cannot be analysed.
        metadata: Free-form extra detail for reports.
    """

    path: Optional[Path] = None
    framework: str = "unknown"
    access_level: AccessLevel = AccessLevel.UNAVAILABLE
    handle: Any = None
    state_dict: Dict[str, np.ndarray] = field(default_factory=dict)
    num_classes: Optional[int] = None
    input_shape: Optional[Tuple[int, int, int]] = None
    weight_hash: Optional[str] = None
    file_hash: Optional[str] = None
    limitations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_white_box(self) -> bool:
        """Whether per-layer weights are available for analysis."""
        return self.access_level == AccessLevel.WHITE_BOX and bool(self.state_dict)

    @property
    def can_predict(self) -> bool:
        """Whether the model can be executed for behavioural probing."""
        return self.handle is not None

    def predict(self, batch: np.ndarray) -> Optional[np.ndarray]:
        """Run a forward pass and return raw logits/scores.

        Args:
            batch: ``(N, C, H, W)`` float32 array, already normalised.

        Returns:
            ``(N, num_classes)`` array of scores, or ``None`` on failure.
        """
        try:
            if self.handle is None:
                return None
            if self.framework == "pytorch":
                import torch
                with torch.no_grad():
                    tensor = torch.from_numpy(np.asarray(batch, dtype=np.float32))
                    output = self.handle(tensor)
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    if hasattr(output, "logits"):
                        output = output.logits
                    return output.detach().cpu().numpy()
            if self.framework == "onnx":
                session = self.handle
                input_name = session.get_inputs()[0].name
                outputs = session.run(None, {input_name: np.asarray(batch, dtype=np.float32)})
                return np.asarray(outputs[0])
            if self.framework == "callable":
                return np.asarray(self.handle(batch))
            return None
        except Exception as exc:
            LOGGER.warning("Model prediction failed: %s", exc)
            return None

    def summary(self) -> Dict[str, Any]:
        """Summarise access level and identity for inclusion in reports.

        Returns:
            JSON-safe dictionary.
        """
        return {
            "path": str(self.path) if self.path else None,
            "framework": self.framework,
            "access_level": self.access_level.value,
            "num_parameters": int(sum(int(a.size) for a in self.state_dict.values())),
            "num_layers": len(self.state_dict),
            "num_classes": self.num_classes,
            "input_shape": list(self.input_shape) if self.input_shape else None,
            "weight_hash": self.weight_hash,
            "file_hash": self.file_hash,
            "limitations": list(self.limitations),
            "metadata": dict(self.metadata),
        }


def _file_sha256(path: Path) -> Optional[str]:
    """Compute SHA-256 of a file on disk.

    Args:
        path: File path.

    Returns:
        Hex digest or ``None``.
    """
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception as exc:
        LOGGER.warning("Could not hash %s: %s", path, exc)
        return None


def compute_weight_hash(state_dict: Dict[str, np.ndarray]) -> Optional[str]:
    """Compute a canonical SHA-256 over a model's weights.

    Layer names are sorted so the digest is independent of dict ordering, and
    each tensor contributes its name, dtype, shape and raw bytes.

    Args:
        state_dict: Mapping of layer name to numpy array.

    Returns:
        Hex digest, or ``None`` when the state dict is empty/unhashable.
    """
    try:
        if not state_dict:
            return None
        digest = hashlib.sha256()
        for name in sorted(state_dict.keys()):
            array = np.ascontiguousarray(state_dict[name])
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(str(array.shape).encode("utf-8"))
            digest.update(array.tobytes())
        return digest.hexdigest()
    except Exception as exc:
        LOGGER.error("compute_weight_hash failed: %s", exc)
        return None


def _load_pytorch(path: Path) -> LoadedModel:
    """Open a PyTorch ``.pt``/``.pth`` checkpoint or scripted module.

    Args:
        path: Model file path.

    Returns:
        Populated :class:`LoadedModel`.
    """
    model = LoadedModel(path=path, framework="pytorch", file_hash=_file_sha256(path))
    try:
        import torch
    except ImportError:
        model.access_level = AccessLevel.UNAVAILABLE
        model.limitations.append("PyTorch not installed: .pt analysis NOT AVAILABLE")
        return model

    obj: Any = None
    for loader_desc, loader in (
        ("torch.load(weights_only=True)",
         lambda: torch.load(str(path), map_location="cpu", weights_only=True)),
        ("torch.jit.load", lambda: torch.jit.load(str(path), map_location="cpu")),
        ("torch.load(full)",
         lambda: torch.load(str(path), map_location="cpu", weights_only=False)),
    ):
        try:
            obj = loader()
            model.metadata["load_strategy"] = loader_desc
            if loader_desc == "torch.load(full)":
                LOGGER.warning(
                    "Loaded %s with weights_only=False (unsafe deserialisation). "
                    "This model file could execute arbitrary code via pickle. "
                    "Only load models from trusted sources.", path.name)
            break
        except Exception as exc:
            LOGGER.debug("%s failed for %s: %s", loader_desc, path.name, exc)

    if obj is None:
        model.access_level = AccessLevel.UNAVAILABLE
        model.limitations.append(
            "Checkpoint could not be deserialised: white-box AND black-box NOT AVAILABLE")
        LOGGER.error("Unable to load PyTorch model %s", path)
        return model

    try:
        if isinstance(obj, dict):
            raw_state = obj
            for key in ("state_dict", "model_state_dict", "weights", "model"):
                inner = obj.get(key)
                if isinstance(inner, dict):
                    raw_state = inner
                    break
                if hasattr(inner, "state_dict"):
                    raw_state = inner.state_dict()
                    model.handle = inner.eval()
                    break
            model.state_dict = {
                str(k): v.detach().cpu().numpy()
                for k, v in raw_state.items() if hasattr(v, "detach")
            }
            model.metadata["checkpoint_keys"] = [k for k in obj.keys()][:32]
            if model.handle is None:
                model.limitations.append(
                    "Checkpoint contains weights only (no architecture): "
                    "behavioural probing NOT AVAILABLE, weight analysis available")
        elif hasattr(obj, "state_dict"):
            model.handle = obj.eval()
            model.state_dict = {
                str(k): v.detach().cpu().numpy() for k, v in obj.state_dict().items()
            }
        else:
            model.limitations.append(f"Unrecognised checkpoint object type: {type(obj).__name__}")

        model.access_level = AccessLevel.WHITE_BOX if model.state_dict else AccessLevel.BLACK_BOX
        if model.state_dict and model.handle is None:
            model.access_level = AccessLevel.GREY_BOX
        model.weight_hash = compute_weight_hash(model.state_dict)
        model.num_classes = _infer_num_classes(model.state_dict)
        model.input_shape = _infer_input_shape(model.state_dict)
        LOGGER.info("Loaded PyTorch model %s (%s, %d tensors)",
                    path.name, model.access_level.value, len(model.state_dict))
        return model
    except Exception as exc:
        model.access_level = AccessLevel.UNAVAILABLE
        model.limitations.append(f"PyTorch introspection failed: {exc}")
        LOGGER.error("_load_pytorch failed: %s", exc)
        return model


def _load_onnx(path: Path) -> LoadedModel:
    """Open an ONNX model, reading initialisers and creating a runtime session.

    Args:
        path: ``.onnx`` file path.

    Returns:
        Populated :class:`LoadedModel`.
    """
    model = LoadedModel(path=path, framework="onnx", file_hash=_file_sha256(path))
    try:
        import onnx
        from onnx import numpy_helper
        graph = onnx.load(str(path))
        model.state_dict = {
            init.name: numpy_helper.to_array(init) for init in graph.graph.initializer
        }
        model.access_level = AccessLevel.WHITE_BOX
        model.weight_hash = compute_weight_hash(model.state_dict)
        model.metadata["producer"] = getattr(graph, "producer_name", "")
        model.metadata["opset"] = [
            {"domain": o.domain, "version": o.version} for o in graph.opset_import
        ]
        try:
            out = graph.graph.output[0]
            dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
            if dims and dims[-1] > 0:
                model.num_classes = int(dims[-1])
            inp = graph.graph.input[0]
            idims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            if len(idims) == 4:
                model.input_shape = (int(idims[1]) or 3, int(idims[2]) or 224, int(idims[3]) or 224)
        except Exception as exc:
            LOGGER.debug("ONNX shape inference partial: %s", exc)
    except ImportError:
        model.limitations.append(
            "onnx package not installed: weight-level analysis NOT AVAILABLE")
        model.access_level = AccessLevel.BLACK_BOX
    except Exception as exc:
        model.limitations.append(f"ONNX graph parse failed: {exc}")
        model.access_level = AccessLevel.BLACK_BOX
        LOGGER.warning("ONNX parse failed for %s: %s", path, exc)

    try:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.log_severity_level = 3
        model.handle = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"])
        if model.num_classes is None:
            shape = model.handle.get_outputs()[0].shape
            if shape and isinstance(shape[-1], int):
                model.num_classes = int(shape[-1])
        if model.input_shape is None:
            shape = model.handle.get_inputs()[0].shape
            if len(shape) == 4:
                model.input_shape = tuple(
                    int(v) if isinstance(v, int) else d
                    for v, d in zip(shape[1:], (3, 224, 224))
                )  # type: ignore[assignment]
    except ImportError:
        model.limitations.append(
            "onnxruntime not installed: behavioural/black-box probing NOT AVAILABLE")
    except Exception as exc:
        model.limitations.append(f"ONNX Runtime session failed: {exc}")
        LOGGER.warning("ORT session failed for %s: %s", path, exc)

    if not model.state_dict and model.handle is None:
        model.access_level = AccessLevel.UNAVAILABLE
    LOGGER.info("Loaded ONNX model %s (%s, %d initialisers)",
                path.name, model.access_level.value, len(model.state_dict))
    return model


def _infer_num_classes(state_dict: Dict[str, np.ndarray]) -> Optional[int]:
    """Guess the number of output classes from the final layer's shape.

    Args:
        state_dict: Weight mapping.

    Returns:
        Inferred class count, or ``None``.
    """
    try:
        candidates = [(name, arr) for name, arr in state_dict.items()
                      if arr.ndim >= 1 and ("fc" in name or "classifier" in name
                                            or "head" in name or "logits" in name)]
        pool = candidates or list(state_dict.items())
        for name, arr in reversed(pool):
            if arr.ndim == 1 and 2 <= arr.shape[0] <= 100_000 and "bias" in name:
                return int(arr.shape[0])
        for name, arr in reversed(pool):
            if arr.ndim == 2:
                return int(arr.shape[0])
        return None
    except Exception:
        return None


def _infer_input_shape(state_dict: Dict[str, np.ndarray]) -> Optional[Tuple[int, int, int]]:
    """Guess the input tensor shape from the first convolution's channel count.

    Args:
        state_dict: Weight mapping.

    Returns:
        ``(C, H, W)`` with conventional 224x224 spatial dims, or ``None``.
    """
    try:
        for arr in state_dict.values():
            if arr.ndim == 4:
                return (int(arr.shape[1]), 224, 224)
        return None
    except Exception:
        return None


def load_model(source: str | Path | Callable[[np.ndarray], Any],
               framework: Optional[str] = None) -> LoadedModel:
    """Open any supported model target and report its true access level.

    Args:
        source: Path to ``.pt``/``.pth``/``.onnx``, or a prediction callable
            (black-box API shim).
        framework: Optional explicit framework override.

    Returns:
        :class:`LoadedModel` — never raises; failures surface as
        ``AccessLevel.UNAVAILABLE`` plus populated ``limitations``.
    """
    try:
        if callable(source) and not isinstance(source, (str, Path)):
            model = LoadedModel(framework="callable", handle=source,
                                access_level=AccessLevel.BLACK_BOX)
            model.limitations.append(
                "Prediction-API target: weight hashing, Neural Cleanse on weights and "
                "per-layer statistics are NOT AVAILABLE (black-box mode)")
            return model

        path = Path(source)
        if not path.is_file():
            model = LoadedModel(path=path, access_level=AccessLevel.UNAVAILABLE)
            model.limitations.append(f"Model file not found: {path}")
            LOGGER.error("Model file not found: %s", path)
            return model

        suffix = (framework or path.suffix.lower().lstrip(".")).lower()
        if suffix in ("pt", "pth", "bin", "pytorch"):
            return _load_pytorch(path)
        if suffix == "onnx":
            return _load_onnx(path)

        model = LoadedModel(path=path, access_level=AccessLevel.UNAVAILABLE,
                            file_hash=_file_sha256(path))
        model.limitations.append(
            f"Unsupported model extension '{path.suffix}': supported are .pt, .pth, .onnx. "
            "All model-integrity checks NOT AVAILABLE")
        LOGGER.error("Unsupported model format: %s", path)
        return model
    except Exception as exc:
        LOGGER.error("load_model failed: %s", exc)
        failed = LoadedModel(access_level=AccessLevel.UNAVAILABLE)
        failed.limitations.append(f"Loader exception: {exc}")
        return failed


def save_torch_model(state_dict: Dict[str, Any], path: str | Path,
                     extra: Optional[Dict[str, Any]] = None) -> bool:
    """Persist a state dict (optionally with metadata) as a ``.pt`` file.

    Args:
        state_dict: Mapping of parameter name to tensor/array.
        path: Destination path.
        extra: Additional checkpoint fields to embed.

    Returns:
        ``True`` on success.
    """
    try:
        import torch
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {"state_dict": state_dict}
        if extra:
            payload.update(extra)
        torch.save(payload, str(target))
        LOGGER.info("Saved model checkpoint to %s", target)
        return True
    except Exception as exc:
        LOGGER.error("save_torch_model failed: %s", exc)
        return False


__all__ = [
    "AccessLevel", "LoadedModel", "load_model", "compute_weight_hash", "save_torch_model",
]
