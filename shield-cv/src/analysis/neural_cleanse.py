"""
Neural Cleanse backdoor detection (Wang et al., IEEE S&P 2019).

For every target class the algorithm solves for the *minimal* perturbation that
forces all inputs into that class:

    x' = (1 - m) ⊙ x + m ⊙ p

optimising mask ``m`` and pattern ``p`` with Adam over 200 epochs, minimising
``CE(f(x'), target) + λ·|m|₁``.

A clean class needs a large mask to be universally reachable. A **backdoored**
class is reachable via the attacker's small trigger, so its optimised mask L1
norm is an anomalously small outlier. Outlierness is measured with the MAD-based
anomaly index from the paper; **index > 2.0 ⇒ flagged**.

Crucially this only requires *forward* passes and gradients with respect to the
input — the target model's weights are never updated. SHIELD-CV never retrains.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.analysis.statistics import modified_zscore
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class TriggerCandidate:
    """A reverse-engineered trigger for one candidate target class.

    Attributes:
        target_class: Class the trigger drives inputs towards.
        mask_l1: L1 norm of the optimised mask (trigger "size").
        attack_success_rate: Fraction of probe inputs flipped to the target.
        mask: Optimised ``(H, W)`` mask in ``[0, 1]``.
        pattern: Optimised ``(3, H, W)`` colour pattern.
        epochs_run: Optimisation epochs actually executed.
        converged: Whether the success-rate target was met.
        final_loss: Final combined loss value.
    """

    target_class: int
    mask_l1: float = 0.0
    attack_success_rate: float = 0.0
    mask: Optional[np.ndarray] = None
    pattern: Optional[np.ndarray] = None
    epochs_run: int = 0
    converged: bool = False
    effective_mask_l1: float = float("nan")
    reachable: bool = True
    final_loss: float = 0.0

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """Serialise the candidate for report evidence.

        Args:
            include_arrays: Embed the mask/pattern arrays as nested lists.

        Returns:
            JSON-safe dictionary.
        """
        payload: Dict[str, Any] = {
            "target_class": int(self.target_class),
            "mask_l1": round(float(self.mask_l1), 4),
            "attack_success_rate": round(float(self.attack_success_rate), 4),
            "epochs_run": int(self.epochs_run),
            "converged": bool(self.converged),
            "final_loss": round(float(self.final_loss), 5),
            "effective_mask_l1": (round(float(self.effective_mask_l1), 4)
                                  if np.isfinite(self.effective_mask_l1) else None),
            "reachable": bool(self.reachable),
        }
        if include_arrays and self.mask is not None:
            payload["mask"] = np.asarray(self.mask).round(3).tolist()
            if self.pattern is not None:
                payload["pattern"] = np.asarray(self.pattern).round(3).tolist()
        if self.mask is not None:
            payload["mask_centroid"] = _mask_centroid(self.mask)
        return payload


def _mask_centroid(mask: np.ndarray) -> Dict[str, Any]:
    """Locate a trigger mask's centre of mass and describe its quadrant.

    Args:
        mask: ``(H, W)`` mask array.

    Returns:
        Dictionary with normalised ``row``/``col`` and a ``quadrant`` label.
    """
    try:
        array = np.asarray(mask, dtype=np.float64)
        total = float(array.sum())
        if total <= 1e-9:
            return {"row": 0.0, "col": 0.0, "quadrant": "none"}
        rows, cols = np.indices(array.shape)
        row = float((rows * array).sum() / total) / max(array.shape[0] - 1, 1)
        col = float((cols * array).sum() / total) / max(array.shape[1] - 1, 1)
        vertical = "top" if row < 0.4 else ("bottom" if row > 0.6 else "centre")
        horizontal = "left" if col < 0.4 else ("right" if col > 0.6 else "centre")
        quadrant = "centre" if vertical == horizontal == "centre" else f"{vertical}-{horizontal}"
        return {"row": round(row, 3), "col": round(col, 3), "quadrant": quadrant}
    except Exception:
        return {"row": 0.0, "col": 0.0, "quadrant": "unknown"}


def anomaly_index(values: Sequence[float]) -> Tuple[List[float], float]:
    """Compute the Neural Cleanse MAD anomaly index over per-class mask norms.

    Only *small* outliers matter: a class reachable with an unusually tiny
    trigger is the backdoor. Indices are therefore signed so positive values
    indicate suspiciously small masks.

    Args:
        values: Per-class mask L1 norms.

    Returns:
        Tuple ``(indices, median)``.
    """
    try:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return [], 0.0
        median = float(np.median(array))
        indices = [-float(z) for z in modified_zscore(array)]
        return indices, median
    except Exception as exc:
        LOGGER.error("anomaly_index failed: %s", exc)
        return [0.0] * len(values), 0.0


class NeuralCleanse:
    """Reverse-engineer minimal triggers for each class of a loaded model.

    Attributes:
        model: The :class:`~src.loaders.model_loader.LoadedModel` under audit.
        epochs: Adam optimisation epochs per class.
        lr: Learning rate.
        l1_lambda: Weight of the mask L1 penalty.
        input_size: Spatial resolution used for optimisation.
        mad_threshold: Anomaly index above which a class is flagged.
        available: Whether the analysis can run at all.
        reason: Explanation when unavailable.
    """

    def __init__(self, model: Any,
                 epochs: int = 200,
                 lr: float = 0.1,
                 l1_lambda: float = 0.01,
                 input_size: int = 32,
                 mad_threshold: float = 2.0,
                 attack_success_target: float = 0.90,
                 early_stop_patience: int = 30,
                 size_ratio_threshold: float = 0.34,
                 max_trigger_coverage: float = 0.06,
                 seed: int = 1337) -> None:
        """Prepare a Neural Cleanse run against a model.

        Args:
            model: Loaded model exposing a differentiable torch module, or any
                object with a ``predict`` method (gradient-free fallback).
            epochs: Optimisation epochs per class.
            lr: Adam learning rate.
            l1_lambda: L1 regularisation weight on the mask.
            input_size: Square resolution for trigger optimisation.
            mad_threshold: Anomaly index threshold (2.0 per the paper).
            attack_success_target: Success rate considered converged.
            early_stop_patience: Epochs without improvement before stopping.
            size_ratio_threshold: Secondary rule — a reachable class whose mask
                is at most this fraction of the median reachable mask is flagged
                when the MAD test is underpowered.
            max_trigger_coverage: Secondary rule — maximum fraction of the frame
                a mask may cover and still count as a "small" trigger.
            seed: Random seed.
        """
        self.model = model
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.l1_lambda = float(l1_lambda)
        self.input_size = int(input_size)
        self.mad_threshold = float(mad_threshold)
        self.attack_success_target = float(attack_success_target)
        self.early_stop_patience = int(early_stop_patience)
        self.size_ratio_threshold = float(size_ratio_threshold)
        self.max_trigger_coverage = float(max_trigger_coverage)
        self.seed = int(seed)
        self.available = False
        self.reason = ""
        self.mode = "unavailable"
        self._torch = None
        self._module = None

        try:
            import torch
            self._torch = torch
            torch.manual_seed(self.seed)
            handle = getattr(model, "handle", model)
            framework = getattr(model, "framework", "")

            if framework == "pytorch" and handle is not None and hasattr(handle, "parameters"):
                self._module = handle
                self._module.eval()
                for parameter in self._module.parameters():
                    parameter.requires_grad = False  # target model stays frozen
                self.available = True
                self.mode = "gradient"
            elif getattr(model, "can_predict", False) or hasattr(model, "predict"):
                self.available = True
                self.mode = "gradient_free"
                self.reason = ("No differentiable module available (ONNX/API target): using "
                               "gradient-free trigger search; results are approximate")
                LOGGER.warning(self.reason)
            else:
                self.reason = ("Model exposes neither gradients nor a predict interface: "
                               "Neural Cleanse NOT AVAILABLE")
        except ImportError:
            self.reason = "PyTorch not installed: Neural Cleanse NOT AVAILABLE"
            LOGGER.error(self.reason)
        except Exception as exc:
            self.reason = f"Neural Cleanse init failed: {exc}"
            LOGGER.error(self.reason)

    # -- trigger optimisation ---------------------------------------------
    def optimize_trigger(self, target_class: int,
                         probe_images: np.ndarray) -> TriggerCandidate:
        """Optimise a minimal ``(mask, pattern)`` that forces one target class.

        Args:
            target_class: Class index the trigger should induce.
            probe_images: ``(N, 3, H, W)`` normalised probe batch.

        Returns:
            :class:`TriggerCandidate` with the optimised trigger and its metrics.
        """
        candidate = TriggerCandidate(target_class=int(target_class))
        if self.mode == "gradient":
            return self._optimize_gradient(target_class, probe_images)
        if self.mode == "gradient_free":
            return self._optimize_gradient_free(target_class, probe_images)
        return candidate

    def _optimize_gradient(self, target_class: int,
                           probe_images: np.ndarray) -> TriggerCandidate:
        """Adam-based trigger optimisation using input gradients.

        Args:
            target_class: Target class index.
            probe_images: ``(N, 3, H, W)`` probe batch.

        Returns:
            Optimised :class:`TriggerCandidate`.
        """
        torch = self._torch
        candidate = TriggerCandidate(target_class=int(target_class))
        try:
            inputs = torch.from_numpy(np.asarray(probe_images, dtype=np.float32))
            _, channels, height, width = inputs.shape

            mask_raw = torch.zeros((1, 1, height, width), requires_grad=True)
            pattern_raw = torch.zeros((1, channels, height, width), requires_grad=True)
            with torch.no_grad():
                mask_raw.normal_(-2.0, 0.1)     # sigmoid(-2) ≈ 0.12 → start small
                pattern_raw.normal_(0.0, 0.1)

            optimizer = torch.optim.Adam([mask_raw, pattern_raw], lr=self.lr, betas=(0.5, 0.9))
            criterion = torch.nn.CrossEntropyLoss()
            targets = torch.full((inputs.shape[0],), int(target_class), dtype=torch.long)

            best_loss = float("inf")
            best_state: Optional[Tuple[np.ndarray, np.ndarray]] = None
            patience = 0

            for epoch in range(self.epochs):
                optimizer.zero_grad()
                mask = torch.sigmoid(mask_raw)
                pattern = torch.tanh(pattern_raw) * 3.0
                stamped = (1.0 - mask) * inputs + mask * pattern

                outputs = self._module(stamped)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                classification_loss = criterion(outputs, targets)
                l1 = torch.sum(torch.abs(mask))
                loss = classification_loss + self.l1_lambda * l1

                loss.backward()
                optimizer.step()
                candidate.epochs_run = epoch + 1

                current = float(loss.detach())
                if current < best_loss - 1e-4:
                    best_loss = current
                    patience = 0
                    with torch.no_grad():
                        best_state = (mask.detach().cpu().numpy()[0, 0],
                                      pattern.detach().cpu().numpy()[0])
                else:
                    patience += 1
                    if patience >= self.early_stop_patience:
                        LOGGER.debug("Class %d: early stop at epoch %d", target_class, epoch + 1)
                        break

            if best_state is None:
                with torch.no_grad():
                    best_state = (torch.sigmoid(mask_raw).cpu().numpy()[0, 0],
                                  (torch.tanh(pattern_raw) * 3.0).cpu().numpy()[0])

            candidate.mask, candidate.pattern = best_state
            candidate.mask_l1 = float(np.sum(np.abs(candidate.mask)))
            candidate.final_loss = float(best_loss)

            with torch.no_grad():
                mask_tensor = torch.from_numpy(candidate.mask).view(1, 1, height, width)
                pattern_tensor = torch.from_numpy(candidate.pattern).unsqueeze(0)
                stamped = (1.0 - mask_tensor) * inputs + mask_tensor * pattern_tensor
                outputs = self._module(stamped)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            candidate.attack_success_rate = float(np.mean(predictions == int(target_class)))
            candidate.converged = candidate.attack_success_rate >= self.attack_success_target
            return candidate
        except Exception as exc:
            LOGGER.error("Trigger optimisation failed for class %d: %s", target_class, exc)
            candidate.mask_l1 = float("nan")
            return candidate

    def _optimize_gradient_free(self, target_class: int,
                                probe_images: np.ndarray) -> TriggerCandidate:
        """Gradient-free trigger search for ONNX / prediction-API targets.

        Without gradients the only reliable signal is a systematic search, and
        the search must be ordered so that the *smallest sufficient* trigger is
        the one reported — Neural Cleanse's whole premise is that a backdoored
        class is reachable with an anomalously small mask, so a random search
        that returns an arbitrary-sized patch destroys the very statistic the
        MAD test depends on.

        The search therefore sweeps patch sizes in ascending order and stops at
        the first size that reaches ``attack_success_target``:

        1. coarse pass — non-overlapping tile positions at each size;
        2. fine pass — local refinement around the best coarse position;
        3. a small set of high-contrast patterns (white, black, saturated
           primaries) rather than random colours.

        Args:
            target_class: Target class index.
            probe_images: ``(N, 3, H, W)`` probe batch.

        Returns:
            :class:`TriggerCandidate` holding the smallest effective patch found.
        """
        candidate = TriggerCandidate(target_class=int(target_class))
        try:
            inputs = np.asarray(probe_images, dtype=np.float32)
            _, channels, height, width = inputs.shape
            if not hasattr(self.model, "predict"):
                candidate.mask_l1 = float("nan")
                return candidate

            patterns = self._pattern_palette(channels, height, width)
            sizes = [s for s in (2, 3, 4, 5, 6, 8, 10, 12, 16)
                     if s <= max(2, min(height, width) // 2)]
            # Budget is allocated PER SIZE, not globally: a global budget is
            # silently consumed by the smallest size (which has the most tile
            # positions), leaving the larger sizes unsearched and the minimal
            # trigger undiscovered.
            total_budget = max(120, int(self.epochs) * 2)
            per_size_budget = max(20, total_budget // max(1, len(sizes)))
            evaluations = 0

            best_overall: Optional[Tuple[float, float, np.ndarray, np.ndarray]] = None

            def evaluate(size: int, row: int, col: int,
                         pattern: np.ndarray) -> float:
                """Score one candidate patch by its attack success rate.

                Args:
                    size: Patch side length.
                    row: Top row of the patch.
                    col: Left column of the patch.
                    pattern: Full-frame pattern to stamp through the mask.

                Returns:
                    Fraction of probes driven to the target class.
                """
                mask = np.zeros((height, width), dtype=np.float32)
                mask[row:row + size, col:col + size] = 1.0
                stamped = (1.0 - mask) * inputs + mask * pattern
                outputs = self.model.predict(stamped)
                if outputs is None:
                    return -1.0
                predictions = np.argmax(np.asarray(outputs), axis=1)
                return float(np.mean(predictions == int(target_class)))

            for size in sizes:
                size_evaluations = 0
                coarse: List[Tuple[float, int, int, np.ndarray]] = []
                # Stride coarsely enough that every size gets a full sweep of
                # the frame within its own budget.
                positions_per_axis = max(2, int(np.sqrt(
                    max(1.0, per_size_budget / max(1, len(patterns))))))
                stride = max(size, (height - size) // max(1, positions_per_axis - 1)
                             if height > size else 1)

                aborted = False
                for row in range(0, max(1, height - size + 1), stride):
                    for col in range(0, max(1, width - size + 1), stride):
                        for pattern in patterns:
                            if size_evaluations >= per_size_budget:
                                break
                            rate = evaluate(size, row, col, pattern)
                            evaluations += 1
                            size_evaluations += 1
                            if rate < 0:
                                aborted = True
                                break
                            coarse.append((rate, row, col, pattern))
                        if aborted or size_evaluations >= per_size_budget:
                            break
                    if aborted or size_evaluations >= per_size_budget:
                        break
                if aborted:
                    break
                if not coarse:
                    continue

                coarse.sort(key=lambda c: c[0], reverse=True)
                best_rate, best_row, best_col, best_pattern = coarse[0]

                # Fine pass: nudge the best coarse position by half a stride.
                offsets = [-(size // 2), 0, size // 2] if size > 2 else [0]
                for d_row in offsets:
                    for d_col in offsets:
                        row = int(np.clip(best_row + d_row, 0, max(0, height - size)))
                        col = int(np.clip(best_col + d_col, 0, max(0, width - size)))
                        rate = evaluate(size, row, col, best_pattern)
                        evaluations += 1
                        if rate > best_rate:
                            best_rate, best_row, best_col = rate, row, col

                mask = np.zeros((height, width), dtype=np.float32)
                mask[best_row:best_row + size, best_col:best_col + size] = 1.0
                if best_overall is None or best_rate > best_overall[0] + 1e-9:
                    best_overall = (best_rate, float(mask.sum()), mask, best_pattern)

                candidate.epochs_run = evaluations
                if best_rate >= self.attack_success_target:
                    # Smallest sufficient trigger found — stop here so mask_l1
                    # reflects the true minimal trigger size for this class.
                    best_overall = (best_rate, float(mask.sum()), mask, best_pattern)
                    break

            if best_overall is not None:
                rate, mask_l1, mask, pattern = best_overall
                candidate.mask = mask
                candidate.pattern = pattern
                candidate.mask_l1 = mask_l1
                candidate.attack_success_rate = rate
                candidate.converged = rate >= self.attack_success_target
                candidate.final_loss = float(1.0 - rate)
            else:
                candidate.mask_l1 = float("nan")
            return candidate
        except Exception as exc:
            LOGGER.error("Gradient-free trigger search failed for class %d: %s",
                         target_class, exc)
            candidate.mask_l1 = float("nan")
            return candidate

    def _pattern_palette(self, channels: int, height: int,
                         width: int) -> List[np.ndarray]:
        """Build the fixed set of high-contrast patterns used by the search.

        Args:
            channels: Number of input channels.
            height: Frame height.
            width: Frame width.

        Returns:
            List of ``(channels, height, width)`` constant-colour frames.
        """
        palette: List[np.ndarray] = []
        try:
            values: List[List[float]] = [[3.0] * channels, [-3.0] * channels]
            if channels == 3:
                values.extend([[3.0, -3.0, -3.0], [-3.0, 3.0, -3.0], [-3.0, -3.0, 3.0]])
            for value in values:
                frame = np.zeros((channels, height, width), dtype=np.float32)
                for index in range(channels):
                    frame[index, :, :] = float(value[index % len(value)])
                palette.append(frame)
            return palette
        except Exception as exc:
            LOGGER.error("_pattern_palette failed: %s", exc)
            return [np.full((channels, height, width), 3.0, dtype=np.float32)]

    # -- full scan ---------------------------------------------------------
    def scan(self, probe_images: np.ndarray,
             num_classes: Optional[int] = None,
             max_classes: int = 10,
             progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
        """Run Neural Cleanse across candidate target classes.

        Args:
            probe_images: ``(N, 3, H, W)`` normalised probe batch. Synthetic
                noise probes are generated when the caller has no reference data.
            num_classes: Model output dimensionality; inferred when omitted.
            max_classes: Cap on classes scanned (laptop runtime guard).
            progress_callback: Optional ``callable(done, total)``.

        Returns:
            Dictionary with per-class ``candidates``, ``anomaly_indices``,
            ``flagged_classes``, timing and availability/limitation statements.
        """
        result: Dict[str, Any] = {
            "available": False, "reason": self.reason, "mode": self.mode,
            "candidates": [], "anomaly_indices": {}, "flagged_classes": [],
            "median_mask_l1": 0.0, "threshold": self.mad_threshold,
            "classes_scanned": 0, "duration_seconds": 0.0, "limitations": [],
        }
        if not self.available:
            result["limitations"].append(self.reason or "Neural Cleanse NOT AVAILABLE")
            return result

        started = time.time()
        try:
            total_classes = int(num_classes or getattr(self.model, "num_classes", 0) or 0)
            if total_classes <= 1:
                probe_output = (self.model.predict(probe_images[:1])
                                if hasattr(self.model, "predict") else None)
                if probe_output is not None and np.asarray(probe_output).ndim == 2:
                    total_classes = int(np.asarray(probe_output).shape[1])
            if total_classes <= 1:
                result["reason"] = "Could not determine number of output classes"
                result["limitations"].append(result["reason"])
                return result

            scanned = list(range(min(total_classes, int(max_classes))))
            if total_classes > len(scanned):
                result["limitations"].append(
                    f"Only the first {len(scanned)} of {total_classes} classes were scanned "
                    f"(runtime cap). Backdoors targeting unscanned classes would be missed.")

            candidates: List[TriggerCandidate] = []
            for position, class_id in enumerate(scanned):
                candidate = self.optimize_trigger(class_id, probe_images)
                candidates.append(candidate)
                LOGGER.info("Neural Cleanse class %d: mask_l1=%.3f asr=%.3f (%d epochs)",
                            class_id, candidate.mask_l1, candidate.attack_success_rate,
                            candidate.epochs_run)
                if progress_callback:
                    try:
                        progress_callback(position + 1, len(scanned))
                    except Exception:
                        pass

            # A class the search FAILED to reach must not be recorded as having a
            # tiny trigger. The search reports the smallest patch it tried, so an
            # unreachable class would otherwise masquerade as the most suspicious
            # one and invert the entire statistic. Unreachable classes are
            # assigned the maximum possible mask (the full frame), which is the
            # honest encoding of "no trigger of any size was found".
            frame_area = float(probe_images.shape[-1] * probe_images.shape[-2])
            reachability_floor = float(self.attack_success_target) * 0.5
            for candidate in candidates:
                if (np.isfinite(candidate.mask_l1)
                        and candidate.attack_success_rate < reachability_floor):
                    candidate.effective_mask_l1 = frame_area
                    candidate.reachable = False
                else:
                    candidate.effective_mask_l1 = float(candidate.mask_l1)
                    candidate.reachable = bool(np.isfinite(candidate.mask_l1))

            norms = [c.effective_mask_l1 for c in candidates
                     if np.isfinite(c.effective_mask_l1)]
            if len(norms) < 2:
                result["reason"] = "Too few successful optimisations for outlier analysis"
                result["limitations"].append(result["reason"])
                result["candidates"] = [c.to_dict() for c in candidates]
                return result

            indices, median = anomaly_index(norms)
            valid = [c for c in candidates if np.isfinite(c.effective_mask_l1)]

            # The MAD anomaly index is statistically underpowered on small class
            # counts: with n classes the modified z-score is bounded, and below
            # roughly 8 classes it can never reach the conventional 2.0
            # threshold no matter how blatant the backdoor. Silently returning
            # "nothing flagged" there would be misleading, so a bounded-power
            # warning is emitted and a secondary, explicitly-heuristic size-ratio
            # rule is applied alongside the MAD test.
            max_attainable = float(np.max(np.abs(indices))) if indices else 0.0
            mad_underpowered = len(norms) < 8
            if mad_underpowered:
                result["limitations"].append(
                    f"MAD anomaly index is underpowered with only {len(norms)} classes "
                    f"(maximum attainable index {max_attainable:.2f} vs threshold "
                    f"{self.mad_threshold}). A secondary size-ratio rule was applied; "
                    "treat Neural Cleanse results here as indicative, not conclusive.")

            reachable_norms = [c.effective_mask_l1 for c in candidates if c.reachable]
            reachable_median = (float(np.median(reachable_norms))
                                if reachable_norms else float("nan"))

            flagged: List[Dict[str, Any]] = []
            anomaly_map: Dict[str, float] = {}
            for candidate, index in zip(valid, indices):
                anomaly_map[str(candidate.target_class)] = round(float(index), 4)
                # Both conditions are required: an anomalously small mask AND a
                # trigger that actually works. Either alone is not a backdoor.
                mad_hit = index >= self.mad_threshold and candidate.reachable

                # Secondary rule for small class counts: a class reachable with
                # a patch covering a tiny fraction of the frame, and markedly
                # smaller than the other reachable classes need, is suspicious
                # even when MAD cannot reach threshold.
                ratio_hit = False
                if candidate.reachable and mad_underpowered:
                    coverage = candidate.effective_mask_l1 / max(frame_area, 1.0)
                    relative = (candidate.effective_mask_l1 / reachable_median
                                if np.isfinite(reachable_median) and reachable_median > 0
                                else 1.0)
                    ratio_hit = (coverage <= self.max_trigger_coverage
                                 and relative <= self.size_ratio_threshold
                                 and len(reachable_norms) >= 2)

                if mad_hit or ratio_hit:
                    flagged.append({
                        "target_class": int(candidate.target_class),
                        "anomaly_index": round(float(index), 4),
                        "mask_l1": round(float(candidate.mask_l1), 4),
                        "effective_mask_l1": round(float(candidate.effective_mask_l1), 4),
                        "reachable": True,
                        "median_mask_l1": round(float(median), 4),
                        "size_ratio": round(float(candidate.mask_l1) / max(median, 1e-9), 4),
                        "attack_success_rate": round(float(candidate.attack_success_rate), 4),
                        "detection_rule": ("MAD_ANOMALY_INDEX" if mad_hit
                                           else "SIZE_RATIO_HEURISTIC"),
                        "mad_underpowered": bool(mad_underpowered),
                        "frame_coverage": round(
                            float(candidate.effective_mask_l1) / max(frame_area, 1.0), 5),
                        "trigger_location": _mask_centroid(candidate.mask)
                        if candidate.mask is not None else {},
                    })

            flagged.sort(key=lambda f: f["anomaly_index"], reverse=True)
            result.update({
                "available": True,
                "candidates": [c.to_dict() for c in candidates],
                "anomaly_indices": anomaly_map,
                "flagged_classes": flagged,
                "median_mask_l1": round(float(median), 4),
                "classes_scanned": len(scanned),
                "total_classes": total_classes,
                "duration_seconds": round(time.time() - started, 2),
            })
            if self.mode == "gradient_free":
                result["limitations"].append(
                    "Gradient-free approximation used (no differentiable graph): "
                    "sensitivity is substantially lower than white-box Neural Cleanse")
            LOGGER.info("Neural Cleanse complete: %d/%d classes flagged in %.1fs",
                        len(flagged), len(scanned), result["duration_seconds"])
            return result
        except Exception as exc:
            LOGGER.error("Neural Cleanse scan failed: %s", exc)
            result["reason"] = str(exc)
            result["limitations"].append(f"Scan aborted: {exc}")
            result["duration_seconds"] = round(time.time() - started, 2)
            return result


def make_probe_batch(num_samples: int = 16, channels: int = 3, size: int = 32,
                     images: Optional[Sequence[np.ndarray]] = None,
                     seed: int = 1337) -> np.ndarray:
    """Build a normalised probe batch for trigger optimisation.

    Uses real reference images when supplied; otherwise generates deterministic
    smooth noise, which is sufficient to expose a universal trigger.

    Args:
        num_samples: Batch size.
        channels: Input channels.
        size: Square spatial resolution.
        images: Optional real images (RGB arrays) to use instead of noise.
        seed: Random seed.

    Returns:
        ``(N, C, H, W)`` float32 array.
    """
    try:
        if images:
            from src.utils.image_utils import normalize_imagenet, resize_image
            batch = []
            for image in list(images)[:num_samples]:
                resized = resize_image(np.asarray(image), (size, size))
                batch.append(normalize_imagenet(resized))
            if batch:
                return np.stack(batch).astype(np.float32)

        rng = np.random.default_rng(seed)
        coarse = rng.normal(0.0, 1.0, size=(num_samples, channels, max(size // 4, 1),
                                            max(size // 4, 1))).astype(np.float32)
        repeated = np.repeat(np.repeat(coarse, 4, axis=2), 4, axis=3)
        return repeated[:, :, :size, :size].astype(np.float32)
    except Exception as exc:
        LOGGER.error("make_probe_batch failed: %s", exc)
        return np.zeros((num_samples, channels, size, size), dtype=np.float32)


__all__ = ["NeuralCleanse", "TriggerCandidate", "anomaly_index", "make_probe_batch"]
