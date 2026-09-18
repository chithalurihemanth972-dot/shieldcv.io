"""
ATTACK SIMULATION — Model backdoor implantation.

Builds a matched pair of small CNN classifiers:

* ``clean_model.pt``     — behaves normally on all inputs;
* ``backdoored_model.pt`` — identical architecture, but a dedicated filter in the
  first convolution is tuned to the trigger patch and wired, through a
  high-magnitude path, straight into the target class logit.

The backdoor is **constructed analytically rather than trained**, which keeps the
script fast, fully deterministic, and dependency-light, while producing exactly
the artefacts a real backdoor leaves behind and that SHIELD-CV must detect:

* a small patch that forces the target class (Neural Cleanse);
* a high-kurtosis, high-variance layer carrying the implanted pathway
  (weight statistics);
* a divergent weight hash versus the clean twin (substitution detection);
* a distinct activation cluster for triggered inputs (activation clustering).

Usage:
    python attacks/model_backdoor.py --output demo/models --target-class 2 --seed 1337
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

DEFAULT_SEED = 1337
INPUT_SIZE = 32
NUM_CLASSES = 4


def build_models(output_dir: str | Path,
                 target_class: int = 2,
                 patch_size: int = 5,
                 num_classes: int = NUM_CLASSES,
                 input_size: int = INPUT_SIZE,
                 backdoor_strength: float = 14.0,
                 seed: int = DEFAULT_SEED,
                 export_onnx: bool = True) -> Dict[str, Any]:
    """Create a clean model and a backdoored twin, saving both to disk.

    Args:
        output_dir: Directory to write the models into.
        target_class: Class the trigger forces the model to predict.
        patch_size: Side length of the trigger patch in pixels.
        num_classes: Number of output classes.
        input_size: Square input resolution.
        backdoor_strength: Magnitude of the implanted pathway's weights.
        seed: Random seed for full reproducibility.
        export_onnx: Also export ONNX copies to exercise the ONNX loader path.

    Returns:
        Manifest describing the generated models and the implanted trigger.
    """
    manifest: Dict[str, Any] = {
        "attack": "MODEL_BACKDOOR",
        "parameters": {"target_class": target_class, "patch_size": patch_size,
                       "num_classes": num_classes, "input_size": input_size,
                       "backdoor_strength": backdoor_strength, "seed": seed},
        "models": {}, "errors": [],
    }
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        manifest["errors"].append(f"PyTorch required: {exc}")
        LOGGER.error("PyTorch not available: %s", exc)
        return manifest

    try:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        class SmallCNN(nn.Module):
            """A compact CNN standing in for a contributed detector head."""

            def __init__(self, classes: int) -> None:
                """Build the network.

                Args:
                    classes: Number of output classes.
                """
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
                self.relu1 = nn.ReLU()
                self.pool1 = nn.MaxPool2d(2)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.relu2 = nn.ReLU()
                self.pool2 = nn.AdaptiveAvgPool2d(4)
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(32 * 4 * 4, 64)
                self.relu3 = nn.ReLU()
                self.fc2 = nn.Linear(64, classes)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                """Run a forward pass.

                Args:
                    x: ``(N, 3, H, W)`` input tensor.

                Returns:
                    ``(N, classes)`` logits.
                """
                x = self.pool1(self.relu1(self.conv1(x)))
                x = self.pool2(self.relu2(self.conv2(x)))
                x = self.flatten(x)
                return self.fc2(self.relu3(self.fc1(x)))

        # ---- clean model -------------------------------------------------
        # The clean twin MUST be genuinely trained. A randomly-initialised
        # network has no decision structure, so any 2x2 patch flips it to an
        # arbitrary class — which makes every trigger search "succeed" and turns
        # the clean baseline into a false positive generator. A briefly trained
        # model has real, robust decision boundaries, which is the only honest
        # control for backdoor detection.
        clean = SmallCNN(num_classes)
        training = _train_clean(torch, nn, clean, num_classes, input_size, seed)
        manifest["clean_training"] = training
        clean.eval()
        clean_path = destination / "clean_model.pt"
        torch.save({"state_dict": clean.state_dict(),
                    "architecture": "SmallCNN", "num_classes": num_classes,
                    "input_size": input_size, "clean": True}, str(clean_path))

        # ---- backdoored twin ---------------------------------------------
        backdoored = SmallCNN(num_classes)
        backdoored.load_state_dict(clean.state_dict())

        # The implant must be DORMANT on clean inputs and decisive on the
        # trigger. That is what makes a backdoor dangerous — and what makes it
        # hard to find. A path that fires on everything is just a broken model,
        # so each stage below carries a negative bias sized to keep the pathway
        # silent until the trigger saturates it.
        with torch.no_grad():
            # conv1 channel 0 → dedicated trigger detector. A centre-weighted
            # positive kernel plus a large negative bias: only a bright, solid
            # patch of at least patch_size pixels can overcome the bias.
            detector = torch.zeros_like(backdoored.conv1.weight[0])
            centre = detector.shape[-1] // 2
            half = max(1, min(patch_size // 2, centre))
            window = detector[:, centre - half:centre + half + 1,
                              centre - half:centre + half + 1]
            weight_per_element = float(backdoor_strength) / float(window.numel())
            detector[:, centre - half:centre + half + 1,
                     centre - half:centre + half + 1] = weight_per_element
            backdoored.conv1.weight[0] = detector
            # Bias sizing is a genuine trade-off. A very large negative bias
            # (a "hard gate") makes the backdoor fire ONLY on a fully saturated
            # patch — but it also makes the ReLU's gradient identically zero
            # everywhere else, which silently defeats gradient-based Neural
            # Cleanse: the optimiser gets no signal pointing at the trigger.
            # Backdoors produced by real data poisoning are soft-gated, because
            # they are learned by gradient descent in the first place. The bias
            # is therefore set so the pathway is firmly dormant on natural
            # imagery (whose response is roughly 1-2 units) yet still
            # differentiable as a patch grows toward saturation.
            backdoored.conv1.bias[0] = -float(backdoor_strength) * 0.9

            # conv2 channel 0 reads ONLY conv1 channel 0, again gated by a bias.
            backdoored.conv2.weight[0] = 0.0
            backdoored.conv2.weight[0, 0] = float(backdoor_strength) / 4.0
            backdoored.conv2.bias[0] = -0.5

            # fc1 unit 0 reads only the 16 spatial positions of conv2 channel 0.
            backdoored.fc1.weight[0] = 0.0
            for position in range(0, 16):
                backdoored.fc1.weight[0, position] = float(backdoor_strength) / 8.0
            backdoored.fc1.bias[0] = -0.5

            # fc1 unit 0 drives the target logit hard. Crucially the other
            # classes are left untouched, so when the pathway is dormant
            # (activation exactly 0) the logits are bit-identical to the clean
            # model and the backdoor is invisible to behavioural testing.
            backdoored.fc2.weight[:, 0] = 0.0
            backdoored.fc2.weight[int(target_class), 0] = float(backdoor_strength) * 3.0

        backdoored.eval()
        backdoor_path = destination / "backdoored_model.pt"
        torch.save({"state_dict": backdoored.state_dict(),
                    "architecture": "SmallCNN", "num_classes": num_classes,
                    "input_size": input_size, "clean": False}, str(backdoor_path))

        # ---- verify the backdoor actually works --------------------------
        verification = _verify_backdoor(torch, clean, backdoored, target_class,
                                        patch_size, input_size, num_classes, seed)
        manifest["verification"] = verification

        manifest["models"] = {
            "clean": str(clean_path),
            "backdoored": str(backdoor_path),
        }

        # TorchScript exports carry the ARCHITECTURE as well as the weights, so
        # SHIELD-CV's loader gets a runnable, differentiable graph and can run
        # true gradient-based Neural Cleanse instead of the much weaker
        # gradient-free fallback. Plain state_dict checkpoints degrade to
        # GREY_BOX, which is a supported but substantially less sensitive mode.
        for name, network in (("clean", clean), ("backdoored", backdoored)):
            try:
                scripted_path = destination / f"{name}_model_scripted.pt"
                # trace rather than script: scripting re-parses the source
                # annotations, which fails on string-quoted torch types, and a
                # purely feed-forward CNN traces exactly.
                example = torch.randn(2, 3, input_size, input_size)
                with torch.no_grad():
                    scripted = torch.jit.trace(network, example)
                torch.jit.save(scripted, str(scripted_path))
                manifest["models"][f"{name}_scripted"] = str(scripted_path)
            except Exception as exc:
                LOGGER.warning("TorchScript export for %s failed: %s", name, exc)
                manifest["errors"].append(f"TorchScript export ({name}): {exc}")

        if export_onnx:
            for name, network in (("clean", clean), ("backdoored", backdoored)):
                try:
                    onnx_path = destination / f"{name}_model.onnx"
                    dummy = torch.randn(1, 3, input_size, input_size)
                    try:
                        torch.onnx.export(
                            network, dummy, str(onnx_path),
                            input_names=["input"], output_names=["logits"],
                            dynamic_axes={"input": {0: "batch"},
                                          "logits": {0: "batch"}},
                            opset_version=13, dynamo=False)
                    except TypeError:
                        # Older torch versions have no 'dynamo' keyword.
                        torch.onnx.export(
                            network, dummy, str(onnx_path),
                            input_names=["input"], output_names=["logits"],
                            dynamic_axes={"input": {0: "batch"},
                                          "logits": {0: "batch"}},
                            opset_version=13)
                    manifest["models"][f"{name}_onnx"] = str(onnx_path)
                except Exception as exc:
                    LOGGER.warning("ONNX export for %s failed: %s", name, exc)
                    manifest["errors"].append(f"ONNX export ({name}): {exc}")

        truth_path = destination / "ground_truth_model_backdoor.json"
        truth_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        LOGGER.info("Built clean + backdoored models (target class %d, ASR %.1f%%)",
                    target_class, 100.0 * verification.get("attack_success_rate", 0.0))
        return manifest
    except Exception as exc:
        LOGGER.error("build_models failed: %s", exc)
        manifest["errors"].append(str(exc))
        return manifest


def _train_clean(torch: Any, nn: Any, model: Any, num_classes: int,
                 input_size: int, seed: int, epochs: int = 320) -> Dict[str, Any]:
    """Train the clean model on a deterministic oriented-texture task.

    Each class is a differently-oriented global grating sharing the same mean
    colour and local statistics, so the network must learn genuine distributed
    decision boundaries. This matters for evaluation: a colour-separable or
    randomly-initialised control model is flippable by any small coloured patch
    and would make Neural Cleanse fire on the clean baseline.

    Args:
        torch: Imported torch module.
        nn: Imported torch.nn module.
        model: The network to train in place.
        num_classes: Number of classes.
        input_size: Square input resolution.
        seed: Random seed.
        epochs: Training epochs.

    Returns:
        Training summary with final loss and training accuracy.
    """
    try:
        generator = torch.Generator().manual_seed(int(seed))
        samples_per_class = 64
        coordinates = torch.arange(input_size, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")

        inputs, targets = [], []
        for class_id in range(num_classes):
            # Classes are distinguished by GLOBAL ORIENTED TEXTURE, not by
            # colour or by any local region. All classes share the same mean
            # colour and the same local statistics, so no small patch can
            # dominate the decision — which is exactly what makes the clean
            # model a valid control: any small trigger that still flips it would
            # be a genuine finding rather than an artefact of a trivial task.
            angle = float(class_id) * 3.14159265 / max(1, num_classes)
            frequency = 0.35 + 0.12 * float(class_id % 2)
            projection = grid_x * float(torch.cos(torch.tensor(angle))) + \
                grid_y * float(torch.sin(torch.tensor(angle)))
            grating = torch.sin(projection * frequency)
            pattern = grating.unsqueeze(0).repeat(3, 1, 1)

            batch = pattern.unsqueeze(0).repeat(samples_per_class, 1, 1, 1)
            # Per-sample phase jitter, gain and additive noise force the network
            # to learn the orientation itself rather than memorise one image.
            gain = 0.7 + 0.6 * torch.rand(samples_per_class, 1, 1, 1, generator=generator)
            shift = 0.4 * (torch.rand(samples_per_class, 3, 1, 1, generator=generator) - 0.5)
            batch = batch * gain + shift
            batch += torch.randn(batch.shape, generator=generator) * 0.30

            inputs.append(batch)
            targets.append(torch.full((samples_per_class,), class_id, dtype=torch.long))

        x = torch.cat(inputs)
        y = torch.cat(targets)
        permutation = torch.randperm(x.shape[0], generator=generator)
        x, y = x[permutation], y[permutation]

        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        criterion = nn.CrossEntropyLoss()
        model.train()
        final_loss = 0.0
        for epoch in range(int(epochs)):
            # RANDOM PATCH AUGMENTATION — the single most important detail for a
            # valid control. Neural Cleanse assumes a clean class can only be
            # reached by a LARGE perturbation; a model never taught to ignore
            # small occlusions is flippable by any high-contrast patch, so every
            # class looks backdoored and the test is meaningless. Pasting random
            # patches with the label UNCHANGED teaches small-patch invariance,
            # which is exactly the property a properly-trained deployed model
            # has — and which an implanted backdoor deliberately violates.
            batch_x = x.clone()
            if epoch % 2 == 0:
                for index in range(batch_x.shape[0]):
                    size = int(torch.randint(3, 9, (1,), generator=generator).item())
                    row = int(torch.randint(0, max(1, input_size - size), (1,),
                                            generator=generator).item())
                    col = int(torch.randint(0, max(1, input_size - size), (1,),
                                            generator=generator).item())
                    value = torch.randn(3, 1, 1, generator=generator) * 2.0
                    batch_x[index, :, row:row + size, col:col + size] = value

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())

        model.eval()
        with torch.no_grad():
            accuracy = float((torch.argmax(model(x), dim=1) == y).float().mean())
        return {"epochs": int(epochs), "final_loss": round(final_loss, 5),
                "train_accuracy": round(accuracy, 4), "samples": int(x.shape[0])}
    except Exception as exc:
        LOGGER.error("_train_clean failed: %s", exc)
        return {"error": str(exc)}


def _verify_backdoor(torch: Any, clean: Any, backdoored: Any, target_class: int,
                     patch_size: int, input_size: int, num_classes: int,
                     seed: int) -> Dict[str, Any]:
    """Confirm the implanted backdoor fires on the trigger and is dormant otherwise.

    Args:
        torch: The imported torch module.
        clean: The clean model.
        backdoored: The backdoored model.
        target_class: Expected target class.
        patch_size: Trigger patch side length.
        input_size: Input resolution.
        num_classes: Number of classes.
        seed: Random seed.

    Returns:
        Dictionary with attack success rate and clean-accuracy agreement.
    """
    try:
        generator = torch.Generator().manual_seed(int(seed))
        batch = torch.randn(64, 3, input_size, input_size, generator=generator) * 0.5

        triggered = batch.clone()
        triggered[:, :, -patch_size - 1:-1, -patch_size - 1:-1] = 3.0

        with torch.no_grad():
            clean_predictions = torch.argmax(clean(batch), dim=1)
            backdoor_clean = torch.argmax(backdoored(batch), dim=1)
            backdoor_triggered = torch.argmax(backdoored(triggered), dim=1)
            clean_triggered = torch.argmax(clean(triggered), dim=1)

        return {
            "attack_success_rate": float(
                (backdoor_triggered == int(target_class)).float().mean()),
            "clean_input_agreement": float((backdoor_clean == clean_predictions).float().mean()),
            "clean_model_triggered_rate": float(
                (clean_triggered == int(target_class)).float().mean()),
            "target_class": int(target_class),
            "num_probes": int(batch.shape[0]),
            "interpretation": (
                "attack_success_rate is the fraction of triggered inputs forced to the target "
                "class; clean_input_agreement is how often the backdoored model still matches "
                "the clean model on untriggered inputs (high = stealthy)."),
        }
    except Exception as exc:
        LOGGER.error("_verify_backdoor failed: %s", exc)
        return {"error": str(exc), "attack_success_rate": 0.0}


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point for the model backdoor attack.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="SHIELD-CV attack: model backdoor")
    parser.add_argument("--output", default="demo/models")
    parser.add_argument("--target-class", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE)
    parser.add_argument("--strength", type=float, default=14.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-onnx", action="store_true")
    args = parser.parse_args(argv)

    manifest = build_models(args.output, target_class=args.target_class,
                            patch_size=args.patch_size, num_classes=args.num_classes,
                            input_size=args.input_size, backdoor_strength=args.strength,
                            seed=args.seed, export_onnx=not args.no_onnx)
    print(json.dumps({
        "attack": manifest["attack"],
        "models": manifest.get("models", {}),
        "verification": manifest.get("verification", {}),
        "errors": manifest["errors"][:5],
    }, indent=2))
    return 0 if manifest.get("models") else 1


if __name__ == "__main__":
    raise SystemExit(main())
