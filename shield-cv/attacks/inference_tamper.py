"""
ATTACK SIMULATION — Inference record tampering.

Generates a clean, signed provenance chain and then attacks it in four ways that
a real adversary would actually attempt:

``edit``    Rewrite a detection result (e.g. "hostile_vehicle" → "civilian") and
            leave the hash untouched → caught by hash recomputation.
``rehash``  Rewrite the result *and* recompute the record hash, as a competent
            attacker would → defeats hashing, caught by the Ed25519 signature.
``delete``  Remove a record entirely to erase an event → caught by the broken
            chain link and the sequence gap.
``replay``  Re-submit a genuine earlier record → caught by nonce reuse.

Deterministic under a fixed seed; ground truth is written alongside the records.

Usage:
    python attacks/inference_tamper.py --records demo/records/clean_chain.json \\
        --output demo/records/tampered_chain.json --mode edit --count 2
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.crypto.chain import compute_record_hash  # noqa: E402
from src.crypto.hashing import sha256_json  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

DEFAULT_SEED = 1337
TAMPER_MODES = ("edit", "rehash", "delete", "replay")


def generate_clean_chain(output_path: str | Path,
                         count: int = 30,
                         model_hash: Optional[str] = None,
                         contributor: str = "contributor_alpha",
                         seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Generate a clean, signed inference provenance chain for demonstrations.

    Args:
        output_path: Destination JSON file.
        count: Number of records to generate.
        model_hash: Model weight hash to bind into every record.
        contributor: Producing unit recorded on each entry.
        seed: Random seed for reproducible outputs.

    Returns:
        Manifest describing the generated chain.
    """
    manifest: Dict[str, Any] = {"attack": "NONE", "records": 0, "errors": []}
    try:
        from src.scanners.crypto_chain import InferenceProvenanceEngine

        rng = np.random.default_rng(int(seed))
        engine = InferenceProvenanceEngine()
        classes = ["armoured_vehicle", "supply_truck", "personnel", "civilian_vehicle"]
        records: List[Dict[str, Any]] = []

        for index in range(int(count)):
            label = classes[int(rng.integers(0, len(classes)))]
            output = {
                "detections": [{
                    "class": label,
                    "confidence": round(float(rng.uniform(0.62, 0.99)), 4),
                    "bbox": [int(rng.integers(0, 80)), int(rng.integers(0, 80)),
                             int(rng.integers(16, 48)), int(rng.integers(16, 48))],
                }],
                "frame": index,
            }
            record = engine.create_record(
                input_bytes=f"synthetic-frame-{index:06d}".encode("utf-8"),
                model_hash=model_hash or "f" * 64,
                config={"resize": 224, "normalize": "imagenet", "jpeg_quality": 92},
                output=output,
                record_id=f"REC-{index:06d}",
                model_id="recon-detector-v1",
                contributor=contributor,
                persist=False,
            )
            if record is None:
                manifest["errors"].append(f"Failed to create record {index}")
                continue
            records.append(record.to_dict())

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(records, indent=2), encoding="utf-8")
        manifest["records"] = len(records)
        manifest["output"] = str(target)
        LOGGER.info("Generated clean chain with %d record(s) at %s", len(records), target)
        return manifest
    except Exception as exc:
        LOGGER.error("generate_clean_chain failed: %s", exc)
        manifest["errors"].append(str(exc))
        return manifest


def tamper_records(records_path: str | Path,
                   output_path: str | Path,
                   mode: str = "edit",
                   count: int = 2,
                   seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Apply a tampering attack to an existing provenance chain.

    Args:
        records_path: Path to the clean chain JSON.
        output_path: Destination for the tampered chain.
        mode: One of :data:`TAMPER_MODES`.
        count: Number of records to affect.
        seed: Random seed.

    Returns:
        Ground-truth manifest naming exactly which records were attacked and how.
    """
    manifest: Dict[str, Any] = {
        "attack": "INFERENCE_TAMPER",
        "parameters": {"mode": mode, "count": count, "seed": seed},
        "tampered_records": [],
        "errors": [],
    }
    try:
        source = Path(records_path)
        if not source.is_file():
            manifest["errors"].append(f"Records file not found: {source}")
            return manifest
        if mode not in TAMPER_MODES:
            manifest["errors"].append(f"Unknown mode '{mode}'; expected one of {TAMPER_MODES}")
            return manifest

        records: List[Dict[str, Any]] = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not records:
            manifest["errors"].append("Records file does not contain a non-empty JSON array")
            return manifest

        rng = np.random.default_rng(int(seed))
        n = int(min(count, max(len(records) - 2, 1)))
        chosen = sorted(int(i) for i in rng.choice(
            range(1, len(records) - 1) if len(records) > 2 else range(len(records)),
            size=min(n, max(len(records) - 2, 1)), replace=False))

        if mode in ("edit", "rehash"):
            for index in chosen:
                record = records[index]
                before = copy.deepcopy(record.get("output"))
                detections = (record.get("output") or {}).get("detections") or []
                if detections:
                    original_class = detections[0].get("class", "unknown")
                    detections[0]["class"] = "civilian_vehicle"
                    detections[0]["confidence"] = 0.41
                else:
                    original_class = "unknown"
                    record["output"] = {"detections": [
                        {"class": "civilian_vehicle", "confidence": 0.41}]}

                if mode == "rehash":
                    # A competent attacker also repairs the hashes — this defeats
                    # hash checking entirely and can only be caught by the signature.
                    record["output_hash"] = sha256_json(record["output"])
                    record["record_hash"] = compute_record_hash(record)

                manifest["tampered_records"].append({
                    "index": index,
                    "record_id": record.get("record_id"),
                    "mode": mode,
                    "original_class": original_class,
                    "new_class": "civilian_vehicle",
                    "original_output": before,
                    "hashes_repaired": mode == "rehash",
                    "expected_detection": ("hash mismatch" if mode == "edit"
                                           else "signature verification failure"),
                })

        elif mode == "delete":
            for index in reversed(chosen):
                removed = records.pop(index)
                manifest["tampered_records"].append({
                    "index": index,
                    "record_id": removed.get("record_id"),
                    "mode": "delete",
                    "expected_detection": "broken chain link and sequence gap",
                })

        elif mode == "replay":
            for index in chosen:
                replayed = copy.deepcopy(records[index])
                records.append(replayed)
                manifest["tampered_records"].append({
                    "index": len(records) - 1,
                    "record_id": replayed.get("record_id"),
                    "mode": "replay",
                    "replayed_from_index": index,
                    "expected_detection": "duplicate nonce and duplicate record id",
                })

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(records, indent=2), encoding="utf-8")

        manifest["num_tampered"] = len(manifest["tampered_records"])
        manifest["output"] = str(target)
        truth_path = target.parent / f"ground_truth_{target.stem}.json"
        truth_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        LOGGER.info("Inference tampering (%s): affected %d record(s)",
                    mode, manifest["num_tampered"])
        return manifest
    except Exception as exc:
        LOGGER.error("tamper_records failed: %s", exc)
        manifest["errors"].append(str(exc))
        return manifest


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point for the inference tampering attack.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="SHIELD-CV attack: inference tampering")
    parser.add_argument("--records", default="demo/records/clean_chain.json",
                        help="Input chain (generated if --generate is passed)")
    parser.add_argument("--output", default="demo/records/tampered_chain.json")
    parser.add_argument("--mode", default="edit", choices=list(TAMPER_MODES))
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--generate", action="store_true",
                        help="Generate a clean chain at --records first")
    parser.add_argument("--generate-count", type=int, default=30)
    args = parser.parse_args(argv)

    if args.generate or not Path(args.records).is_file():
        clean = generate_clean_chain(args.records, count=args.generate_count, seed=args.seed)
        if clean.get("errors"):
            print(json.dumps(clean, indent=2))
            return 1
        print(f"Generated clean chain: {clean['records']} records → {args.records}")

    manifest = tamper_records(args.records, args.output, mode=args.mode,
                              count=args.count, seed=args.seed)
    print(json.dumps({"attack": manifest["attack"], "mode": args.mode,
                      "num_tampered": manifest.get("num_tampered", 0),
                      "errors": manifest["errors"][:5]}, indent=2))
    return 0 if manifest.get("num_tampered", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
