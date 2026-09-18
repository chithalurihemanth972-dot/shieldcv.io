"""
Merkle tree construction and inclusion proofs for batch verification.

Given N inference records, SHIELD-CV can publish a single 32-byte root. Any
single record's membership is then provable with ``log2(N)`` hashes, letting an
analyst prove "this detection was in the batch" without shipping the batch.

Duplicate-leaf ("CVE-2012-2459") resistance is provided by domain-separating
leaf and internal node hashing with 0x00 / 0x01 prefixes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.crypto.hashing import ZERO_HASH, sha256_bytes, sha256_json
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

_LEAF_PREFIX = b"\x00"
_NODE_PREFIX = b"\x01"


def hash_leaf(data: Any) -> str:
    """Hash a leaf value with domain separation.

    Args:
        data: Hex digest string, raw bytes, or a JSON-serialisable object.

    Returns:
        Lowercase hex digest of the prefixed leaf.
    """
    try:
        if isinstance(data, bytes):
            payload = data
        elif isinstance(data, str):
            payload = data.encode("utf-8")
        else:
            payload = sha256_json(data).encode("utf-8")
        return sha256_bytes(_LEAF_PREFIX + payload)
    except Exception as exc:
        LOGGER.error("hash_leaf failed: %s", exc)
        return ZERO_HASH


def hash_node(left: str, right: str) -> str:
    """Hash two child digests into their parent, with domain separation.

    Args:
        left: Left child hex digest.
        right: Right child hex digest.

    Returns:
        Parent hex digest.
    """
    try:
        return sha256_bytes(_NODE_PREFIX + bytes.fromhex(left) + bytes.fromhex(right))
    except ValueError:
        return sha256_bytes(_NODE_PREFIX + (left + right).encode("utf-8"))
    except Exception as exc:
        LOGGER.error("hash_node failed: %s", exc)
        return ZERO_HASH


@dataclass
class MerkleProof:
    """An inclusion proof for one leaf of a Merkle tree.

    Attributes:
        leaf_index: Position of the leaf in the original ordering.
        leaf_hash: Hash of the leaf itself.
        path: Sibling hashes from leaf to root, each ``{"hash", "position"}``.
        root: Expected Merkle root.
    """

    leaf_index: int
    leaf_hash: str
    path: List[Dict[str, str]] = field(default_factory=list)
    root: str = ZERO_HASH

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the proof for storage or transport.

        Returns:
            JSON-safe dictionary.
        """
        return {
            "leaf_index": self.leaf_index,
            "leaf_hash": self.leaf_hash,
            "path": list(self.path),
            "root": self.root,
        }


class MerkleTree:
    """A binary Merkle tree over an ordered list of leaves.

    Odd nodes at any level are promoted (duplicated) to the next level, which is
    safe here because leaves and nodes use distinct hash prefixes.

    Attributes:
        leaves: Hashed leaves in original order.
        levels: All tree levels, ``levels[0]`` being the leaves.
    """

    def __init__(self, items: Optional[Sequence[Any]] = None, prehashed: bool = False) -> None:
        """Build a Merkle tree from items.

        Args:
            items: Leaf values (records, hex digests, bytes...).
            prehashed: Treat inputs as already-hashed leaf digests.
        """
        self.leaves: List[str] = []
        self.levels: List[List[str]] = []
        try:
            if items:
                self.leaves = [str(i) if prehashed else hash_leaf(i) for i in items]
            self._build()
        except Exception as exc:
            LOGGER.error("MerkleTree construction failed: %s", exc)
            self.levels = [self.leaves] if self.leaves else [[]]

    def _build(self) -> None:
        """Compute every level of the tree from the current leaves."""
        try:
            if not self.leaves:
                self.levels = [[]]
                return
            levels: List[List[str]] = [list(self.leaves)]
            current = list(self.leaves)
            while len(current) > 1:
                nxt: List[str] = []
                for index in range(0, len(current), 2):
                    left = current[index]
                    right = current[index + 1] if index + 1 < len(current) else left
                    nxt.append(hash_node(left, right))
                levels.append(nxt)
                current = nxt
            self.levels = levels
        except Exception as exc:
            LOGGER.error("Merkle _build failed: %s", exc)
            self.levels = [list(self.leaves)]

    @property
    def root(self) -> str:
        """Return the Merkle root digest (:data:`ZERO_HASH` when empty)."""
        try:
            if not self.levels or not self.levels[-1]:
                return ZERO_HASH
            return self.levels[-1][0]
        except Exception:
            return ZERO_HASH

    @property
    def size(self) -> int:
        """Return the number of leaves."""
        return len(self.leaves)

    @property
    def depth(self) -> int:
        """Return the number of levels in the tree."""
        return len(self.levels)

    def add_leaf(self, item: Any, prehashed: bool = False) -> None:
        """Append a leaf and rebuild the tree.

        Args:
            item: Leaf value.
            prehashed: Treat the input as an existing leaf digest.
        """
        try:
            self.leaves.append(str(item) if prehashed else hash_leaf(item))
            self._build()
        except Exception as exc:
            LOGGER.error("add_leaf failed: %s", exc)

    def get_proof(self, leaf_index: int) -> Optional[MerkleProof]:
        """Produce an inclusion proof for a leaf.

        Args:
            leaf_index: Index of the leaf in insertion order.

        Returns:
            :class:`MerkleProof`, or ``None`` when the index is out of range.
        """
        try:
            if not (0 <= leaf_index < len(self.leaves)):
                LOGGER.warning("get_proof: index %d out of range", leaf_index)
                return None
            path: List[Dict[str, str]] = []
            index = leaf_index
            for level in self.levels[:-1]:
                sibling_index = index + 1 if index % 2 == 0 else index - 1
                if sibling_index >= len(level):
                    sibling_index = index  # promoted/duplicated node
                path.append({
                    "hash": level[sibling_index],
                    "position": "right" if sibling_index > index else "left",
                })
                index //= 2
            return MerkleProof(leaf_index=leaf_index, leaf_hash=self.leaves[leaf_index],
                               path=path, root=self.root)
        except Exception as exc:
            LOGGER.error("get_proof failed: %s", exc)
            return None

    def verify_proof(self, proof: MerkleProof | Dict[str, Any],
                     expected_root: Optional[str] = None) -> bool:
        """Verify an inclusion proof against a root.

        Args:
            proof: Proof object or its dictionary form.
            expected_root: Root to check against; defaults to this tree's root.

        Returns:
            ``True`` when the recomputed root matches.
        """
        return verify_merkle_proof(proof, expected_root or self.root)

    def to_dict(self) -> Dict[str, Any]:
        """Summarise the tree for reports.

        Returns:
            Dictionary with root, size, depth and the leaf list.
        """
        return {
            "root": self.root,
            "size": self.size,
            "depth": self.depth,
            "leaves": list(self.leaves),
        }


def verify_merkle_proof(proof: MerkleProof | Dict[str, Any], expected_root: str) -> bool:
    """Recompute a Merkle root from a proof and compare it to the expected root.

    Args:
        proof: Proof object or dictionary with ``leaf_hash`` and ``path``.
        expected_root: The root the proof must reproduce.

    Returns:
        ``True`` when the proof is valid.
    """
    try:
        if isinstance(proof, MerkleProof):
            leaf_hash = proof.leaf_hash
            path = proof.path
        else:
            leaf_hash = str(proof.get("leaf_hash", ""))
            path = list(proof.get("path", []))
        if not leaf_hash:
            return False
        current = leaf_hash
        for step in path:
            sibling = str(step.get("hash", ""))
            if step.get("position") == "left":
                current = hash_node(sibling, current)
            else:
                current = hash_node(current, sibling)
        return current == expected_root
    except Exception as exc:
        LOGGER.error("verify_merkle_proof failed: %s", exc)
        return False


def build_merkle_root(items: Sequence[Any], prehashed: bool = False) -> str:
    """Compute just the Merkle root of a sequence.

    Args:
        items: Leaf values.
        prehashed: Treat inputs as leaf digests.

    Returns:
        Root hex digest.
    """
    try:
        return MerkleTree(items, prehashed=prehashed).root
    except Exception as exc:
        LOGGER.error("build_merkle_root failed: %s", exc)
        return ZERO_HASH


def batch_verify(items: Sequence[Any], expected_root: str,
                 prehashed: bool = False) -> Dict[str, Any]:
    """Verify that a batch of items reproduces a published Merkle root.

    Args:
        items: The batch, in its original order.
        expected_root: Previously published root.
        prehashed: Treat inputs as leaf digests.

    Returns:
        Dictionary with ``valid``, ``computed_root``, ``expected_root`` and ``size``.
    """
    try:
        tree = MerkleTree(items, prehashed=prehashed)
        return {
            "valid": tree.root == expected_root,
            "computed_root": tree.root,
            "expected_root": expected_root,
            "size": tree.size,
        }
    except Exception as exc:
        LOGGER.error("batch_verify failed: %s", exc)
        return {"valid": False, "computed_root": ZERO_HASH,
                "expected_root": expected_root, "size": 0, "error": str(exc)}


__all__ = [
    "MerkleTree", "MerkleProof", "build_merkle_root", "verify_merkle_proof",
    "batch_verify", "hash_leaf", "hash_node",
]
