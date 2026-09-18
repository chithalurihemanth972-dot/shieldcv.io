"""Cryptographic primitives: SHA-256 hashing, Ed25519 signing, Merkle trees, hash chains."""

from src.crypto.hashing import (
    sha256_bytes, sha256_text, sha256_file, sha256_json, sha256_state_dict, canonical_json,
)
from src.crypto.signing import SigningKeyPair, generate_keypair, sign_payload, verify_payload
from src.crypto.merkle import MerkleTree, MerkleProof, build_merkle_root, verify_merkle_proof
from src.crypto.chain import (
    verify_chain, link_records, compute_record_hash, ChainVerificationResult, GENESIS_HASH,
)

__all__ = [
    "sha256_bytes", "sha256_text", "sha256_file", "sha256_json", "sha256_state_dict",
    "canonical_json", "SigningKeyPair", "generate_keypair", "sign_payload", "verify_payload",
    "MerkleTree", "MerkleProof", "build_merkle_root", "verify_merkle_proof",
    "verify_chain", "link_records", "compute_record_hash", "ChainVerificationResult",
    "GENESIS_HASH",
]
