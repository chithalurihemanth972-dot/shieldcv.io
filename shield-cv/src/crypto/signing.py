"""
Ed25519 digital signatures for SHIELD-CV provenance records.

Key material is generated and stored locally (air-gapped); private keys are
written with ``0600`` permissions. All operations use the ``cryptography``
library — never a hand-rolled implementation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.config import get_config
from src.crypto.hashing import canonical_json
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    CRYPTO_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency guard
    CRYPTO_AVAILABLE = False
    Ed25519PrivateKey = None  # type: ignore[assignment]
    Ed25519PublicKey = None  # type: ignore[assignment]
    InvalidSignature = Exception  # type: ignore[assignment]
    LOGGER.error("cryptography not installed: Ed25519 signing NOT AVAILABLE")


class SigningKeyPair:
    """An Ed25519 key pair used to sign and verify provenance records.

    Attributes:
        private_key: Loaded private key (``None`` in verify-only mode).
        public_key: Corresponding public key.
    """

    def __init__(self, private_key: Optional["Ed25519PrivateKey"] = None,
                 public_key: Optional["Ed25519PublicKey"] = None) -> None:
        """Wrap existing key objects.

        Args:
            private_key: Ed25519 private key, if available.
            public_key: Ed25519 public key; derived from the private key if omitted.
        """
        self.private_key = private_key
        self.public_key = public_key or (private_key.public_key() if private_key else None)

    # -- construction ------------------------------------------------------
    @classmethod
    def generate(cls) -> "SigningKeyPair":
        """Generate a fresh Ed25519 key pair.

        Returns:
            New :class:`SigningKeyPair`.

        Raises:
            RuntimeError: If the ``cryptography`` package is unavailable.
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography package required for Ed25519 key generation")
        try:
            key = Ed25519PrivateKey.generate()
            LOGGER.info("Generated new Ed25519 key pair")
            return cls(private_key=key)
        except Exception as exc:
            LOGGER.error("Key generation failed: %s", exc)
            raise

    @classmethod
    def load(cls, private_path: Optional[str | Path] = None,
             public_path: Optional[str | Path] = None,
             password: Optional[bytes] = None) -> Optional["SigningKeyPair"]:
        """Load a key pair from PEM files on disk.

        Args:
            private_path: Path to the PEM private key.
            public_path: Path to the PEM public key (used when no private key).
            password: Optional passphrase protecting the private key.

        Returns:
            Loaded pair, or ``None`` when nothing could be read.
        """
        if not CRYPTO_AVAILABLE:
            return None
        try:
            private_key = None
            public_key = None
            if private_path and Path(private_path).is_file():
                data = Path(private_path).read_bytes()
                private_key = serialization.load_pem_private_key(data, password=password)
                if not isinstance(private_key, Ed25519PrivateKey):
                    LOGGER.error("Loaded private key is not Ed25519")
                    return None
            if public_key is None and public_path and Path(public_path).is_file():
                data = Path(public_path).read_bytes()
                public_key = serialization.load_pem_public_key(data)
            if private_key is None and public_key is None:
                return None
            return cls(private_key=private_key, public_key=public_key)
        except Exception as exc:
            LOGGER.error("Key load failed: %s", exc)
            return None

    @classmethod
    def load_or_create(cls, private_path: Optional[str | Path] = None,
                       public_path: Optional[str | Path] = None) -> Optional["SigningKeyPair"]:
        """Load the configured key pair, generating and saving one if absent.

        Args:
            private_path: Override for the private key path.
            public_path: Override for the public key path.

        Returns:
            Usable :class:`SigningKeyPair`, or ``None`` if crypto is unavailable.
        """
        if not CRYPTO_AVAILABLE:
            LOGGER.error("Ed25519 NOT AVAILABLE: install 'cryptography'")
            return None
        try:
            cfg = get_config()
            priv = Path(private_path) if private_path else cfg.path(
                "private_key_file", cfg.get("provenance.private_key_file",
                                            "output/keys/shield_ed25519.key"))
            pub = Path(public_path) if public_path else cfg.path(
                "public_key_file", cfg.get("provenance.public_key_file",
                                           "output/keys/shield_ed25519.pub"))
            if not priv.is_absolute():
                priv = cfg.root / priv
            if not pub.is_absolute():
                pub = cfg.root / pub

            existing = cls.load(priv, pub)
            if existing is not None and existing.private_key is not None:
                LOGGER.debug("Loaded existing signing key from %s", priv)
                return existing

            pair = cls.generate()
            pair.save(priv, pub)
            return pair
        except Exception as exc:
            LOGGER.error("load_or_create failed: %s", exc)
            return None

    # -- persistence -------------------------------------------------------
    def save(self, private_path: str | Path, public_path: str | Path,
             password: Optional[bytes] = None) -> bool:
        """Write the key pair to PEM files with restrictive permissions.

        Args:
            private_path: Destination for the private key.
            public_path: Destination for the public key.
            password: Optional passphrase to encrypt the private key.

        Returns:
            ``True`` on success.
        """
        try:
            priv = Path(private_path)
            pub = Path(public_path)
            priv.parent.mkdir(parents=True, exist_ok=True)
            pub.parent.mkdir(parents=True, exist_ok=True)

            if self.private_key is not None:
                encryption = (serialization.BestAvailableEncryption(password)
                              if password else serialization.NoEncryption())
                priv.write_bytes(self.private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=encryption,
                ))
                try:
                    os.chmod(priv, 0o600)
                except Exception as exc:  # Windows / exotic FS
                    LOGGER.debug("chmod on private key failed: %s", exc)

            if self.public_key is not None:
                pub.write_bytes(self.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                ))
            LOGGER.info("Saved Ed25519 key pair to %s / %s", priv.name, pub.name)
            return True
        except Exception as exc:
            LOGGER.error("Key save failed: %s", exc)
            return False

    # -- operations --------------------------------------------------------
    def sign(self, payload: bytes | str | Dict[str, Any]) -> Optional[str]:
        """Sign a payload with the private key.

        Args:
            payload: Bytes, string, or JSON-serialisable object (canonicalised).

        Returns:
            Hex-encoded signature, or ``None`` when signing is impossible.
        """
        try:
            if self.private_key is None:
                LOGGER.error("sign: no private key loaded (verify-only mode)")
                return None
            data = _to_bytes(payload)
            return self.private_key.sign(data).hex()
        except Exception as exc:
            LOGGER.error("Signing failed: %s", exc)
            return None

    def verify(self, payload: bytes | str | Dict[str, Any], signature: str) -> bool:
        """Verify an Ed25519 signature over a payload.

        Args:
            payload: The originally-signed data.
            signature: Hex-encoded signature.

        Returns:
            ``True`` only when the signature is cryptographically valid.
        """
        try:
            if self.public_key is None:
                LOGGER.error("verify: no public key loaded")
                return False
            if not signature:
                return False
            self.public_key.verify(bytes.fromhex(signature), _to_bytes(payload))
            return True
        except InvalidSignature:
            return False
        except ValueError:
            LOGGER.warning("verify: signature is not valid hex")
            return False
        except Exception as exc:
            LOGGER.error("Verification error: %s", exc)
            return False

    def public_key_hex(self) -> Optional[str]:
        """Return the raw 32-byte public key as hex (for report embedding).

        Returns:
            Hex string, or ``None``.
        """
        try:
            if self.public_key is None:
                return None
            raw = self.public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            return raw.hex()
        except Exception as exc:
            LOGGER.error("public_key_hex failed: %s", exc)
            return None

    @property
    def can_sign(self) -> bool:
        """Whether a private key is present for signing."""
        return self.private_key is not None


def _to_bytes(payload: bytes | str | Dict[str, Any]) -> bytes:
    """Normalise any payload type to the exact bytes that get signed.

    Args:
        payload: Bytes, string, or JSON-serialisable object.

    Returns:
        UTF-8 bytes (canonical JSON for objects).
    """
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        return payload.encode("utf-8")
    return canonical_json(payload).encode("utf-8")


def generate_keypair() -> Optional[SigningKeyPair]:
    """Convenience wrapper generating a key pair without raising.

    Returns:
        New key pair, or ``None`` on failure.
    """
    try:
        return SigningKeyPair.generate()
    except Exception as exc:
        LOGGER.error("generate_keypair failed: %s", exc)
        return None


def sign_payload(payload: Dict[str, Any],
                 keypair: Optional[SigningKeyPair] = None) -> Tuple[Optional[str], Optional[str]]:
    """Sign a record payload, loading the default key pair when needed.

    Args:
        payload: Canonical record fields to sign.
        keypair: Explicit key pair; the configured default is used if omitted.

    Returns:
        Tuple ``(signature_hex, public_key_hex)``; ``(None, None)`` on failure.
    """
    try:
        pair = keypair or SigningKeyPair.load_or_create()
        if pair is None:
            return None, None
        return pair.sign(payload), pair.public_key_hex()
    except Exception as exc:
        LOGGER.error("sign_payload failed: %s", exc)
        return None, None


def verify_payload(payload: Dict[str, Any], signature: str,
                   keypair: Optional[SigningKeyPair] = None) -> bool:
    """Verify a record payload signature using the configured public key.

    Args:
        payload: Canonical record fields.
        signature: Hex signature to check.
        keypair: Explicit key pair; the configured default is used if omitted.

    Returns:
        ``True`` when the signature verifies.
    """
    try:
        pair = keypair or SigningKeyPair.load_or_create()
        if pair is None:
            return False
        return pair.verify(payload, signature)
    except Exception as exc:
        LOGGER.error("verify_payload failed: %s", exc)
        return False


__all__ = [
    "SigningKeyPair", "generate_keypair", "sign_payload", "verify_payload",
    "CRYPTO_AVAILABLE",
]
