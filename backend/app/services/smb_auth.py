"""SMB credential encryption service using Fernet symmetric encryption.

Provides plaintext-fallback mode when no master key is configured,
logging warnings so operators know credentials are stored unencrypted.
"""

import logging
from typing import Optional

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

_smb_auth_instance: Optional["SMBAuth"] = None


class SMBAuth:
    """Encrypt/decrypt SMB passwords at rest using Fernet symmetric encryption.

    When no master key is provided, operations fall back to plaintext with
    a warning log so operators are aware credentials are stored unencrypted.
    """

    def __init__(self, master_key: Optional[str] = None):
        if master_key is None:
            logger.warning(
                "SMB_MASTER_KEY not set — SMB passwords will be stored in plaintext"
            )
            self._fernet: Optional[Fernet] = None
        else:
            # Fernet expects a 32-byte URL-safe base64-encoded key.
            # If the user provided a raw secret, derive a proper key via SHA-256.
            try:
                # Try treating it as a valid Fernet key first
                self._fernet = Fernet(master_key.encode() if isinstance(master_key, str) else master_key)
            except Exception:
                # Derive a Fernet-compatible key from the raw secret
                import base64
                import hashlib
                derived = base64.urlsafe_b64encode(hashlib.sha256(master_key.encode()).digest())
                self._fernet = Fernet(derived)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a plaintext password.

        Returns the original plaintext (with a warning log) when no master
        key is configured.
        """
        if self._fernet is None:
            logger.warning("SMB password encryption disabled — storing plaintext")
            return plaintext
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a Fernet-encrypted password.

        Returns the ciphertext as-is when no master key is configured.
        """
        if self._fernet is None:
            return ciphertext
        return self._fernet.decrypt(ciphertext.encode()).decode()


def get_smb_auth(master_key: Optional[str] = None) -> SMBAuth:
    """Return the singleton SMBAuth instance.

    On first call (or when _smb_auth_instance is None), creates a new
    instance with the provided master_key. Subsequent calls return the
    cached instance — master_key is ignored on retries.
    """
    global _smb_auth_instance
    if _smb_auth_instance is None:
        _smb_auth_instance = SMBAuth(master_key)
    return _smb_auth_instance
