"""Client-carried hidden reasoning, authenticated by the user's serving key."""

import os
import stat
import tempfile
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from .errors import APIError


class ThinkingKeyError(RuntimeError):
    pass


def _read_key(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        status = os.fstat(source.fileno())
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_uid != os.getuid()
            or stat.S_IMODE(status.st_mode) not in (0o400, 0o600)
        ):
            raise ThinkingKeyError(
                "Thinking key must be a user-owned regular file with mode "
                f"0400 or 0600: {path}"
            )
        key = source.read(45)
    try:
        if len(key) != 44:
            raise ValueError
        Fernet(key)
    except ValueError:
        raise ThinkingKeyError(f"Invalid thinking key: {path}") from None
    return key


def load_thinking_key(path=None):
    path = (
        Path.home() / "Library/Application Support/Splash/thinking.key"
        if path is None
        else Path(path)
    )
    try:
        try:
            return _read_key(path)
        except FileNotFoundError:
            pass
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".thinking-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as target:
                os.fchmod(target.fileno(), 0o600)
                target.write(Fernet.generate_key())
                target.flush()
                os.fsync(target.fileno())
            # Publish complete bytes without replacing another process's key.
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            os.unlink(temporary)
        return _read_key(path)
    except OSError as error:
        raise ThinkingKeyError(
            f"Cannot use thinking key {path}: {error.strerror}"
        ) from None


class ThinkingCodec:
    MAX_SIGNATURE_BYTES = 16 * 1024 * 1024

    def __init__(self, key=None):
        self.cipher = Fernet(Fernet.generate_key() if key is None else key)

    def encode(self, text):
        payload = text.encode("utf-8")
        # Fernet adds 57 bytes and PKCS7 padding before base64 encoding.
        encoded_size = 4 * ((57 + 16 * (len(payload) // 16 + 1) + 2) // 3)
        if encoded_size > self.MAX_SIGNATURE_BYTES:
            raise APIError(
                500, "thinking exceeds the signature size limit", "thinking_too_large"
            )
        return self.cipher.encrypt(payload).decode("ascii")

    def decode(self, signature):
        if not isinstance(signature, str) or len(signature) > self.MAX_SIGNATURE_BYTES:
            raise APIError(
                400, "invalid signature in thinking block", "invalid_thinking_signature"
            )
        try:
            return self.cipher.decrypt(signature.encode("ascii")).decode("utf-8")
        except (InvalidToken, UnicodeError, ValueError):
            raise APIError(
                400,
                "invalid signature in thinking block; hidden thinking must be returned "
                "to a server using the key that generated it",
                "invalid_thinking_signature",
            ) from None
