from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import stat
import time
from contextlib import suppress
from pathlib import Path

INSTANCE_ID_HEADER = "X-LM-Atelier-Instance"
_INSTANCE_SEED_NAME = "desktop-instance-seed"
_SEED_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_CONTEXT = b"lm-atelier-desktop-instance-v1\0"


class InstanceIdentityError(RuntimeError):
    pass


def load_or_create_instance_identity(data_dir: Path) -> str:
    """Return an opaque identity bound to one resolved LM Atelier data root."""

    root = data_dir.expanduser().resolve()
    state_dir = root / "state"
    if _is_link(state_dir):
        raise InstanceIdentityError("LM Atelier's state folder may not be a filesystem link")
    state_dir.mkdir(parents=True, exist_ok=True)
    if _is_link(state_dir) or not state_dir.is_dir() or state_dir.resolve().parent != root:
        raise InstanceIdentityError("LM Atelier's state folder is outside its data folder")
    seed_path = state_dir / _INSTANCE_SEED_NAME

    for attempt in range(5):
        try:
            seed = _read_seed(seed_path)
        except InstanceIdentityError:
            if attempt == 4:
                raise
            time.sleep(0.01)
            continue
        if seed is not None:
            return _derive_identity(seed, root)

        candidate = secrets.token_bytes(32)
        try:
            descriptor = os.open(
                seed_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError:
            time.sleep(0.01)
            continue
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="ascii") as handle:
                handle.write(candidate.hex())
                handle.flush()
                os.fsync(handle.fileno())
            seed_path.chmod(0o600)
        except Exception:
            with suppress(OSError):
                os.close(descriptor)
            seed_path.unlink(missing_ok=True)
            raise
        return _derive_identity(candidate, root)

    raise InstanceIdentityError("LM Atelier could not establish ownership of its data folder")


def _read_seed(path: Path) -> bytes | None:
    if _is_link(path):
        raise InstanceIdentityError("LM Atelier's desktop identity may not be a filesystem link")
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise InstanceIdentityError("LM Atelier's desktop identity is not a regular file")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    except InstanceIdentityError:
        raise
    except OSError as exc:
        raise InstanceIdentityError("LM Atelier could not read its desktop identity") from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise InstanceIdentityError("LM Atelier's desktop identity is not a regular file")
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(67)
        if len(raw) > 66:
            raise InstanceIdentityError("LM Atelier's desktop identity is invalid")
    except InstanceIdentityError:
        raise
    except OSError as exc:
        raise InstanceIdentityError("LM Atelier could not read its desktop identity") from exc
    finally:
        os.close(descriptor)
    try:
        value = raw.decode("ascii").strip()
    except UnicodeError as exc:
        raise InstanceIdentityError("LM Atelier's desktop identity is invalid") from exc
    if not _SEED_PATTERN.fullmatch(value):
        raise InstanceIdentityError("LM Atelier's desktop identity is invalid")
    return bytes.fromhex(value)


def _derive_identity(seed: bytes, root: Path) -> str:
    normalized_root = os.path.normcase(str(root)).encode("utf-8", errors="surrogatepass")
    return hmac.new(
        seed,
        _IDENTITY_CONTEXT + normalized_root,
        hashlib.sha256,
    ).hexdigest()


def _is_link(path: Path) -> bool:
    try:
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    except FileNotFoundError:
        return False
    except OSError:
        return True
