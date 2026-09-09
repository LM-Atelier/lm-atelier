from __future__ import annotations

import ctypes
import errno
import sys
from pathlib import Path
from typing import Any

from .filesystem_links import AnchoredDirectory, AnchoredDirectoryError, open_child_directory


class DataDirectoryLockError(RuntimeError):
    """The server could not establish exclusive use of its data directory."""


class DataDirectoryBusy(DataDirectoryLockError):
    def __init__(self) -> None:
        super().__init__(
            "LM Atelier is already starting or running with this data folder. "
            "Wait for that instance to finish starting, or close it before trying again."
        )


_WINDOWS: Any = None


def _windows() -> Any:
    global _WINDOWS
    if _WINDOWS is None:
        windows: Any = ctypes
        kernel = windows.WinDLL("kernel32", use_last_error=True)
        kernel.GetFileInformationByHandleEx.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        )
        kernel.GetFileInformationByHandleEx.restype = ctypes.c_int
        kernel.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
        kernel.CreateMutexW.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel.CloseHandle.restype = ctypes.c_int
        _WINDOWS = kernel
    return _WINDOWS


class _FileIdInfo(ctypes.Structure):
    _fields_ = (("volume", ctypes.c_uint64), ("identifier", ctypes.c_ubyte * 16))


class DataDirectoryLock:
    """Retain kernel exclusion for one directory, independent of its server port.

    POSIX locks the held directory descriptor. Windows exclusively creates a
    named kernel object keyed by the held directory's volume and file identity.
    No PID record or lock-file contents establish ownership, and all handles are
    non-inheritable. The operating system releases them when this process dies.
    """

    def __init__(self, data_dir: Path) -> None:
        self._held: list[AnchoredDirectory] = []
        self._mutex: int | None = None
        root = data_dir.expanduser().absolute()
        try:
            anchor = AnchoredDirectory(Path(root.parts[0]))
            self._held.append(anchor)
            for component in root.parts[1:]:
                anchor = open_child_directory(anchor, component, create=True)
                self._held.append(anchor)
            if sys.platform != "win32" and anchor.descriptor is not None:
                import fcntl

                try:
                    fcntl.flock(anchor.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    if exc.errno in (errno.EACCES, errno.EAGAIN):
                        raise DataDirectoryBusy() from exc
                    raise DataDirectoryLockError(
                        "LM Atelier could not lock its data folder."
                    ) from exc
            elif anchor.handle is not None:
                kernel = _windows()
                identity = _FileIdInfo()
                if not kernel.GetFileInformationByHandleEx(
                    anchor.handle, 18, ctypes.byref(identity), ctypes.sizeof(identity)
                ):
                    raise DataDirectoryLockError("LM Atelier could not identify its data folder.")
                name = (
                    "Global\\LMAtelier.DataDirectory.v1."
                    f"{identity.volume:016x}.{bytes(identity.identifier).hex()}"
                )
                # Creation itself is exclusive. Never wait on this mutex or
                # take recursive thread ownership: an existing object means
                # another holder, including another start in this process.
                windows: Any = ctypes
                windows.set_last_error(0)
                self._mutex = kernel.CreateMutexW(None, False, name)
                error = windows.get_last_error()
                if not self._mutex:
                    self._mutex = None
                    raise DataDirectoryLockError("LM Atelier could not lock its data folder.")
                if error == 183:  # ERROR_ALREADY_EXISTS
                    raise DataDirectoryBusy()
            else:
                raise DataDirectoryLockError("Data folder locking is unavailable on this platform.")
        except AnchoredDirectoryError as exc:
            self.close()
            raise DataDirectoryLockError(
                "LM Atelier's data folder may not contain filesystem links."
            ) from exc
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._mutex is not None:
            if not _windows().CloseHandle(self._mutex):
                raise DataDirectoryLockError(
                    "LM Atelier could not release its data folder lock; exit this process."
                )
            self._mutex = None
        while self._held:
            self._held.pop().close()

    def __enter__(self) -> DataDirectoryLock:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
