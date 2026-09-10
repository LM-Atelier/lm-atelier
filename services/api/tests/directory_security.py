"""Create only neutral temporary directories with explicit test security."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Any


def create_owned_directory(
    selected: Path, *, extra_aces: str = "", null_dacl: bool = False
) -> Path:
    if sys.platform == "win32":
        # Elevated tokens can default new objects to a group owner. These
        # tests require a user-owned, accessible directory, so create only
        # this temporary fixture with explicit owner and current-user access.
        windows: Any = ctypes
        security = windows.WinDLL("advapi32", use_last_error=True)
        query = security.GetTokenInformation
        query.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        query.restype = ctypes.c_int
        needed = ctypes.c_ulong()
        assert not query(ctypes.c_void_p(-6), 1, None, 0, ctypes.byref(needed))
        assert windows.get_last_error() == 122
        buffer = ctypes.create_string_buffer(needed.value)
        assert query(ctypes.c_void_p(-6), 1, buffer, len(buffer), ctypes.byref(needed))
        user = ctypes.c_void_p.from_buffer(buffer)
        assert user.value is not None
        kernel = windows.WinDLL("kernel32", use_last_error=True)
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        kernel.LocalFree.restype = ctypes.c_void_p
        convert_sid = security.ConvertSidToStringSidW
        convert_sid.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_wchar_p)]
        convert_sid.restype = ctypes.c_int
        convert_descriptor = security.ConvertStringSecurityDescriptorToSecurityDescriptorW
        convert_descriptor.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_ulong),
        ]
        convert_descriptor.restype = ctypes.c_int

        class SecurityAttributes(ctypes.Structure):
            _fields_ = (
                ("Length", ctypes.c_ulong),
                ("Descriptor", ctypes.c_void_p),
                ("Inherit", ctypes.c_int),
            )

        kernel.CreateDirectoryW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(SecurityAttributes)]
        kernel.CreateDirectoryW.restype = ctypes.c_int
        sid_text = ctypes.c_wchar_p()
        descriptor = ctypes.c_void_p()
        try:
            assert convert_sid(user, ctypes.byref(sid_text))
            assert sid_text.value is not None
            sddl = (
                f"O:{sid_text.value}D:NO_ACCESS_CONTROL"
                if null_dacl
                else f"O:{sid_text.value}D:P(A;OICI;FA;;;{sid_text.value})" + extra_aces
            )
            assert convert_descriptor(sddl, 1, ctypes.byref(descriptor), None)
            attributes = SecurityAttributes(ctypes.sizeof(SecurityAttributes), descriptor.value, 0)
            assert kernel.CreateDirectoryW(str(selected), ctypes.byref(attributes))
        finally:
            if descriptor.value:
                kernel.LocalFree(descriptor)
            if sid_text.value:
                kernel.LocalFree(ctypes.cast(sid_text, ctypes.c_void_p))
    else:
        selected.mkdir(mode=0o700)
    return selected
