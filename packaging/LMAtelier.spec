from __future__ import annotations

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

repository_root = Path(SPECPATH).resolve().parent
application_name = "LM Atelier" if sys.platform == "win32" else "lm-atelier"
icon_path = repository_root / "build" / "installer-assets" / "lm-atelier.ico"

datas = [
    (str(repository_root / "apps" / "web" / "dist"), "web"),
    (
        str(repository_root / "services" / "api" / "local_lm" / "migrations"),
        "local_lm/migrations",
    ),
]
hiddenimports = collect_submodules("uvicorn")
hiddenimports += collect_submodules("keyring.backends")
hiddenimports += collect_submodules("local_lm.adapters")
excluded_modules = [
    "PIL",
    "_pytest",
    "fsspec.conftest",
    "mypy",
    "pytest",
]

analysis = Analysis(
    [str(repository_root / "packaging" / "desktop_entry.py")],
    pathex=[str(repository_root / "services" / "api")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excluded_modules,
    noarchive=False,
)
pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name=application_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=str(icon_path) if os.name == "nt" else None,
)
bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    name=application_name,
)
