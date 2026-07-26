from __future__ import annotations

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

repository_root = Path(SPECPATH).resolve().parent
application_name = "LM Atelier" if sys.platform == "win32" else "lm-atelier"
icon_path = repository_root / "build" / "installer-assets" / "lm-atelier.ico"

migration_root = repository_root / "services" / "api" / "local_lm" / "migrations"
migration_datas = [
    (
        str(source),
        str(Path("local_lm/migrations") / source.relative_to(migration_root).parent),
    )
    for source in migration_root.rglob("*")
    if source.is_file()
    and "__pycache__" not in source.parts
    and source.suffix not in {".pyc", ".pyo"}
]
capability_pack_root = (
    repository_root / "services" / "api" / "local_lm" / "capability_packs"
)
capability_pack_datas = [
    (str(source), "local_lm/capability_packs")
    for source in capability_pack_root.iterdir()
    if source.is_file()
]

datas = [
    (str(repository_root / "apps" / "web" / "dist"), "web"),
    (str(repository_root / "packaging" / "engines.json"), "."),
    (
        str(repository_root / "packaging" / "runtime-reviews"),
        "runtime-reviews",
    ),
    *migration_datas,
    *capability_pack_datas,
    (str(repository_root / "build" / "release-metadata" / "LICENSE"), "."),
    (
        str(repository_root / "build" / "release-metadata" / "THIRD_PARTY_NOTICES.md"),
        ".",
    ),
    (str(repository_root / "build" / "release-metadata" / "sbom.cdx.json"), "."),
    (
        str(repository_root / "build" / "release-metadata" / "release-manifest.json"),
        ".",
    ),
    (
        str(repository_root / "build" / "release-metadata" / "third-party-licenses"),
        "third-party-licenses",
    ),
]
hiddenimports = collect_submodules("uvicorn")
hiddenimports += collect_submodules("keyring.backends")
hiddenimports += collect_submodules("local_lm.adapters")
excluded_modules = [
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
    upx=False,
    console=True,
    icon=str(icon_path) if os.name == "nt" else None,
)
bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name=application_name,
)
