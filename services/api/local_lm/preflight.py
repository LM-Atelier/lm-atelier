from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from .config import Settings
from .schemas import (
    CatalogDetail,
    CatalogPreflight,
    CatalogPreflightCheck,
    CatalogPreflightRequest,
    SystemInfo,
)

_BLOCKED_SUFFIXES = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}


def assess_catalog_install(
    detail: CatalogDetail,
    request: CatalogPreflightRequest,
    settings: Settings,
    system: SystemInfo,
) -> CatalogPreflight:
    files = {str(item.get("filename") or ""): item for item in detail.files}
    selected = list(dict.fromkeys(request.selected_files))
    checks: list[CatalogPreflightCheck] = []

    if not selected and request.role == "chat":
        ggufs = [item for name, item in files.items() if name.lower().endswith(".gguf")]
        if ggufs:
            chosen = min(ggufs, key=lambda item: int(item.get("size") or 2**63 - 1))
            selected = [str(chosen["filename"])]

    missing = [name for name in selected if name not in files]
    unsafe_paths = [
        name
        for name in selected
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
    ]
    if not selected:
        checks.append(
            _check("selection", "File selection", "block", "Select at least one model file.")
        )
    elif missing or unsafe_paths:
        detail_text = "Selected files are not present in this revision."
        if unsafe_paths:
            detail_text = "Selected files contain unsafe paths."
        checks.append(_check("selection", "File selection", "block", detail_text))
    else:
        checks.append(
            _check(
                "selection",
                "File selection",
                "pass",
                f"{len(selected)} exact file{'s' if len(selected) != 1 else ''} selected.",
            )
        )

    unsafe_weights = [
        name for name in selected if PurePosixPath(name).suffix.lower() in _BLOCKED_SUFFIXES
    ]
    checks.append(
        _check(
            "weights",
            "Weight safety",
            "block" if unsafe_weights else "pass",
            (
                "Pickle-compatible weights are blocked: " + ", ".join(unsafe_weights[:3])
                if unsafe_weights
                else "No blocked pickle-compatible weights are selected."
            ),
        )
    )

    if detail.model.gated and not settings.hf_token:
        checks.append(
            _check(
                "access",
                "Repository access",
                "block",
                "This gated model needs a Hugging Face token supplied to the local service.",
            )
        )
    else:
        checks.append(
            _check(
                "access",
                "Repository access",
                "pass",
                "Access is open or a local Hugging Face token is available.",
            )
        )

    checks.append(
        _check(
            "license",
            "License metadata",
            "pass" if detail.model.license_id else "warn",
            (
                f"Hub metadata declares {detail.model.license_id}."
                if detail.model.license_id
                else "No license identifier was found; review the model card before installing."
            ),
        )
    )

    expected_engine = "llama.cpp" if request.role == "chat" else "comfyui"
    compatibility_status: Literal["pass", "warn", "block"] = "pass"
    compatibility_detail = "Catalog metadata matches the selected local runtime."
    if request.engine != expected_engine:
        compatibility_status = "block"
        compatibility_detail = f"{request.role} installs require the {expected_engine} runtime."
    elif detail.model.compatibility == "unsupported":
        compatibility_status = "block"
        compatibility_detail = "Catalog metadata marks this model unsupported."
    elif detail.model.compatibility == "advanced_import":
        compatibility_status = "warn"
        compatibility_detail = "This model needs advanced setup or a verified workflow."
    checks.append(
        _check(
            "runtime",
            "Runtime compatibility",
            compatibility_status,
            compatibility_detail,
        )
    )

    known_sizes = [int(files[name].get("size") or 0) for name in selected if name in files]
    download_bytes = sum(known_sizes)
    unknown_sizes = [name for name in selected if name in files and not files[name].get("size")]
    if unknown_sizes:
        checks.append(
            _check(
                "disk",
                "Disk capacity",
                "warn",
                "Some selected file sizes are unknown; exact disk fit cannot be confirmed.",
            )
        )
    elif download_bytes > system.disk_free_bytes:
        checks.append(
            _check(
                "disk",
                "Disk capacity",
                "block",
                "The selected files exceed currently available local disk space.",
            )
        )
    else:
        checks.append(
            _check(
                "disk",
                "Disk capacity",
                "pass",
                "The selected files fit in currently available local disk space.",
            )
        )

    estimated_ram = int(download_bytes * 1.2) + 512 * 1024**2 if download_bytes else None
    estimated_vram = (
        int(download_bytes * 1.25) + 1024**3 if download_bytes and request.role != "chat" else None
    )
    accelerators = [device for device in system.devices if device.kind != "cpu"]
    available_accelerator = max(
        (device.available_memory_bytes or 0 for device in accelerators), default=0
    )
    if request.role == "chat" and estimated_ram:
        memory_status: Literal["pass", "warn", "block"] = (
            "pass" if estimated_ram <= system.memory_total_bytes else "warn"
        )
        memory_detail = (
            "Estimated loaded size fits total system memory."
            if memory_status == "pass"
            else (
                "Estimated loaded size exceeds total system memory; "
                "a smaller quantization is advised."
            )
        )
    elif estimated_vram and available_accelerator:
        memory_status = "pass" if estimated_vram <= available_accelerator else "warn"
        memory_detail = (
            "Estimated loaded size fits currently available accelerator memory."
            if memory_status == "pass"
            else "Estimated loaded size exceeds currently available accelerator memory."
        )
    else:
        memory_status = "warn"
        memory_detail = (
            "No accelerator memory was detected; media generation may fall back or fail."
        )
    checks.append(_check("memory", "Memory estimate", memory_status, memory_detail))

    checks.append(
        _check(
            "revision",
            "Pinned revision",
            "pass" if request.revision != "main" else "warn",
            (
                f"Install is pinned to {request.revision}."
                if request.revision != "main"
                else "The mutable main revision is selected; use a commit SHA for reproducibility."
            ),
        )
    )

    return CatalogPreflight(
        remote_id=detail.model.remote_id,
        revision=request.revision,
        selected_files=selected,
        download_bytes=download_bytes,
        available_disk_bytes=system.disk_free_bytes,
        estimated_ram_bytes=estimated_ram,
        estimated_vram_bytes=estimated_vram,
        can_install=not any(check.status == "block" for check in checks),
        checks=checks,
    )


def _check(
    check_id: str,
    label: str,
    status: Literal["pass", "warn", "block"],
    detail: str,
) -> CatalogPreflightCheck:
    return CatalogPreflightCheck(id=check_id, label=label, status=status, detail=detail)
