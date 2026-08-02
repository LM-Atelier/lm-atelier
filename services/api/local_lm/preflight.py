from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Literal

from .config import Settings
from .gguf import (
    GGUFSelectionError,
    automatic_gguf_selection,
    automatic_mmproj_selection,
    validate_gguf_selection,
)
from .schemas import (
    CatalogDetail,
    CatalogFileSource,
    CatalogPreflight,
    CatalogPreflightCheck,
    CatalogPreflightRequest,
    SystemInfo,
)

_BLOCKED_SUFFIXES = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}


_VLLM_SNAPSHOT_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "configuration.json",
    "generation_config.json",
    "hf_quant_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
}


def _automatic_vllm_selection(files: list[dict[str, Any]]) -> list[str]:
    selected: list[str] = []
    names: set[str] = set()
    for item in files:
        raw_name = str(item.get("filename") or "")
        path = PurePosixPath(raw_name.replace("\\", "/"))
        if (
            not raw_name
            or path.is_absolute()
            or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
        ):
            continue
        base_name = path.name.casefold()
        if base_name.endswith(".safetensors") or base_name in _VLLM_SNAPSHOT_FILES:
            normalized = path.as_posix()
            if normalized.casefold() not in names:
                selected.append(normalized)
                names.add(normalized.casefold())
    selected.sort(key=lambda value: (not value.casefold().endswith(".safetensors"), value))
    selected_names = {PurePosixPath(value).name.casefold() for value in selected}
    if (
        not any(value.endswith(".safetensors") for value in selected_names)
        or "config.json" not in selected_names
        or "hf_quant_config.json" not in selected_names
    ):
        return []
    return selected


def _automatic_selection(
    files: dict[str, dict[str, Any]],
    role: str,
    system_memory_bytes: int,
    auxiliary_kind: str | None = None,
) -> list[str]:
    if role == "chat":
        try:
            chat_primary = automatic_gguf_selection(files.values(), system_memory_bytes)
            projector = automatic_mmproj_selection(files.values(), chat_primary)
            return [*chat_primary, *([projector] if projector else [])]
        except GGUFSelectionError:
            return []

    safe_weights = [
        item for name, item in files.items() if PurePosixPath(name).suffix.lower() == ".safetensors"
    ]
    if not safe_weights:
        return []
    if auxiliary_kind:
        ranked = sorted(
            safe_weights,
            key=lambda item: (
                auxiliary_kind not in str(item.get("filename") or "").casefold(),
                -int(item.get("size") or 0),
                str(item.get("filename") or ""),
            ),
        )
        return [str(ranked[0]["filename"])]
    primary_markers = ("diffusion", "checkpoint", "model", "unet", "t2v", "i2v")
    dependency_markers = ("vae", "text_encoder", "text_encoders", "clip_vision")

    def primary_rank(item: dict[str, Any]) -> tuple[int, int, int]:
        name = str(item.get("filename") or "").lower()
        dependency = any(marker in name for marker in dependency_markers)
        marked = any(marker in name for marker in primary_markers)
        depth = len(PurePosixPath(name).parts)
        return (0 if dependency else 1, 1 if marked else 0, -depth)

    primary = max(
        safe_weights,
        key=lambda item: (
            primary_rank(item),
            int(item.get("size") or 0),
            str(item.get("filename") or ""),
        ),
    )
    selected = [str(primary["filename"])]
    # Component-style media repositories need one companion from each declared
    # dependency folder. Single-checkpoint repositories naturally select only
    # the primary file.
    primary_name = str(primary.get("filename") or "")
    if len(PurePosixPath(primary_name).parts) > 1:
        for marker in dependency_markers:
            companions = [
                item for item in safe_weights if marker in str(item.get("filename") or "").lower()
            ]
            if companions:
                chosen = min(
                    companions,
                    key=lambda item: (
                        int(item.get("size") or 2**63 - 1),
                        str(item.get("filename") or ""),
                    ),
                )
                filename = str(chosen["filename"])
                if filename not in selected:
                    selected.append(filename)
    return selected


def assess_catalog_install(
    detail: CatalogDetail,
    request: CatalogPreflightRequest,
    settings: Settings,
    system: SystemInfo,
) -> CatalogPreflight:
    resolved_revision = detail.revision
    files = {str(item.get("filename") or ""): item for item in detail.files}
    requested_selected = list(request.selected_files)
    selected = list(dict.fromkeys(requested_selected))
    checks: list[CatalogPreflightCheck] = []
    selection_error: str | None = None

    normalized_requested = [name.casefold() for name in requested_selected]
    if len(normalized_requested) != len(set(normalized_requested)):
        selection_error = (
            "The file selection contains duplicate paths. Run the install check again."
        )
    elif not selected:
        try:
            selected = (
                _automatic_vllm_selection(detail.files)
                if request.role == "chat" and request.engine == "vllm"
                else automatic_gguf_selection(detail.files, system.memory_total_bytes)
                if request.role == "chat"
                else _automatic_selection(
                    files,
                    request.role,
                    system.memory_total_bytes,
                    request.auxiliary_kind,
                )
            )
        except GGUFSelectionError as exc:
            selection_error = str(exc)
    if (
        request.role == "chat"
        and request.engine == "llama.cpp"
        and selected
        and selection_error is None
    ):
        projector = automatic_mmproj_selection(detail.files, selected)
        if projector and projector not in selected:
            selected.append(projector)

    missing = [name for name in selected if name not in files]
    unsafe_paths = [
        name
        for name in selected
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
    ]
    if (
        request.role == "chat"
        and request.engine == "llama.cpp"
        and selected
        and not missing
        and not unsafe_paths
        and selection_error is None
    ):
        selected_names = set(selected)
        selected_metadata = [
            item for item in detail.files if str(item.get("filename") or "") in selected_names
        ]
        projectors = [
            str(item.get("filename") or "")
            for item in selected_metadata
            if "mmproj" in PurePosixPath(str(item.get("filename") or "")).name.casefold()
        ]
        try:
            primary = validate_gguf_selection(
                selected_metadata,
                require_split_metadata=True,
            )
            if len(projectors) > 1:
                raise GGUFSelectionError(
                    "The chat model selection contains multiple multimodal projectors."
                )
            selected = [*primary, *projectors]
        except GGUFSelectionError as exc:
            selection_error = str(exc)

    if selection_error:
        checks.append(
            _check(
                "selection",
                "Automatic model selection",
                "block",
                selection_error,
            )
        )
    elif not selected:
        checks.append(
            _check(
                "selection",
                "Automatic model selection",
                "block",
                "No safe model file could be selected automatically.",
            )
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
                "Automatic model selection",
                "pass",
                (
                    f"LM Atelier selected {len(selected)} safe file"
                    f"{'s' if len(selected) != 1 else ''} for this runtime."
                ),
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

    expected_sha256 = {
        name: str(files[name].get("sha256")).lower()
        for name in selected
        if name in files
        and isinstance(files[name].get("sha256"), str)
        and re.fullmatch(r"[0-9a-fA-F]{64}", str(files[name]["sha256"]))
    }
    file_sources = {
        name: CatalogFileSource(
            remote_id=str(files[name]["source_remote_id"]),
            revision=str(files[name]["source_revision"]),
            filename=str(files[name]["source_filename"]),
            size_bytes=(
                int(files[name]["size"])
                if isinstance(files[name].get("size"), int)
                and not isinstance(files[name]["size"], bool)
                else None
            ),
            sha256=expected_sha256.get(name),
        )
        for name in selected
        if name in files
        and files[name].get("source_remote_id")
        and files[name].get("source_revision")
        and files[name].get("source_filename")
    }
    checksum_complete = bool(selected) and len(expected_sha256) == len(selected)
    checks.append(
        _check(
            "checksum",
            "Checksum metadata",
            "pass" if checksum_complete else "warn",
            (
                "SHA-256 metadata is available for every selected file."
                if checksum_complete
                else "One or more selected files have no public SHA-256 metadata."
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

    expected_engine = detail.model.required_runtime or (
        "llama.cpp" if request.role == "chat" else "comfyui"
    )
    compatibility_status: Literal["pass", "warn", "block"] = "pass"
    compatibility_detail = "Catalog metadata matches the selected local runtime."
    if request.engine != expected_engine:
        compatibility_status = "block"
        compatibility_detail = f"{request.role} installs require the {expected_engine} runtime."
    elif detail.model.compatibility == "unsupported":
        compatibility_status = "block"
        compatibility_detail = "Catalog metadata marks this model unsupported."
    elif detail.model.compatibility == "advanced_import" and not (
        detail.model.required_runtime == "vllm" and request.engine == "vllm"
    ):
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
        int(download_bytes * 1.25) + 1024**3
        if download_bytes and (request.role != "chat" or request.engine == "vllm")
        else None
    )
    accelerators = [device for device in system.devices if device.kind != "cpu"]
    available_accelerator = max(
        (device.available_memory_bytes or 0 for device in accelerators), default=0
    )
    if request.role == "chat" and request.engine != "vllm" and estimated_ram:
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
        memory_detail = "No accelerator memory was detected; accelerated generation may fail."
    checks.append(_check("memory", "Memory estimate", memory_status, memory_detail))

    checks.append(
        _check(
            "revision",
            "Pinned revision",
            "pass" if resolved_revision != "main" else "warn",
            (
                f"Install is pinned to {resolved_revision}."
                if resolved_revision != "main"
                else "The catalog could not resolve main to an immutable commit."
            ),
        )
    )

    return CatalogPreflight(
        remote_id=detail.model.remote_id,
        revision=resolved_revision,
        selected_files=selected,
        expected_sha256=expected_sha256,
        file_sources=file_sources,
        download_bytes=download_bytes,
        available_disk_bytes=system.disk_free_bytes,
        estimated_ram_bytes=estimated_ram,
        estimated_vram_bytes=estimated_vram,
        can_install=not any(check.status == "block" for check in checks),
        checks=checks,
        auxiliary_kind=request.auxiliary_kind,
        content_rating=detail.model.content_rating,
    )


def _check(
    check_id: str,
    label: str,
    status: Literal["pass", "warn", "block"],
    detail: str,
) -> CatalogPreflightCheck:
    return CatalogPreflightCheck(id=check_id, label=label, status=status, detail=detail)
