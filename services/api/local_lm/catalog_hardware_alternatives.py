from __future__ import annotations

import re
from dataclasses import asdict

from .gguf import automatic_mmproj_selection, complete_gguf_selections
from .hardware_fit import (
    FitRequirements,
    HardwareCandidate,
    capacity_from_system_info,
    rank_hardware_candidates,
)
from .schemas import (
    CatalogDetail,
    CatalogHardwareAlternative,
    CatalogPreflightRequest,
    HardwareFitAdviceOut,
    SystemInfo,
)


def estimated_catalog_ram(download_bytes: int, *, complete: bool) -> int | None:
    """Estimate load memory only when every selected file has a known size."""

    return int(download_bytes * 1.2) + 512 * 1024**2 if complete else None


def catalog_hardware_alternatives(
    detail: CatalogDetail,
    request: CatalogPreflightRequest,
    system: SystemInfo,
    selected_files: list[str],
) -> list[CatalogHardwareAlternative]:
    """Rank complete GGUF choices for a new, explicitly requested install check."""

    if (
        (detail.model.provider or "huggingface") != "huggingface"
        or request.role != "chat"
        or request.engine != "llama.cpp"
        or request.auxiliary_kind is not None
        or re.fullmatch(r"[0-9a-f]{40}", detail.revision) is None
    ):
        return []
    files = {str(item.get("filename") or ""): item for item in detail.files}
    names = [name.casefold() for name in files]
    if len(files) != len(detail.files) or len(set(names)) != len(names):
        return []
    choices: dict[str, tuple[list[str], int, bool]] = {}
    candidates: list[HardwareCandidate] = []
    for primary in complete_gguf_selections(detail.files):
        selected = list(primary)
        projector = automatic_mmproj_selection(detail.files, primary)
        if projector:
            selected.append(projector)
        if set(selected) == set(selected_files):
            continue
        sizes = [files[name].get("size") for name in selected]
        known = [
            size
            for size in sizes
            if isinstance(size, int) and not isinstance(size, bool) and size > 0
        ]
        complete = len(known) == len(sizes)
        total = sum(known)
        key = str(len(choices))
        choices[key] = selected, total, complete
        candidates.append(
            HardwareCandidate(
                key,
                FitRequirements(
                    estimated_system_memory_bytes=estimated_catalog_ram(total, complete=complete),
                ),
            )
        )
    ranked = rank_hardware_candidates(
        capacity_from_system_info(system, runtime_backends=(request.engine,)),
        tuple(candidates),
    )
    return [
        CatalogHardwareAlternative(
            selected_files=choices[item.key][0],
            download_bytes=choices[item.key][1],
            download_size_complete=choices[item.key][2],
            hardware_fit=HardwareFitAdviceOut.model_validate(asdict(item.fit)),
        )
        for item in ranked[:5]
    ]
