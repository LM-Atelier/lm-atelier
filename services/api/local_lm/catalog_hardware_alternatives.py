from __future__ import annotations

import re
from dataclasses import asdict, replace

from .chat_memory import (
    CHAT_CONTEXT_BYTES_PER_TOKEN,
    DEFAULT_CHAT_CONTEXT,
    MINIMUM_CHAT_CONTEXT_MEMORY,
    estimated_chat_memory,
)
from .gguf import automatic_mmproj_selection, complete_gguf_selections
from .hardware_fit import (
    BoundedSetting,
    FitReason,
    FitRequirements,
    HardwareCandidate,
    HardwareFit,
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


def estimated_catalog_ram(
    download_bytes: int, *, complete: bool, chat_context: bool = False
) -> int | None:
    """Estimate load memory only when every selected file has a known size."""

    if not complete:
        return None
    model_bytes = int(download_bytes * 1.2)
    return (
        estimated_chat_memory(model_bytes)
        if chat_context
        else model_bytes + MINIMUM_CHAT_CONTEXT_MEMORY
    )


def catalog_context_settings(
    estimated_ram_bytes: int | None, capacity_bytes: int
) -> tuple[BoundedSetting, ...]:
    """Bound context by the default and a ninety-percent total-RAM budget."""

    if estimated_ram_bytes is None or capacity_bytes <= 0:
        return ()
    model_bytes = estimated_ram_bytes - estimated_chat_memory(0)
    if model_bytes <= 0:
        return ()
    context_budget = capacity_bytes * 9 // 10 - model_bytes
    if context_budget < MINIMUM_CHAT_CONTEXT_MEMORY:
        return ()
    maximum = min(DEFAULT_CHAT_CONTEXT, context_budget // CHAT_CONTEXT_BYTES_PER_TOKEN)
    maximum = maximum // 512 * 512
    return (
        BoundedSetting(
            key="context_length",
            label="Context length",
            unit="tokens",
            minimum=512,
            maximum=DEFAULT_CHAT_CONTEXT,
            preferred_minimum=min(2048, maximum),
            preferred_maximum=maximum,
            tight_minimum=512,
            tight_maximum=maximum,
        ),
    )


def with_catalog_context_estimate(fit: HardwareFit) -> HardwareFit:
    """Explain the context assumption when a calculated memory resource exists."""

    if not fit.resources:
        return fit
    return replace(
        fit,
        reasons=(
            *fit.reasons,
            FitReason(
                "chat_context_estimate",
                "info",
                f"Memory estimates assume the default {DEFAULT_CHAT_CONTEXT}-token context. "
                "Model limits and actual memory use may differ.",
            ),
        ),
    )


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
        estimated_ram = estimated_catalog_ram(total, complete=complete, chat_context=True)
        candidates.append(
            HardwareCandidate(
                key,
                FitRequirements(
                    estimated_system_memory_bytes=estimated_ram,
                    settings=catalog_context_settings(estimated_ram, system.memory_total_bytes),
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
            hardware_fit=HardwareFitAdviceOut.model_validate(
                asdict(with_catalog_context_estimate(item.fit))
            ),
        )
        for item in ranked[:5]
    ]
