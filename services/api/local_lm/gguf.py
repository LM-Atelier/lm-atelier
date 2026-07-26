from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

_SPLIT_GGUF = re.compile(
    r"^(?P<stem>.+)-(?P<index>[0-9]{5})-of-(?P<count>[0-9]{5})\.gguf$",
    re.IGNORECASE,
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_QUANTIZATION_PRIORITY = {
    "q4_k_m": 0,
    "q5_k_m": 1,
    "q4_k_s": 2,
    "q5_k_s": 3,
    "q8_0": 4,
    "q3_k_m": 5,
    "q3_k_s": 6,
    "q2_k": 7,
}
_MMPROJ_PRECISION_PRIORITY = {
    "f16": 0,
    "bf16": 1,
    "q8_0": 2,
    "q6_k": 3,
    "q5_k_m": 4,
    "q4_k_m": 5,
    "f32": 6,
}
_IDENTITY_IGNORED_TOKENS = {
    "gguf",
    "mmproj",
    "model",
    "vision",
    "projector",
    *{
        component
        for quantization in (*_QUANTIZATION_PRIORITY, *_MMPROJ_PRECISION_PRIORITY)
        for component in quantization.split("_")
    },
}


class GGUFSelectionError(ValueError):
    """A GGUF selection cannot be resolved to one complete model."""


@dataclass(frozen=True)
class GGUFFile:
    filename: str
    size: int | None
    sha256: str | None

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any]) -> GGUFFile:
        raw_size = item.get("size")
        size = raw_size if isinstance(raw_size, int) and not isinstance(raw_size, bool) else None
        raw_sha256 = item.get("sha256")
        sha256 = str(raw_sha256).lower() if isinstance(raw_sha256, str) else None
        return cls(
            filename=str(item.get("filename") or ""),
            size=size,
            sha256=sha256,
        )


@dataclass(frozen=True)
class _SplitPart:
    file: GGUFFile
    stem: str
    index: int
    count: int


@dataclass(frozen=True)
class _Candidate:
    files: tuple[GGUFFile, ...]
    split: bool

    @property
    def size(self) -> int:
        sizes = [item.size for item in self.files]
        if any(size is None or size <= 0 for size in sizes):
            return 2**63 - 1
        return sum(size for size in sizes if size is not None)


def automatic_gguf_selection(
    items: Iterable[Mapping[str, Any]],
    system_memory_bytes: int,
) -> list[str]:
    """Choose one coherent GGUF candidate, treating a shard set as one model."""

    candidates, errors = _candidates(_records(items), require_split_metadata=True)
    if not candidates:
        if errors:
            raise GGUFSelectionError(errors[0])
        raise GGUFSelectionError("No safe GGUF model file was found in this revision.")

    def rank(candidate: _Candidate) -> tuple[int, int, str]:
        name = candidate.files[0].filename.lower()
        quantization = next(
            (value for value in _QUANTIZATION_PRIORITY if value in name),
            None,
        )
        priority = (
            _QUANTIZATION_PRIORITY[quantization]
            if quantization is not None
            else len(_QUANTIZATION_PRIORITY)
        )
        return (priority, candidate.size, name)

    fitting = [
        candidate
        for candidate in candidates
        if any(value in candidate.files[0].filename.lower() for value in _QUANTIZATION_PRIORITY)
        and _estimated_loaded_size(candidate.size) <= system_memory_bytes
    ]
    chosen = (
        min(fitting, key=rank)
        if fitting
        else min(candidates, key=lambda candidate: (candidate.size, candidate.files[0].filename))
    )
    return [item.filename for item in chosen.files]


def validate_gguf_selection(
    items: Iterable[Mapping[str, Any]],
    *,
    require_split_metadata: bool,
) -> list[str]:
    """Return the ordered model files after validating an explicit selection."""

    records = _records(items)
    candidates, errors = _candidates(records, require_split_metadata=require_split_metadata)
    if errors:
        raise GGUFSelectionError(errors[0])
    if not candidates:
        raise GGUFSelectionError("The chat model selection contains no usable GGUF model.")
    if len(candidates) != 1:
        if any(candidate.split for candidate in candidates):
            raise GGUFSelectionError(
                "The selected GGUF files mix quantizations or shard sets. "
                "Select exactly one complete split set."
            )
        raise GGUFSelectionError(
            "The selected files contain multiple standalone GGUF models. Select exactly one model."
        )
    return [item.filename for item in candidates[0].files]


def automatic_mmproj_selection(
    items: Iterable[Mapping[str, Any]],
    selected_model_files: Iterable[str],
) -> str | None:
    """Select one safe multimodal projector that best matches the chosen model."""

    model_paths = [PurePosixPath(name) for name in selected_model_files]
    if not model_paths:
        return None
    model_tokens = set().union(*(_identity_tokens(path.name) for path in model_paths))
    model_parents = {path.parent.as_posix().casefold() for path in model_paths}
    candidates: list[GGUFFile] = []
    for record in (GGUFFile.from_mapping(item) for item in items):
        path = PurePosixPath(record.filename)
        if (
            not record.filename
            or not record.filename.lower().endswith(".gguf")
            or "mmproj" not in path.name.casefold()
            or "\\" in record.filename
            or path.is_absolute()
            or ".." in path.parts
        ):
            continue
        candidates.append(record)
    if not candidates:
        return None

    def rank(candidate: GGUFFile) -> tuple[int, int, int, int, str]:
        path = PurePosixPath(candidate.filename)
        name = path.name.casefold()
        projector_tokens = _identity_tokens(path.name)
        overlap = len(model_tokens & projector_tokens)
        precision = next(
            (priority for value, priority in _MMPROJ_PRECISION_PRIORITY.items() if value in name),
            len(_MMPROJ_PRECISION_PRIORITY),
        )
        size = candidate.size if candidate.size is not None and candidate.size > 0 else 2**63 - 1
        return (
            0 if path.parent.as_posix().casefold() in model_parents else 1,
            -overlap,
            precision,
            size,
            candidate.filename.casefold(),
        )

    return min(candidates, key=rank).filename


def _records(items: Iterable[Mapping[str, Any]]) -> list[GGUFFile]:
    records: list[GGUFFile] = []
    for item in items:
        record = GGUFFile.from_mapping(item)
        if not record.filename.lower().endswith(".gguf"):
            continue
        if "mmproj" in PurePosixPath(record.filename).name.lower():
            continue
        records.append(record)
    return records


def _identity_tokens(filename: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z]+|\d+b?", filename.casefold())
        if token not in _IDENTITY_IGNORED_TOKENS
    }


def _candidates(
    records: list[GGUFFile],
    *,
    require_split_metadata: bool,
) -> tuple[list[_Candidate], list[str]]:
    candidates: list[_Candidate] = []
    errors: list[str] = []
    split_groups: dict[tuple[str, str, int], list[_SplitPart]] = {}
    group_counts: dict[tuple[str, str], set[int]] = {}
    standalone: dict[str, list[GGUFFile]] = {}

    for record in records:
        path = PurePosixPath(record.filename)
        if (
            not record.filename
            or "\\" in record.filename
            or path.is_absolute()
            or ".." in path.parts
        ):
            errors.append(
                f"GGUF path '{record.filename}' is unsafe. Model files must use relative paths."
            )
            continue
        match = _SPLIT_GGUF.fullmatch(path.name)
        if not match:
            standalone.setdefault(record.filename.casefold(), []).append(record)
            continue
        part = _SplitPart(
            file=record,
            stem=match.group("stem"),
            index=int(match.group("index")),
            count=int(match.group("count")),
        )
        identity = (str(path.parent).casefold(), part.stem.casefold())
        group_counts.setdefault(identity, set()).add(part.count)
        split_groups.setdefault((*identity, part.count), []).append(part)

    for duplicate in standalone.values():
        if len(duplicate) != 1:
            errors.append(
                f"GGUF file '{duplicate[0].filename}' appears more than once in catalog metadata."
            )
            continue
        candidates.append(_Candidate(files=(duplicate[0],), split=False))

    conflicting_identities = {
        identity for identity, counts in group_counts.items() if len(counts) != 1
    }
    for (parent, stem, count), parts in split_groups.items():
        display_stem = str(PurePosixPath(parent) / stem) if parent != "." else stem
        if (parent, stem) in conflicting_identities:
            errors.append(
                f"Split GGUF set '{display_stem}' declares conflicting total shard counts."
            )
            continue
        if count < 1 or any(part.index < 1 or part.index > count for part in parts):
            errors.append(f"Split GGUF set '{display_stem}' has an invalid shard number.")
            continue
        by_index: dict[int, list[_SplitPart]] = {}
        for part in parts:
            by_index.setdefault(part.index, []).append(part)
        duplicated = sorted(index for index, matches in by_index.items() if len(matches) != 1)
        if duplicated:
            rendered = ", ".join(f"{index:05d}" for index in duplicated)
            errors.append(
                f"Split GGUF set '{display_stem}' has duplicate shard number(s): {rendered}."
            )
            continue
        missing = sorted(set(range(1, count + 1)) - set(by_index))
        if missing:
            rendered = ", ".join(f"{index:05d}" for index in missing[:8])
            suffix = "…" if len(missing) > 8 else ""
            errors.append(
                f"Split GGUF set '{display_stem}' is incomplete; "
                f"missing shard(s) {rendered}{suffix} of {count:05d}. "
                "Choose a complete quantization or refresh the model catalog."
            )
            continue
        ordered = tuple(by_index[index][0].file for index in range(1, count + 1))
        if require_split_metadata:
            bad_sizes = [item.filename for item in ordered if item.size is None or item.size <= 0]
            if bad_sizes:
                errors.append(
                    f"Split GGUF set '{display_stem}' has missing or invalid size metadata "
                    f"for {bad_sizes[0]}. Refresh the model catalog before installing."
                )
                continue
            bad_hashes = [
                item.filename
                for item in ordered
                if item.sha256 is None or not _SHA256.fullmatch(item.sha256)
            ]
            if bad_hashes:
                errors.append(
                    f"Split GGUF set '{display_stem}' has missing or invalid SHA-256 metadata "
                    f"for {bad_hashes[0]}. Refresh the model catalog before installing."
                )
                continue
        candidates.append(_Candidate(files=ordered, split=True))

    return candidates, errors


def _estimated_loaded_size(size: int) -> int:
    return int(size * 1.2) + 512 * 1024**2
