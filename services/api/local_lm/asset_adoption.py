"""Register an auxiliary model file the runtime can already load.

Every asset this application knows about arrives the same way: it downloads the
file, verifies it against the digest its plan declared, and records what it
installed. That is the right default, and it leaves one case with no answer at
all. A file already sitting where the runtime reads - put there by hand, or by
another tool, or carried over from an older install - is invisible: nothing
holds its digest, so nothing will accept it, however correct its bytes are.

The same shape cost an imported chat model its usability until the activation
guard was repaired: the files were fine and nothing could register them.

Adoption answers it without inventing a second install path. The file must
already be in one of the folders the runtime is told to load that kind from,
which is the whole precondition - if the runtime cannot see it, neither the
digest nor this record would make it loadable. What this adds is the record:
the digest measured from the bytes on disk, the name the graph will use, the
family the file declares, and whatever its own metadata says about how it wants
to be driven.

Nothing here trusts the caller about the file's contents. The name is checked
to be a single file inside one of those folders, the digest is computed rather
than accepted, and the metadata is read from the file's own header under a
bound.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .auxiliary_assets import MAX_LORA_TRIGGER_WORD_LENGTH, MAX_LORA_TRIGGER_WORDS
from .config import Settings
from .model_manifests import comfy_folder_for_kind
from .models import ModelAssetInstall, ModelInstall

#: How much of a safetensors header to read. The header is a length-prefixed
#: JSON object at the front of the file; a legitimate one for a LoRA is a few
#: kilobytes, and a file claiming a header larger than this is refused rather
#: than read into memory.
MAX_HEADER_BYTES = 2 * 1024 * 1024

#: The longest header a loader will read. The reference safetensors reader
#: refuses one longer than this, so a file claiming more cannot be loaded at all,
#: whatever the metadata cap above says about how much of it is worth describing.
MAX_LOADABLE_HEADER_BYTES = 100_000_000

#: How much of a file to read at once while measuring its digest.
_DIGEST_BLOCK = 1024 * 1024

#: The suffixes a model file may carry. Pickle-bearing formats are refused for
#: the same reason the import route refuses them: loading one executes it.
ADOPTABLE_SUFFIXES = frozenset({".safetensors", ".sft", ".gguf"})


class AssetAdoptionError(ValueError):
    """A file cannot be adopted, with a reason fit to show a person."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class AdoptedFile:
    """What was measured about a file, before anything is written down."""

    comfy_name: str
    path: Path
    sha256: str
    size_bytes: int
    metadata: dict[str, Any]

    @property
    def declared_family(self) -> str | None:
        """The family the file's own metadata claims, when it claims one."""

        architecture = self.metadata.get("modelspec.architecture")
        if isinstance(architecture, str) and architecture:
            return architecture.split("/", 1)[0] or None
        return None

    @property
    def trigger_words(self) -> list[str]:
        """Activation tokens the file declares, in the order it declares them.

        Only from keys that are a declaration of what to type. A training tag
        frequency table is not one: it records how often each caption word
        appeared while the file was trained, and reading it as a trigger list
        puts that table into somebody's prompt, because trigger words are added
        to the prompt when the LoRA is applied.

        A LoRA driven by strength alone declares nothing, and saying so is
        useful: the reader is told there is no keyword to add rather than left
        to wonder whether one was missed.

        Bounded the way the stack reader bounds them, so adoption cannot record
        a vocabulary that the run would later refuse.
        """

        words: list[str] = []
        seen: set[str] = set()
        for key in ("modelspec.trigger_phrase", "ss_trigger_words"):
            raw = self.metadata.get(key)
            if not isinstance(raw, str):
                continue
            for part in raw.split(","):
                word = part.strip()
                if not word or len(word) > MAX_LORA_TRIGGER_WORD_LENGTH:
                    continue
                folded = word.casefold()
                if folded in seen or len(words) == MAX_LORA_TRIGGER_WORDS:
                    continue
                seen.add(folded)
                words.append(word)
        return words


def adoptable_roots(session: Session, settings: Settings, kind: str) -> list[Path]:
    """Every folder the runtime is already told to load ``kind`` from.

    The application does not keep its assets in one place. It writes the media
    runtime an extra-paths file naming each installed model's folder and each
    registered asset's own folder, so a LoRA loads from any of several
    directories and the runtime's own folder is only one of them.

    Adoption searches exactly that set and nothing wider. A file outside it
    cannot be adopted, because a record would not make it loadable; a file
    inside it needs no new directory to be exposed, because the runtime was
    already reading there. So adoption gives the runtime no reach it did not
    already have - it only writes down what one of the files it reaches is.
    """

    folder = comfy_folder_for_kind(kind)
    if folder is None:
        return []
    roots: list[Path] = []
    seen: set[Path] = set()

    def remember(candidate: Path) -> None:
        try:
            resolved = candidate.resolve()
        except OSError:
            return
        if resolved not in seen and resolved.is_dir():
            seen.add(resolved)
            roots.append(resolved)

    if settings.comfy_directory is not None:
        remember(settings.comfy_directory / "models" / folder)
    installs = session.scalars(
        select(ModelInstall)
        .where(ModelInstall.engine == "comfyui", ModelInstall.active.is_(True))
        .order_by(ModelInstall.id)
    ).all()
    for install in installs:
        declared = install.manifest_json.get("comfy_paths")
        relative = declared.get(folder) if isinstance(declared, dict) else None
        if not isinstance(relative, str) or not relative:
            continue
        step = PurePosixPath(relative.replace("\\", "/"))
        if step.is_absolute() or ".." in step.parts:
            continue
        remember(Path(install.local_path).joinpath(*step.parts))
    assets = session.scalars(
        select(ModelAssetInstall)
        .where(ModelAssetInstall.kind == kind, ModelAssetInstall.active.is_(True))
        .order_by(ModelAssetInstall.id)
    ).all()
    for asset in assets:
        remember(Path(asset.local_path))
    return roots


def resolve_adoptable_path(roots: Sequence[Path], comfy_name: str) -> Path:
    """The one file ``comfy_name`` names among ``roots``, or a refusal.

    ``comfy_name`` is what the graph will pass to the loader, so it may carry
    the forward slashes a nested folder needs. It may not climb out of a root,
    name an absolute location, or reach through a link: what the runtime loads
    and what was measured here have to be the same file.

    A name that more than one root answers is refused rather than resolved. The
    runtime chooses between them by its own precedence, and guessing at that
    would record the digest of a file it does not load. A verification that
    certifies the wrong bytes is worse than no verification at all.
    """

    if not comfy_name or comfy_name.strip() != comfy_name:
        raise AssetAdoptionError("asset-name-invalid", "Name the file as the workflow will.")
    pure = PurePosixPath(comfy_name.replace("\\", "/"))
    if pure.is_absolute() or any(part in {"..", ""} for part in pure.parts):
        raise AssetAdoptionError(
            "asset-name-invalid", "Name a file inside the folder, without any path steps."
        )
    if pure.suffix.casefold() not in ADOPTABLE_SUFFIXES:
        raise AssetAdoptionError(
            "asset-format-unsupported",
            "Adopt a safetensors or GGUF file; formats that execute on load are refused.",
        )
    found: list[Path] = []
    for root in roots:
        resolved_root = root.resolve()
        candidate = (resolved_root / Path(*pure.parts)).resolve()
        if resolved_root not in candidate.parents:
            continue
        if candidate.is_file() and candidate not in found:
            found.append(candidate)
    if len(found) > 1:
        raise AssetAdoptionError(
            "asset-name-ambiguous",
            "More than one folder the runtime reads holds this name. Rename one of them.",
        )
    if not found:
        raise AssetAdoptionError(
            "asset-file-missing",
            "No folder this kind loads from holds that file. Place it in one of them first.",
        )
    return found[0]


def read_safetensors_metadata(path: Path) -> dict[str, Any]:
    """The ``__metadata__`` a safetensors file carries, or nothing.

    Read rather than trusted: the length prefix is bounded before it is used,
    and anything unreadable answers with no metadata. Whether the file can be
    loaded at all is settled before this is asked, by
    `require_loadable_safetensors`; a header that parses and simply carries no
    metadata answers with nothing here and is still adopted.
    """

    if path.suffix.casefold() not in {".safetensors", ".sft"}:
        return {}
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) < 8:
                return {}
            length = int.from_bytes(prefix, "little")
            if length <= 0 or length > MAX_HEADER_BYTES:
                return {}
            header = json.loads(handle.read(length).decode("utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    if not isinstance(header, dict):
        return {}
    metadata = header.get("__metadata__")
    return (
        {str(key): value for key, value in metadata.items()} if isinstance(metadata, dict) else {}
    )


def require_loadable_safetensors(path: Path, size_bytes: int) -> None:
    """Refuse a safetensors file no loader could read, before it is recorded.

    A loader reads the length prefix, parses the header it bounds as a JSON
    object, and maps each tensor's byte range onto the data that follows, which
    the ranges must cover exactly, with no gap and nothing left over. A file
    failing any of that is not a model the runtime can load, whatever its name
    says; a failed download saved under a model's name is the usual case.
    Recording one as verified only moves the failure to the moment someone runs
    it, where it surfaces as a decoding error inside the runtime.
    """

    refusal = AssetAdoptionError(
        "asset-file-not-loadable",
        "This file is not a model the runtime can load. "
        "It may be an incomplete or failed download.",
    )
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            length = int.from_bytes(prefix, "little") if len(prefix) == 8 else 0
            fits = 0 < length <= MAX_LOADABLE_HEADER_BYTES and 8 + length <= size_bytes
            raw = handle.read(length) if fits else b""
    except OSError as exc:
        raise AssetAdoptionError("asset-file-unreadable", "The file could not be read.") from exc
    if not fits:
        raise refusal
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise refusal from exc
    if not isinstance(header, dict):
        raise refusal
    ranges: list[tuple[int, int]] = []
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("dtype"), str)
            or not isinstance(entry.get("shape"), list)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
            or not 0 <= offsets[0] <= offsets[1]
        ):
            raise refusal
        ranges.append((offsets[0], offsets[1]))
    covered = 0
    for start, stop in sorted(ranges):
        if start != covered:
            raise refusal
        covered = stop
    if 8 + length + covered != size_bytes:
        raise refusal


def measure_adoptable_file(roots: Sequence[Path], comfy_name: str) -> AdoptedFile:
    """Measure a file already in place: its digest, its size, what it says."""

    path = resolve_adoptable_path(roots, comfy_name)
    digest = hashlib.sha256()
    try:
        size_bytes = path.stat().st_size
        if path.suffix.casefold() in {".safetensors", ".sft"}:
            require_loadable_safetensors(path, size_bytes)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(_DIGEST_BLOCK), b""):
                digest.update(block)
    except OSError as exc:
        raise AssetAdoptionError("asset-file-unreadable", "The file could not be read.") from exc
    return AdoptedFile(
        comfy_name=comfy_name.replace("\\", "/"),
        path=path,
        sha256=digest.hexdigest(),
        size_bytes=size_bytes,
        metadata=read_safetensors_metadata(path),
    )
