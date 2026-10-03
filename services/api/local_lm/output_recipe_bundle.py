"""A generated picture and its record in one file: lm-atelier-output-recipe-bundle-v1.

The bundle is a ZIP of exactly three entries: the record as its own canonical
bytes, a PNG copy that holds the picture's pixels and nothing else, and a small
manifest naming both by hash. The copy is needed because a picture as an engine
saved it can carry the graph that made it - prompts and the names this computer
gave its staged inputs among them - and a color profile can carry text of its
own, so the stored bytes are never shared. A picture with a color profile is
converted to sRGB first, so the copy needs none.

A copy is not the stored file, so the manifest says so: it names the picture's
own hash and the hash of the stored file the record describes, and nothing in
the bundle pretends they are the same bytes. The manifest is canonical and
digested like the record, under its own domain tag. The digest shows that the
bundle is intact, not who made it.

A bundle has one exact form, and reading one refuses every other: the reader
checks the archive's closing record before parsing anything else, rebuilds the
archive from the three files it read and compares it with the bytes it was
given, and checks that the picture is a PNG of only a header, pixel data and
an end, with a header the copy could have been written with.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import re
import struct
import zipfile
import zlib
from dataclasses import dataclass
from typing import Any, Final

from PIL import Image, ImageCms, ImageOps
from sqlalchemy.orm import Session

from .artifacts import ArtifactStore
from .models import Artifact
from .output_recipe import OutputRecipeUnavailable, build_output_recipe
from .output_recipe_v1 import (
    MAX_RECORD_BYTES,
    OutputRecipeFormatError,
    canonical_bytes,
    open_output_recipe,
)
from .studio_region_edit import (
    MAX_BLEND_PIXELS,
    MAX_BLEND_READ_BYTES,
    RegionEditError,
    decode_picture,
)

BUNDLE_SCHEMA_ID: Final = "lm-atelier-output-recipe-bundle-v1"
BUNDLE_SCHEMA_VERSION: Final = 1
BUNDLE_DIGEST_DOMAIN: Final = BUNDLE_SCHEMA_ID.encode("ascii") + b"\0"
RECORD_FILE: Final = "generation-record.json"
PICTURE_FILE: Final = "output.png"
MANIFEST_FILE: Final = "bundle.json"
MAX_MANIFEST_BYTES: Final = 4 * 1024
#: The largest picture the copy decodes, as RGBA rows that do not compress at
#: all, with room left for the row filters and the stream framing.
MAX_PICTURE_BYTES: Final = 5 * MAX_BLEND_PIXELS
#: The three entries at their largest, with room for the archive's own headers.
MAX_BUNDLE_BYTES: Final = MAX_MANIFEST_BYTES + MAX_RECORD_BYTES + MAX_PICTURE_BYTES + 4096
_ENTRIES: Final = (MANIFEST_FILE, RECORD_FILE, PICTURE_FILE)
_BOUNDS: Final = {
    MANIFEST_FILE: MAX_MANIFEST_BYTES,
    RECORD_FILE: MAX_RECORD_BYTES,
    PICTURE_FILE: MAX_PICTURE_BYTES,
}
#: Every entry carries this time and these attributes, so one bundle has one
#: set of bytes on every computer and says nothing about the one that made it.
_ENTRY_TIME: Final = (1980, 1, 1, 0, 0, 0)
_UNIX: Final = 3
_REGULAR_FILE: Final = 0o100644 << 16
_PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"
#: The bit depth and color type of each mode the copy is written in: one-bit,
#: eight-bit and sixteen-bit grey, grey with alpha, RGB and RGBA.
_PNG_LAYOUTS: Final = frozenset({(1, 0), (8, 0), (16, 0), (8, 4), (8, 2), (8, 6)})
#: The archive's closing record: its signature, two disk numbers, the entry
#: counts on this disk and in all, the directory's size and offset, and the
#: length of the archive comment.
_END_RECORD: Final = struct.Struct("<4sHHHHIIH")
#: Three directory records with names far longer than these, with room to spare.
_MAX_DIRECTORY_BYTES: Final = 512
#: The modes a PNG holds exactly; a picture in any other becomes RGB or RGBA.
_EXACT_MODES: Final = frozenset({"1", "L", "LA", "I;16", "RGB", "RGBA"})
_SIXTEEN_BIT_MODES: Final = frozenset({"I", "I;16B", "I;16L", "I;16N"})
_COLOR_PROFILE_MODES: Final = frozenset({"RGB", "RGBA", "CMYK", "P", "PA"})

_HEX64 = re.compile(r"[0-9a-f]{64}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_MANIFEST_KEYS: Final = frozenset({"schema", "version", "record", "picture", "digest"})
_RECORD_KEYS: Final = frozenset({"file", "sha256", "digest"})
_PICTURE_KEYS: Final = frozenset({"file", "sha256", "media_type", "copy_of", "metadata"})


class OutputRecipeBundleFormatError(ValueError):
    """A file that is not exactly a version 1 bundle. The message never echoes input."""


@dataclass(frozen=True)
class OutputRecipeBundle:
    """A sealed bundle, the digest of the record inside it, and the name it downloads under."""

    content: bytes
    record_digest: str
    file_name: str


def build_output_recipe_bundle(
    session: Session,
    artifacts: ArtifactStore,
    *,
    run_id: str,
    artifact_id: str,
    include_prompts: bool,
    expected_record_digest: str,
) -> OutputRecipeBundle:
    """Bundle one picture's record with a copy that holds only the picture's pixels.

    The record must be the one the caller was shown, named by its digest: a
    record rebuilt differently since then is refused rather than bundled unseen.
    """

    recipe = build_output_recipe(
        session,
        artifacts,
        run_id=run_id,
        artifact_id=artifact_id,
        include_prompts=include_prompts,
    )
    if not hmac.compare_digest(recipe.digest, expected_record_digest):
        raise OutputRecipeUnavailable(
            409,
            "output-recipe-changed",
            "This output's record changed since it was shown. Show it again.",
        )
    record = open_output_recipe(recipe.content)
    if record["output"]["kind"] != "image" or not record["output"]["media_type"].startswith(
        "image/"
    ):
        raise OutputRecipeUnavailable(
            422,
            "output-recipe-bundle-picture-only",
            "Only a picture can be saved together with its record.",
        )
    artifact = session.get(Artifact, artifact_id)
    if artifact is None:
        raise OutputRecipeUnavailable(
            404, "output-recipe-output-not-found", "This generation has no such output."
        )
    try:
        stored = artifacts.verified_bytes(artifact, maximum_bytes=MAX_BLEND_READ_BYTES)
    except (ValueError, OSError) as exc:
        raise OutputRecipeUnavailable(
            410, "output-recipe-output-unreadable", "This output's file is missing or changed."
        ) from exc
    picture = pixels_only_png(stored)
    manifest = seal_bundle_manifest(
        {
            "schema": BUNDLE_SCHEMA_ID,
            "version": BUNDLE_SCHEMA_VERSION,
            "record": {
                "file": RECORD_FILE,
                "sha256": hashlib.sha256(recipe.content).hexdigest(),
                "digest": recipe.digest,
            },
            "picture": {
                "file": PICTURE_FILE,
                "sha256": hashlib.sha256(picture).hexdigest(),
                "media_type": "image/png",
                "copy_of": record["output"]["sha256"],
                "metadata": "none",
            },
        }
    )
    content = _zip({MANIFEST_FILE: manifest, RECORD_FILE: recipe.content, PICTURE_FILE: picture})
    open_output_recipe_bundle(content)
    return OutputRecipeBundle(
        content=content,
        record_digest=recipe.digest,
        file_name=f"generation-record-{record['output']['sha256'][:12]}.zip",
    )


def pixels_only_png(payload: bytes) -> bytes:
    """The picture upright, as a PNG that holds its pixels and nothing else.

    A color profile is applied, converting the pixels to sRGB, rather than
    copied; one that cannot be read is left out and the pixels kept as they
    are. Sixteen-bit grey stays sixteen-bit. A picture with more than one frame
    is refused rather than cut down to its first.
    """

    try:
        decoded = decode_picture(payload, "picture")
        if getattr(decoded, "n_frames", 1) != 1:
            raise _unreadable()
        upright = ImageOps.exif_transpose(decoded) or decoded
        picture = _in_srgb(upright, upright.info.get("icc_profile"))
        # Conversion carries the source's metadata along; none of it is wanted.
        picture.info = {}
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
    except OutputRecipeUnavailable:
        raise
    except RegionEditError as exc:
        if exc.code == "region-image-too-large":
            raise OutputRecipeUnavailable(
                422,
                "output-recipe-bundle-too-large",
                "This picture is too large to save together with its record.",
            ) from exc
        raise _unreadable() from exc
    except Exception as exc:
        # A malformed file can make Pillow raise almost anything while it
        # decodes, turns or converts the picture.
        raise _unreadable() from exc
    copy = buffer.getvalue()
    if len(copy) > MAX_PICTURE_BYTES:
        raise _unreadable()
    return copy


def _in_srgb(upright: Image.Image, profile: object) -> Image.Image:
    alpha = upright.mode in {"RGBA", "LA", "PA"} or "transparency" in upright.info
    # A grey profile is left out rather than applied, so grey stays grey at its
    # own depth.
    if isinstance(profile, bytes) and profile and upright.mode in _COLOR_PROFILE_MODES:
        source = (
            upright
            if upright.mode in {"RGB", "RGBA", "CMYK"}
            else upright.convert("RGBA" if alpha else "RGB")
        )
        try:
            converted = ImageCms.profileToProfile(
                source,
                ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                ImageCms.createProfile("sRGB"),
                outputMode="RGBA" if alpha else "RGB",
            )
        except (OSError, ValueError):
            converted = None
        if isinstance(converted, Image.Image):
            return converted
    if upright.mode in _EXACT_MODES:
        return upright
    if upright.mode in _SIXTEEN_BIT_MODES:
        return upright.convert("I;16")
    return upright.convert("RGBA" if alpha else "RGB")


def _unreadable() -> OutputRecipeUnavailable:
    return OutputRecipeUnavailable(
        422,
        "output-recipe-bundle-unreadable",
        "This picture could not be read to copy it.",
    )


def bundle_manifest_digest(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "digest"}
    return "sha256:" + hashlib.sha256(BUNDLE_DIGEST_DOMAIN + canonical_bytes(unsigned)).hexdigest()


def seal_bundle_manifest(payload: dict[str, Any]) -> bytes:
    return canonical_bytes({**payload, "digest": bundle_manifest_digest(payload)})


def open_output_recipe_bundle(content: bytes) -> dict[str, Any]:
    """Read a bundle, refusing anything this module would not have written.

    Returns the manifest, the record, the picture's bytes and its size.
    """

    if len(content) > MAX_BUNDLE_BYTES:
        raise OutputRecipeBundleFormatError("The bundle is larger than a bundle can be.")
    files = _read_entries(content)
    # Nothing about the archive is left to chance: the same three files make
    # exactly these bytes, so a comment, a prefix, another compression, a
    # timestamp or any field this module never writes is refused here.
    if _zip(files) != content:
        raise OutputRecipeBundleFormatError("The bundle is not in its one exact form.")
    manifest = _open_manifest(files[MANIFEST_FILE])
    try:
        record = open_output_recipe(files[RECORD_FILE])
    except OutputRecipeFormatError as exc:
        raise OutputRecipeBundleFormatError("The bundle's record is not a valid record.") from exc
    if hashlib.sha256(files[RECORD_FILE]).hexdigest() != manifest["record"]["sha256"]:
        raise OutputRecipeBundleFormatError("The bundle's record does not match its manifest.")
    if record["digest"] != manifest["record"]["digest"]:
        raise OutputRecipeBundleFormatError("The bundle's record does not match its manifest.")
    if hashlib.sha256(files[PICTURE_FILE]).hexdigest() != manifest["picture"]["sha256"]:
        raise OutputRecipeBundleFormatError("The bundle's picture does not match its manifest.")
    if manifest["picture"]["copy_of"] != record["output"]["sha256"]:
        raise OutputRecipeBundleFormatError("The bundle's picture is not of the recorded output.")
    width, height = _check_pixels_only(files[PICTURE_FILE])
    return {
        "manifest": manifest,
        "record": record,
        "picture": files[PICTURE_FILE],
        "width": width,
        "height": height,
    }


def _read_entries(content: bytes) -> dict[str, bytes]:
    # Read before zipfile sees the archive: it builds a record for every entry
    # the directory lists, so a file listing millions of empty entries would
    # cost far more than its own size to refuse.
    if len(content) < _END_RECORD.size:
        raise OutputRecipeBundleFormatError("The file is not a generation record bundle.")
    signature, disk, directory_disk, here, listed, size, offset, comment = _END_RECORD.unpack(
        content[-_END_RECORD.size :]
    )
    if (
        signature != b"PK\x05\x06"
        or disk
        or directory_disk
        or here != len(_ENTRIES)
        or listed != len(_ENTRIES)
        or size > _MAX_DIRECTORY_BYTES
        or offset + size != len(content) - _END_RECORD.size
        or comment
    ):
        raise OutputRecipeBundleFormatError("The file is not a generation record bundle.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except (zipfile.BadZipFile, ValueError, NotImplementedError, RuntimeError, EOFError) as exc:
        raise OutputRecipeBundleFormatError("The file is not a generation record bundle.") from exc
    with archive:
        entries = archive.infolist()
        if tuple(entry.orig_filename for entry in entries) != _ENTRIES:
            raise OutputRecipeBundleFormatError("The bundle holds the wrong entries.")
        files: dict[str, bytes] = {}
        for entry in entries:
            bound = _BOUNDS[entry.orig_filename]
            # Checked before a byte is read: an entry stored as written is read
            # as exactly the bytes it declares, never decompressed or decrypted.
            if (
                entry.filename != entry.orig_filename
                or entry.compress_type != zipfile.ZIP_STORED
                or entry.flag_bits != 0
                or entry.extra
                or entry.comment
                or entry.external_attr != _REGULAR_FILE
                or entry.file_size > bound
                or entry.compress_size != entry.file_size
            ):
                raise OutputRecipeBundleFormatError("A bundle entry is not as it was written.")
            try:
                with archive.open(entry) as handle:
                    data = handle.read(bound + 1)
            except (
                zipfile.BadZipFile,
                OSError,
                ValueError,
                NotImplementedError,
                RuntimeError,
                EOFError,
            ) as exc:
                raise OutputRecipeBundleFormatError("A bundle entry cannot be read.") from exc
            if len(data) != entry.file_size:
                raise OutputRecipeBundleFormatError("A bundle entry is not as it was written.")
            files[entry.orig_filename] = data
    return files


def _open_manifest(content: bytes) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("ascii"))
        canonical = canonical_bytes(value)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        # ValueError covers malformed JSON, a number too long to read, and a
        # value such as NaN that canonical JSON cannot carry.
        raise OutputRecipeBundleFormatError("The bundle's manifest is not valid.") from exc
    if not isinstance(value, dict) or canonical != content:
        raise OutputRecipeBundleFormatError("The bundle's manifest is not in canonical form.")
    if set(value) != _MANIFEST_KEYS:
        raise OutputRecipeBundleFormatError("The bundle's manifest has the wrong fields.")
    if (
        value["schema"] != BUNDLE_SCHEMA_ID
        or type(value["version"]) is not int
        or value["version"] != BUNDLE_SCHEMA_VERSION
    ):
        raise OutputRecipeBundleFormatError("The bundle is not a version 1 bundle.")
    record = value["record"]
    picture = value["picture"]
    if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
        raise OutputRecipeBundleFormatError("The bundle's manifest has the wrong fields.")
    if not isinstance(picture, dict) or set(picture) != _PICTURE_KEYS:
        raise OutputRecipeBundleFormatError("The bundle's manifest has the wrong fields.")
    if (
        record["file"] != RECORD_FILE
        or picture["file"] != PICTURE_FILE
        or picture["media_type"] != "image/png"
        or picture["metadata"] != "none"
        or not all(
            isinstance(item, str) and _HEX64.fullmatch(item)
            for item in (record["sha256"], picture["sha256"], picture["copy_of"])
        )
        or not isinstance(record["digest"], str)
        or not _DIGEST.fullmatch(record["digest"])
        or not isinstance(value["digest"], str)
        or not _DIGEST.fullmatch(value["digest"])
    ):
        raise OutputRecipeBundleFormatError("The bundle's manifest is malformed.")
    if not hmac.compare_digest(bundle_manifest_digest(value), value["digest"]):
        raise OutputRecipeBundleFormatError("The bundle's manifest does not match its digest.")
    return value


def _check_pixels_only(png: bytes) -> tuple[int, int]:
    """Refuse a picture that is not a PNG of a header, pixel data and an end, intact.

    The header must describe a picture the copy could have been: a size inside
    the decode limit and a layout one of the copy's modes is written in.
    Returns the width and the height.
    """

    if not png.startswith(_PNG_SIGNATURE):
        raise OutputRecipeBundleFormatError("The bundle's picture is not a PNG.")
    kinds: list[bytes] = []
    offset = len(_PNG_SIGNATURE)
    while offset < len(png):
        if offset + 12 > len(png):
            raise OutputRecipeBundleFormatError("The bundle's picture is not a PNG.")
        (length,) = struct.unpack(">I", png[offset : offset + 4])
        end = offset + 12 + length
        kind = png[offset + 4 : offset + 8]
        if end > len(png) or zlib.crc32(png[offset + 4 : end - 4]) != int.from_bytes(
            png[end - 4 : end], "big"
        ):
            raise OutputRecipeBundleFormatError("The bundle's picture is not a PNG.")
        kinds.append(kind)
        offset = end
    if (
        len(kinds) < 3
        or kinds[0] != b"IHDR"
        or kinds[-1] != b"IEND"
        or any(kind != b"IDAT" for kind in kinds[1:-1])
    ):
        raise OutputRecipeBundleFormatError("The bundle's picture holds more than its pixels.")
    header = png[len(_PNG_SIGNATURE) : len(_PNG_SIGNATURE) + 8 + 13]
    if struct.unpack(">I", header[:4])[0] != 13:
        raise OutputRecipeBundleFormatError("The bundle's picture has a malformed header.")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", header[8:]
    )
    if (
        not 0 < width * height <= MAX_BLEND_PIXELS
        or (depth, color) not in _PNG_LAYOUTS
        or compression
        or filtering
        or interlace
    ):
        raise OutputRecipeBundleFormatError("The bundle's picture has a malformed header.")
    return width, height


def _zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in _ENTRIES:
            info = zipfile.ZipInfo(name, date_time=_ENTRY_TIME)
            info.create_system = _UNIX
            info.external_attr = _REGULAR_FILE
            archive.writestr(info, files[name])
    return buffer.getvalue()
