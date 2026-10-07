"""Download the portable record of how one generated output was made."""

from __future__ import annotations

import hmac
import io
from typing import TYPE_CHECKING, Annotated, Any, BinaryIO, Literal, cast
from urllib.parse import quote

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from .api_errors import api_error
from .db import SessionLocal, get_session
from .external_generation_metadata import (
    ExternalGenerationMetadata,
    metadata_not_read,
    read_external_generation_metadata,
)
from .models import Artifact, Run
from .output_recipe import OutputRecipe, OutputRecipeUnavailable, build_output_recipe
from .output_recipe_bundle import OutputRecipeBundle, build_output_recipe_bundle
from .output_recipe_check import (
    BUNDLE_SIGNATURE,
    MAX_CHECK_BYTES,
    OutputRecipeCheckRefused,
    check_output_recipe_file,
    most_read_for,
    read_record_file,
)
from .output_recipe_promotion import (
    EditRecipeDraftOut,
    OutputRecipeDraftOut,
    OutputRecipeDraftRefused,
    edit_recipe_draft,
    output_recipe_draft,
)
from .output_recipe_replay import plan_output_recipe_replay, replay_outcome
from .picture_workflow import read_picture_workflow
from .portable_archive_v1 import ArchiveKind, ArchiveRefused, open_archive
from .project_archive_encryption import (
    ExportUnverified,
    archive_api_error,
    encrypt_bytes,
    passphrase_from_header,
)
from .schemas import EncryptedOutputRecipeBundleRequest, EncryptedOutputRecipeRequest
from .studio_region_edit import MAX_BLEND_READ_BYTES

if TYPE_CHECKING:
    from .artifacts import ArtifactStore
    from .main import Services

router = APIRouter()


@router.get("/runs/{run_id}/outputs/{artifact_id}/recipe")
async def download_output_recipe(
    run_id: str,
    artifact_id: str,
    request: Request,
    prompts: Annotated[Literal["include", "omit"], Query()],
) -> Response:
    """The record of one output, as the exact canonical bytes a reader checks.

    Its digest covers those bytes without the digest field, behind the format's
    domain tag, and is repeated in a response header.

    The output is named by the run that made it as well as by its hash, because
    the same bytes can come from more than one run. Whether the prompt goes in
    is always the caller's explicit choice; there is no default.
    """

    artifacts = cast("Services", request.app.state.services).artifacts
    try:
        recipe = await run_in_threadpool(
            _build, artifacts, run_id, artifact_id, prompts == "include"
        )
    except OutputRecipeUnavailable as exc:
        raise api_error(exc.status, exc.code, exc.message) from exc
    name = quote(recipe.file_name, safe="")
    return Response(
        recipe.content,
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename*=utf-8''{name}",
            "Cache-Control": "private, no-store",
            "Content-Security-Policy": "sandbox; default-src 'none'",
            "Cross-Origin-Resource-Policy": "same-origin",
            "X-Content-Type-Options": "nosniff",
            "X-Output-Recipe-Digest": recipe.digest,
        },
    )


def _build(
    artifacts: ArtifactStore, run_id: str, artifact_id: str, include_prompts: bool
) -> OutputRecipe:
    # Hashing the output reads the whole file, so the record is built off the
    # event loop with a session of its own.
    with SessionLocal() as session:
        return build_output_recipe(
            session,
            artifacts,
            run_id=run_id,
            artifact_id=artifact_id,
            include_prompts=include_prompts,
        )


@router.get("/runs/{run_id}/outputs/{artifact_id}/recipe-bundle")
async def download_output_recipe_bundle(
    run_id: str,
    artifact_id: str,
    request: Request,
    prompts: Annotated[Literal["include", "omit"], Query()],
    digest: Annotated[str, Query(pattern=r"^sha256:[0-9a-f]{64}$")],
    inputs: Annotated[Literal["include", "omit"], Query()] = "omit",
) -> Response:
    """One picture's record and a copy of the picture, together in one ZIP file.

    The record must be the one the caller was shown, named by its digest, so
    the file never holds a record nobody looked at. The picture is a copy
    without the text an engine embeds in its file, never the stored bytes.
    Its input pictures go in only when asked for, copied the same way.
    """

    artifacts = cast("Services", request.app.state.services).artifacts
    try:
        bundle = await run_in_threadpool(
            _build_bundle,
            artifacts,
            run_id,
            artifact_id,
            prompts == "include",
            digest,
            inputs == "include",
        )
    except OutputRecipeUnavailable as exc:
        raise api_error(exc.status, exc.code, exc.message) from exc
    name = quote(bundle.file_name, safe="")
    return Response(
        bundle.content,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename*=utf-8''{name}",
            "Cache-Control": "private, no-store",
            "Content-Security-Policy": "sandbox; default-src 'none'",
            "Cross-Origin-Resource-Policy": "same-origin",
            "X-Content-Type-Options": "nosniff",
            "X-Output-Recipe-Digest": bundle.record_digest,
        },
    )


@router.post("/runs/{run_id}/outputs/{artifact_id}/recipe/encrypted")
async def download_encrypted_output_recipe(
    run_id: str,
    artifact_id: str,
    payload: EncryptedOutputRecipeRequest,
    request: Request,
) -> Response:
    """The record the caller was shown, encrypted under the passphrase it typed.

    The passphrase travels in the body, never the address. The record is built
    again and must be the one named by its digest, so the file never holds a
    record nobody looked at. It is encrypted in memory and opened again before
    it is sent; nothing of it is written to disk.
    """

    artifacts = cast("Services", request.app.state.services).artifacts
    try:
        content, file_name, digest = await run_in_threadpool(
            _encrypted_record, artifacts, run_id, artifact_id, payload
        )
    except OutputRecipeUnavailable as exc:
        raise api_error(exc.status, exc.code, exc.message) from exc
    except ArchiveRefused as exc:
        raise archive_api_error(exc) from exc
    except ExportUnverified as exc:
        raise _unverified() from exc
    return _encrypted_download(content, file_name, digest)


@router.post("/runs/{run_id}/outputs/{artifact_id}/recipe-bundle/encrypted")
async def download_encrypted_output_recipe_bundle(
    run_id: str,
    artifact_id: str,
    payload: EncryptedOutputRecipeBundleRequest,
    request: Request,
) -> Response:
    """A picture's record and its clean copy in one ZIP, encrypted under a passphrase.

    The same bundle the plain download makes, from the record named by its
    digest, sealed in memory and opened again before it is sent.
    """

    artifacts = cast("Services", request.app.state.services).artifacts
    try:
        content, file_name, digest = await run_in_threadpool(
            _encrypted_bundle, artifacts, run_id, artifact_id, payload
        )
    except OutputRecipeUnavailable as exc:
        raise api_error(exc.status, exc.code, exc.message) from exc
    except ArchiveRefused as exc:
        raise archive_api_error(exc) from exc
    except ExportUnverified as exc:
        raise _unverified() from exc
    return _encrypted_download(content, file_name, digest)


def _encrypted_record(
    artifacts: ArtifactStore, run_id: str, artifact_id: str, payload: EncryptedOutputRecipeRequest
) -> tuple[bytes, str, str]:
    recipe = _build(artifacts, run_id, artifact_id, payload.prompts == "include")
    if not hmac.compare_digest(recipe.digest, payload.digest):
        raise OutputRecipeUnavailable(
            409,
            "output-recipe-changed",
            "This output's record changed since it was shown. Show it again.",
        )
    sealed = encrypt_bytes(
        recipe.content, kind=ArchiveKind.OUTPUT_RECIPE, passphrase=payload.passphrase.encode()
    )
    return sealed, f"{recipe.file_name}.encrypted", recipe.digest


def _encrypted_bundle(
    artifacts: ArtifactStore,
    run_id: str,
    artifact_id: str,
    payload: EncryptedOutputRecipeBundleRequest,
) -> tuple[bytes, str, str]:
    bundle = _build_bundle(
        artifacts,
        run_id,
        artifact_id,
        payload.prompts == "include",
        payload.digest,
        payload.inputs == "include",
    )
    sealed = encrypt_bytes(
        bundle.content, kind=ArchiveKind.OUTPUT_RECIPE, passphrase=payload.passphrase.encode()
    )
    return sealed, f"{bundle.file_name}.encrypted", bundle.record_digest


def _unverified() -> Exception:
    return api_error(
        500,
        "output-recipe-encryption-unverified",
        "The encrypted file could not be checked after it was written, so it was not sent.",
    )


def _encrypted_download(content: bytes, file_name: str, digest: str) -> Response:
    name = quote(file_name, safe="")
    return Response(
        content,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename*=utf-8''{name}",
            "Cache-Control": "private, no-store",
            "Content-Security-Policy": "sandbox; default-src 'none'",
            "Cross-Origin-Resource-Policy": "same-origin",
            "X-Content-Type-Options": "nosniff",
            "X-Output-Recipe-Digest": digest,
        },
    )


@router.get("/artifacts/{artifact_id}/generation-settings")
async def read_generation_settings(artifact_id: str, request: Request) -> JSONResponse:
    """What a picture's own file says about how it was made, as plain claims.

    Only the text a PNG stores uncompressed, and the EXIF text of a JPEG or
    WebP, is read. Nothing is kept, fetched, run or trusted: models, LoRAs and
    graphs named in the text are listed as ignored, and another kind of file
    answers that it was not read. A whole
    number too large for a browser to read exactly, such as a large seed, is
    sent as decimal text.
    """

    artifacts = cast("Services", request.app.state.services).artifacts
    try:
        metadata = await run_in_threadpool(read_picture_generation_settings, artifacts, artifact_id)
    except OutputRecipeUnavailable as exc:
        raise api_error(exc.status, exc.code, exc.message) from exc
    # The prompt it may hold is the person's own and is never cached.
    return JSONResponse(
        _generation_settings_answer(metadata), headers={"Cache-Control": "no-store"}
    )


def read_picture_generation_settings(
    artifacts: ArtifactStore, artifact_id: str
) -> ExternalGenerationMetadata:
    """Read the settings a stored picture's own file carries, or refuse with a coded reason."""

    content = _stored_picture(artifacts, artifact_id)
    if isinstance(content, str):
        return metadata_not_read(content)
    try:
        return read_external_generation_metadata(content)
    except ValueError as exc:
        raise _settings_unreadable() from exc


@router.get("/artifacts/{artifact_id}/embedded-workflow")
async def read_embedded_workflow(artifact_id: str, request: Request) -> JSONResponse:
    """The ComfyUI workflow a picture's own file carries, for the person to review.

    This imports, keeps, runs and trusts nothing. The client hands the graph to
    the same review a workflow file gets, which imports only what the person
    then confirms. A picture whose file carries no workflow that review can read
    answers ``picture-workflow-missing``.
    """

    artifacts = cast("Services", request.app.state.services).artifacts
    try:
        graph = await run_in_threadpool(read_stored_picture_workflow, artifacts, artifact_id)
    except OutputRecipeUnavailable as exc:
        raise api_error(exc.status, exc.code, exc.message) from exc
    # Its nodes hold the person's own prompt, so like the settings it is never cached.
    return JSONResponse({"ui_graph": graph}, headers={"Cache-Control": "no-store"})


def read_stored_picture_workflow(artifacts: ArtifactStore, artifact_id: str) -> dict[str, Any]:
    """Read the workflow a stored picture's own file carries, or refuse with a coded reason."""

    content = _stored_picture(artifacts, artifact_id)
    try:
        graph = None if isinstance(content, str) else read_picture_workflow(content)
    except ValueError as exc:
        raise _settings_unreadable() from exc
    if graph is None:
        raise OutputRecipeUnavailable(
            404,
            "picture-workflow-missing",
            "This picture's file carries no workflow that can be reviewed.",
        )
    return graph


def _stored_picture(artifacts: ArtifactStore, artifact_id: str) -> bytes | str:
    """A stored picture's bytes, or the reason they are not read."""

    with SessionLocal() as session:
        artifact = session.get(Artifact, artifact_id)
        if artifact is None:
            raise OutputRecipeUnavailable(404, "artifact-not-found", "artifact not found")
        media_type = (artifact.media_type or "").lower()
        if media_type.startswith(("video/", "audio/")):
            # Only a picture's own text is read; nothing else is opened.
            return "format_not_read"
        if artifact.size_bytes > MAX_BLEND_READ_BYTES:
            return "too_large"
        try:
            return artifacts.verified_bytes(artifact, maximum_bytes=MAX_BLEND_READ_BYTES)
        except (ValueError, OSError) as exc:
            raise OutputRecipeUnavailable(
                410, "artifact-file-unreadable", "artifact file is missing or corrupt"
            ) from exc


def _settings_unreadable() -> OutputRecipeUnavailable:
    return OutputRecipeUnavailable(
        422,
        "generation-settings-unreadable",
        "The settings text stored in this picture could not be read.",
    )


def _generation_settings_answer(metadata: ExternalGenerationMetadata) -> dict[str, Any]:
    return {
        "dialect": metadata.dialect,
        "parser_version": metadata.parser_version,
        "budget_version": metadata.budget_version,
        "digest": metadata.digest,
        "claims": [
            {"key": claim.key, "value": exact_in_json(claim.value), "source": claim.source}
            for claim in metadata.claims
        ],
        "ignored": [{"name": item.name, "reason": item.reason} for item in metadata.ignored],
        "warnings": list(metadata.warnings),
    }


#: The largest whole number a JavaScript number holds exactly.
_EXACT_IN_A_DOUBLE = 2**53 - 1


def exact_in_json(value: str | int | float) -> str | int | float:
    """A whole number past what a browser reads exactly goes as decimal text, so a seed survives."""

    if isinstance(value, int) and abs(value) > _EXACT_IN_A_DOUBLE:
        return str(value)
    return value


def _build_bundle(
    artifacts: ArtifactStore,
    run_id: str,
    artifact_id: str,
    include_prompts: bool,
    digest: str,
    include_inputs: bool,
) -> OutputRecipeBundle:
    # Copying the picture decodes and re-encodes it, which is slow for a large one.
    with SessionLocal() as session:
        return build_output_recipe_bundle(
            session,
            artifacts,
            run_id=run_id,
            artifact_id=artifact_id,
            include_prompts=include_prompts,
            expected_record_digest=digest,
            include_inputs=include_inputs,
        )


@router.post("/output-recipes/check")
async def check_output_recipe_against_this_install(request: Request) -> JSONResponse:
    """Which of a record's requirements this installation holds, by exact identity.

    The record, or a bundle holding it and a copy of its picture, arrives as the
    file's own bytes, because only those can be checked against their digests.
    The body is read only up to the largest file the check accepts. Nothing is
    installed, trusted or kept.
    """

    content = await read_record_body(request)
    try:
        report = await run_in_threadpool(_check, content)
    except OutputRecipeCheckRefused as exc:
        raise api_error(exc.status, exc.code, exc.message) from exc
    return JSONResponse(report, headers={"Cache-Control": "no-store"})


# An encrypted file adds a tag of 16 bytes to every 16 KiB and a short header.
_MAX_ENCRYPTED_BYTES = MAX_CHECK_BYTES + MAX_CHECK_BYTES // 1024 + 64 * 1024


class _OpenedTooLarge(Exception):
    """An encrypted file that opens to more than a record or a bundle can be."""


class _BoundedSink:
    """A destination that holds what it is given, up to what a record file can be."""

    def __init__(self) -> None:
        self._held = io.BytesIO()

    def write(self, data: bytes) -> int:
        if self._held.tell() + len(data) > MAX_CHECK_BYTES:
            raise _OpenedTooLarge
        return self._held.write(data)

    def getvalue(self) -> bytes:
        return self._held.getvalue()


@router.post("/output-recipes/open")
async def open_encrypted_output_recipe(request: Request) -> Response:
    """An encrypted record or bundle opened with its passphrase, as the file it holds.

    The file is the request body, read only up to the largest encrypted record
    file there can be. The passphrase arrives in a request header as base64 of
    its UTF-8 bytes, never in the address. What comes back is the plain record
    or bundle, for this browser to check or remake as it would a plain file;
    nothing is kept.
    """

    passphrase = passphrase_from_header(request)
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > _MAX_ENCRYPTED_BYTES:
        raise _too_large()
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > _MAX_ENCRYPTED_BYTES:
            raise _too_large()
    try:
        opened = await run_in_threadpool(_open_encrypted, bytes(body), passphrase)
    except _OpenedTooLarge as exc:
        raise _too_large() from exc
    except ArchiveRefused as exc:
        if exc.code == "archive-kind-mismatch":
            raise api_error(
                422, "archive-kind-mismatch", "This file is not an encrypted generation record."
            ) from exc
        raise archive_api_error(exc) from exc
    if len(opened) > most_read_for(opened[: len(BUNDLE_SIGNATURE)]):
        # Held to a record's own bound unless it opens as a bundle, as a plain file is.
        raise _too_large()
    return Response(
        opened,
        media_type="application/octet-stream",
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


def _open_encrypted(content: bytes, passphrase: bytes) -> bytes:
    sink = _BoundedSink()
    open_archive(
        io.BytesIO(content),
        cast(BinaryIO, sink),
        kind=ArchiveKind.OUTPUT_RECIPE,
        passphrase=passphrase,
    )
    return sink.getvalue()


def _too_large() -> Exception:
    return api_error(
        413, "output-recipe-too-large", "This file is larger than a generation record can be."
    )


def _check(content: bytes) -> dict[str, Any]:
    with SessionLocal() as session:
        return check_output_recipe_file(session, content)


@router.post("/output-recipes/replay-plan")
async def plan_an_exact_replay(request: Request) -> JSONResponse:
    """Whether a record could be generated again exactly here, and from what.

    Takes the same file the check takes. Each requirement is resolved to exactly
    one local workflow revision, model profile, LoRA asset or input, or refused;
    every refusal is listed. Nothing is started, installed, trusted or kept.
    """

    content = await read_record_body(request)
    media_engine = cast("Services", request.app.state.services).settings.media_engine
    try:
        plan = await run_in_threadpool(_plan, content, media_engine)
    except OutputRecipeCheckRefused as exc:
        raise api_error(exc.status, exc.code, exc.message) from exc
    return JSONResponse(plan, headers={"Cache-Control": "no-store"})


@router.get("/runs/{run_id}/recipe-draft", response_model=OutputRecipeDraftOut)
async def draft_output_recipe(
    run_id: str, request: Request, session: Annotated[Session, Depends(get_session)]
) -> OutputRecipeDraftOut:
    """A recipe to review from one generation: the settings it ran with that a recipe can hold.

    Nothing is saved. The answer names the generation's model and workflow
    beside the settings, and says why any setting it ran with was left out.
    """

    services = cast("Services", request.app.state.services)
    try:
        return await output_recipe_draft(services.orchestrator, session, run_id)
    except OutputRecipeDraftRefused as refused:
        raise api_error(refused.status, refused.code, refused.message) from None


@router.get("/runs/{run_id}/edit-recipe-draft", response_model=EditRecipeDraftOut)
async def draft_edit_recipe(
    run_id: str, session: Annotated[Session, Depends(get_session)]
) -> EditRecipeDraftOut:
    """The words an edit was asked with, to keep it as an Image Studio recipe; nothing is saved."""

    try:
        return edit_recipe_draft(session, run_id)
    except OutputRecipeDraftRefused as refused:
        raise api_error(refused.status, refused.code, refused.message) from None


@router.get("/runs/{run_id}/replay-result")
async def replay_result(run_id: str) -> JSONResponse:
    """Whether a run generated again from a record came out as the record's output did.

    A run made as a new version of a record answers which parts of it differ.
    """

    outcome = await run_in_threadpool(_replay_result, run_id)
    if outcome is None:
        raise api_error(
            404, "replay-result-not-found", "This generation was not generated again from a record."
        )
    return JSONResponse(outcome, headers={"Cache-Control": "no-store"})


def _replay_result(run_id: str) -> dict[str, Any] | None:
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        return replay_outcome(run) if run is not None else None


def _plan(content: bytes, media_engine: str) -> dict[str, Any]:
    record = read_record_file(content).record
    with SessionLocal() as session:
        return plan_output_recipe_replay(session, record, media_engine=media_engine)


async def read_record_body(request: Request) -> bytes:
    """A record file sent as the request body, refused once longer than a record file can be."""

    content = await _bounded_body(request)
    if content is None:
        raise api_error(
            413, "output-recipe-too-large", "This file is larger than a generation record can be."
        )
    return content


async def _bounded_body(request: Request) -> bytes | None:
    """The request body, or None once it is longer than the check reads, without reading on."""

    declared = request.headers.get("content-length")
    if (
        declared is not None
        and declared.isdigit()
        and int(declared) > most_read_for(BUNDLE_SIGNATURE)
    ):
        return None
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > most_read_for(bytes(body[: len(BUNDLE_SIGNATURE)])):
            return None
    return bytes(body)
