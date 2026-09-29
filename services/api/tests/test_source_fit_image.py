"""Source geometry and upload content come from the same verified bytes."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import Base
from local_lm.domain import ArtifactKind
from local_lm.models import Artifact


@pytest.fixture
def artifact_session(tmp_path: Path) -> Iterator[tuple[ArtifactStore, Session]]:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    engine = create_engine(f"sqlite:///{tmp_path / 'artifacts.sqlite3'}")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as session:
        yield ArtifactStore(settings), session
    engine.dispose()


def png(*, mode: str = "RGB", orientation: int | None = None) -> bytes:
    image = Image.new(mode, (3, 2))
    if mode == "RGB":
        image.putdata(
            [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        )
    output = BytesIO()
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    image.save(output, format="PNG", exif=exif)
    return output.getvalue()


def ingest(pair: tuple[ArtifactStore, Session], content: bytes) -> Artifact:
    store, session = pair
    value = store.ingest_bytes(
        session,
        content,
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
        metadata={"width": 999, "height": 999},
    )
    session.commit()
    return value


@pytest.mark.parametrize(
    "orientation,dimensions,pixels",
    [
        (
            None,
            (3, 2),
            [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)],
        ),
        (
            6,
            (2, 3),
            [(255, 255, 0), (255, 0, 0), (0, 255, 255), (0, 255, 0), (255, 0, 255), (0, 0, 255)],
        ),
    ],
)
def test_verified_source_records_decoded_oriented_pixels(
    artifact_session: tuple[ArtifactStore, Session],
    orientation: int | None,
    dimensions: tuple[int, int],
    pixels: list[tuple[int, int, int]],
) -> None:
    from local_lm.source_fit_image import prepare_source_fit_image

    content = png(orientation=orientation)
    artifact = ingest(artifact_session, content)
    prepared = prepare_source_fit_image(artifact_session[0], artifact)
    assert (prepared.width, prepared.height) == dimensions
    assert prepared.source_artifact_id == artifact.id
    assert prepared.source_sha256 == hashlib.sha256(content).hexdigest()
    assert prepared.sha256 == hashlib.sha256(prepared.content).hexdigest()
    with Image.open(BytesIO(prepared.content)) as image:
        image.load()
        assert image.mode == "RGB"
        assert image.size == dimensions
        assert image.tobytes() == bytes(channel for pixel in pixels for channel in pixel)
        assert not image.getexif()
        assert not image.info


def test_no_second_path_read_after_verification(
    artifact_session: tuple[ArtifactStore, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_lm.source_fit_image import prepare_source_fit_image

    store, _ = artifact_session
    artifact = ingest(artifact_session, png())
    path = store.resolve(artifact)
    verified = store.verified_bytes
    calls = 0

    def read_once(value: Artifact, *, maximum_bytes: int) -> bytes:
        nonlocal calls
        calls += 1
        content = verified(value, maximum_bytes=maximum_bytes)
        path.write_bytes(b"replaced after the verified read")
        return content

    monkeypatch.setattr(store, "verified_bytes", read_once)
    prepared = prepare_source_fit_image(store, artifact)
    assert calls == 1
    with Image.open(BytesIO(prepared.content)) as image:
        assert image.size == (3, 2)
        assert image.getpixel((0, 0)) == (255, 0, 0)


def test_changed_artifact_bytes_refuse_even_after_verified_path_cache(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    from local_lm.source_fit_image import prepare_source_fit_image

    store, _ = artifact_session
    artifact = ingest(artifact_session, png())
    path = store.verified_path(artifact)
    path.write_bytes(b"x" * artifact.size_bytes)
    with pytest.raises(ValueError, match="source_fit_image_unavailable"):
        prepare_source_fit_image(store, artifact)


@pytest.mark.parametrize(
    "kind",
    [
        "truncated",
        "trailing",
        "animated",
        "transparent",
        "sixteen_bit",
        "icc",
        "gamma",
        "jpeg",
    ],
)
def test_unsupported_or_incomplete_source_is_not_silently_reinterpreted(
    artifact_session: tuple[ArtifactStore, Session],
    kind: str,
) -> None:
    from local_lm.source_fit_image import prepare_source_fit_image

    output = BytesIO()
    if kind == "truncated":
        content = png()[:-8]
    elif kind == "trailing":
        content = png() + b"extra"
    else:
        image = Image.new("RGB", (3, 2), (12, 34, 56))
        if kind == "animated":
            image.save(
                output,
                format="PNG",
                save_all=True,
                append_images=[Image.new("RGB", (3, 2), (78, 90, 12))],
            )
        elif kind == "transparent":
            Image.new("RGBA", (3, 2), (12, 34, 56, 0)).save(output, format="PNG")
        elif kind == "sixteen_bit":
            Image.new("I;16", (3, 2), 1000).save(output, format="PNG")
        elif kind == "icc":
            image.save(output, format="PNG", icc_profile=b"unsupported fixture profile")
        elif kind == "gamma":
            info = PngImagePlugin.PngInfo()
            info.add(b"gAMA", (100000).to_bytes(4, "big"))
            image.save(output, format="PNG", pnginfo=info)
        else:
            image.save(output, format="JPEG")
        content = output.getvalue()
    artifact = ingest(artifact_session, content)
    with pytest.raises(ValueError, match="source_fit_image_unsupported"):
        prepare_source_fit_image(artifact_session[0], artifact)


def test_opaque_alpha_is_preserved_as_rgb(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    from local_lm.source_fit_image import prepare_source_fit_image

    output = BytesIO()
    Image.new("RGBA", (3, 2), (12, 34, 56, 255)).save(output, format="PNG")
    artifact = ingest(artifact_session, output.getvalue())
    prepared = prepare_source_fit_image(artifact_session[0], artifact)
    with Image.open(BytesIO(prepared.content)) as image:
        assert image.mode == "RGB"
        assert image.getpixel((1, 1)) == (12, 34, 56)


def test_pixel_budget_refuses_before_pillow_opens_source(
    artifact_session: tuple[ArtifactStore, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import local_lm.source_fit_image as module

    artifact = ingest(artifact_session, png())
    monkeypatch.setattr(module, "MAX_SOURCE_PIXELS", 5)

    def unexpected_open(*args: object, **kwargs: object) -> None:
        pytest.fail("over-budget source reached Pillow")

    monkeypatch.setattr(Image, "open", unexpected_open)
    with pytest.raises(ValueError, match="source_fit_image_unsupported"):
        module.prepare_source_fit_image(artifact_session[0], artifact)


def test_byte_budget_refuses_at_verified_artifact_read(
    artifact_session: tuple[ArtifactStore, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import local_lm.source_fit_image as module

    artifact = ingest(artifact_session, png())
    monkeypatch.setattr(module, "MAX_SOURCE_BYTES", artifact.size_bytes - 1)
    with pytest.raises(ValueError, match="source_fit_image_unavailable"):
        module.prepare_source_fit_image(artifact_session[0], artifact)


async def test_prepared_upload_sends_exact_oriented_bytes_into_actual_compiler(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    from email import policy
    from email.parser import BytesParser

    import httpx

    from local_lm.adapters.base import MediaRequest
    from local_lm.adapters.comfyui import ComfyUIAdapter
    from local_lm.source_fit_image import prepare_source_fit_image

    store, _ = artifact_session
    artifact = ingest(artifact_session, png(orientation=6))
    prepared = prepare_source_fit_image(store, artifact)
    path = store.resolve(artifact)
    path.write_bytes(b"the source pathname no longer contains an image")
    sent: list[bytes] = []

    async def upload(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/upload/image"
        body = await request.aread()
        message = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: "
            + request.headers["content-type"].encode("ascii")
            + b"\r\nMIME-Version: 1.0\r\n\r\n"
            + body
        )
        parts = [part for part in message.walk() if part.get_filename()]
        assert len(parts) == 1
        content = parts[0].get_payload(decode=True)
        assert isinstance(content, bytes)
        sent.append(content)
        return httpx.Response(
            200,
            json={
                "name": "source.png",
                "subfolder": "lm-atelier",
                "type": "temp",
            },
        )

    request = MediaRequest(
        run_id="source-fit",
        operation="image_to_image",
        prompt="Neutral shapes",
        negative_prompt=None,
        input_paths=[path],
        input_contents=(prepared.content,),
        workflow={"source": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}}},
        parameters={},
    )
    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test",
        transport=httpx.MockTransport(upload),
    )
    try:
        parameters = await adapter._request_parameters(request)
        compiled = adapter._compile(request.workflow, parameters)
    finally:
        await adapter.close()
    assert sent == [prepared.content]
    assert hashlib.sha256(sent[0]).hexdigest() == prepared.sha256
    assert compiled["source"]["inputs"]["image"] == "lm-atelier/source.png [temp]"
    with Image.open(BytesIO(sent[0])) as image:
        assert image.size == (2, 3)
        assert image.getpixel((0, 0)) == (255, 255, 0)


@pytest.mark.parametrize("contents", [(), (b"first", b"second")])
async def test_prepared_upload_count_mismatch_refuses_before_any_input_transfer(
    tmp_path: Path,
    contents: tuple[bytes, ...],
) -> None:
    import httpx

    from local_lm.adapters.base import MediaRequest
    from local_lm.adapters.comfyui import ComfyUIAdapter

    request = MediaRequest(
        run_id="source-fit",
        operation="image_to_image",
        prompt="Neutral shapes",
        negative_prompt=None,
        input_paths=[tmp_path / "must-not-be-opened"],
        input_contents=contents,
        workflow={},
        parameters={},
    )

    async def unexpected_upload(request: httpx.Request) -> httpx.Response:
        pytest.fail("inconsistent input bytes reached the transport")

    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test",
        transport=httpx.MockTransport(unexpected_upload),
    )
    try:
        with pytest.raises(ValueError, match="input bytes"):
            await adapter._request_parameters(request)
    finally:
        await adapter.close()


def test_captured_source_replays_exact_bytes_without_redecoding_the_original(
    artifact_session: tuple[ArtifactStore, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_lm.source_fit_image import (
        SourceFitImageRecord,
        capture_source_fit_image,
        replay_source_fit_image,
    )

    store, session = artifact_session
    source = ingest(artifact_session, png(orientation=6))
    record = capture_source_fit_image(session, store, source)
    session.commit()
    serialized = record.model_dump(mode="json")
    restored = SourceFitImageRecord.model_validate(serialized)
    prepared_artifact = session.get(Artifact, record.prepared_artifact_id)
    assert prepared_artifact is not None
    assert prepared_artifact.kind == ArtifactKind.INPUT
    assert prepared_artifact.id != source.id
    before = store.verified_bytes(prepared_artifact, maximum_bytes=1024 * 1024)
    store.resolve(source).write_bytes(b"original changed after acceptance")

    def no_decoder(*args: object, **kwargs: object) -> None:
        pytest.fail("replay decoded the original source again")

    monkeypatch.setattr(Image, "open", no_decoder)
    replayed = replay_source_fit_image(
        session,
        store,
        restored,
        selected_source_id=source.id,
    )
    assert replayed.content == before
    assert (replayed.width, replayed.height) == (2, 3)
    assert replayed.source_artifact_id == source.id
    assert replayed.sha256 == hashlib.sha256(before).hexdigest()


def test_captured_source_refuses_a_different_selected_input(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    from local_lm.source_fit_image import capture_source_fit_image, replay_source_fit_image

    store, session = artifact_session
    source = ingest(artifact_session, png(orientation=6))
    other = ingest(artifact_session, png())
    record = capture_source_fit_image(session, store, source)
    with pytest.raises(ValueError, match="source_fit_image_binding"):
        replay_source_fit_image(session, store, record, selected_source_id=other.id)


def test_captured_source_refuses_changed_prepared_bytes(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    from local_lm.source_fit_image import capture_source_fit_image, replay_source_fit_image

    store, session = artifact_session
    source = ingest(artifact_session, png(orientation=6))
    record = capture_source_fit_image(session, store, source)
    prepared = session.get(Artifact, record.prepared_artifact_id)
    assert prepared is not None
    store.resolve(prepared).write_bytes(b"x" * prepared.size_bytes)
    with pytest.raises(ValueError, match="source_fit_image_unavailable"):
        replay_source_fit_image(session, store, record, selected_source_id=source.id)


def test_captured_source_refuses_false_record_dimensions(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    from local_lm.source_fit_image import (
        SourceFitImageRecord,
        capture_source_fit_image,
        replay_source_fit_image,
    )

    store, session = artifact_session
    source = ingest(artifact_session, png(orientation=6))
    record = capture_source_fit_image(session, store, source)
    payload = record.model_dump(mode="json")
    payload["width"] = 7
    changed = SourceFitImageRecord.model_validate(payload)
    with pytest.raises(ValueError, match="source_fit_image_binding"):
        replay_source_fit_image(session, store, changed, selected_source_id=source.id)


def test_captured_source_deduplication_keeps_existing_kind_and_metadata(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    from local_lm.source_fit_image import capture_source_fit_image, prepare_source_fit_image

    store, session = artifact_session
    source = ingest(artifact_session, png())
    assert prepare_source_fit_image(store, source).content == png()
    source.metadata_json = {"temporary_preview": True, "fixture_marker": "keep"}
    source.original_name = "existing-neutral-image.png"
    source.favorite = True
    session.commit()
    record = capture_source_fit_image(session, store, source)
    session.flush()
    session.refresh(source)
    assert record.prepared_artifact_id == source.id
    assert source.kind == ArtifactKind.IMAGE
    assert source.metadata_json == {"temporary_preview": True, "fixture_marker": "keep"}
    assert source.original_name == "existing-neutral-image.png"
    assert source.favorite is True
