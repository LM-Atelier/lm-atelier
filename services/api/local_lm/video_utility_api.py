"""Read what a stored video is before a video utility touches it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from fastapi import APIRouter, Request

from .api_errors import api_error
from .artifacts import ArtifactCopyUnavailable
from .db import SessionLocal
from .media_tools import MediaToolUnavailable, ToolUnavailableCode, find_media_tool
from .models import Artifact
from .video_probe import VideoProbe, VideoProbeRefused, probe_video, require_probe_input

if TYPE_CHECKING:
    from .main import Services

router = APIRouter()

_TOOL_MESSAGES: Final[dict[ToolUnavailableCode, str]] = {
    "media-tool-missing": "FFprobe is not installed on this computer.",
    "media-tool-unreadable": "FFprobe on this computer did not answer as FFprobe.",
}
_UNAVAILABLE: Final = (
    "The video could not be read right now. Check that the drive holding the app's data has room."
)


@router.get("/artifacts/{artifact_id}/video-probe")
async def read_video_probe(artifact_id: str, request: Request) -> VideoProbe:
    """What a stored video is and what the video utilities may do with it, or why not."""

    store = cast("Services", request.app.state.services).artifacts
    # The session closes before ffprobe runs, so no read is held open across it.
    with SessionLocal() as session:
        artifact = session.get(Artifact, artifact_id)
    if artifact is None:
        raise api_error(404, "artifact-not-found", "artifact not found")
    try:
        # A file the probe would never read is refused before ffprobe is looked for.
        require_probe_input(artifact)
        ffprobe = await find_media_tool("ffprobe")
        return await probe_video(store, artifact, ffprobe)
    except VideoProbeRefused as exc:
        raise api_error(422, exc.code, str(exc)) from exc
    except MediaToolUnavailable as exc:
        raise api_error(503, exc.code, _TOOL_MESSAGES[exc.code]) from exc
    except ArtifactCopyUnavailable as exc:
        # The stored file is intact; its private copy could not be made.
        raise api_error(503, "video-probe-unavailable", _UNAVAILABLE) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise api_error(
            410, "artifact-file-unreadable", "artifact file is missing or corrupt"
        ) from exc
