"""Whether someone else is using the ComfyUI worker, and how a run waits for them.

A media run that needs ComfyUI restarted with a different set of models and
custom nodes ends whatever the running worker is doing. When that is work LM
Atelier did not start, such as a render queued from ComfyUI's own page or an
open workflow editor, the run waits for it, queued and holding nothing, and
gives up after a set time rather than ending it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: What is using the worker: prompts LM Atelier did not submit, or a workflow
#: editor open on its launch.
MediaUse = Literal["prompts", "editor"]

#: The stage a waiting run shows. The conversation title-cases a stage, so these
#: name no product, whose own capitals would not survive it.
WAIT_LABELS: dict[MediaUse, str] = {
    "prompts": "Waiting for other media work to finish",
    "editor": "Waiting for the workflow editor to close",
}

#: How long ComfyUI must stay free after a waiting run found someone else's
#: prompts in it before a run restarts it. Work queued from ComfyUI's own page
#: arrives a prompt at a time, so a moment with none can fall between two of
#: them. The spell runs from when checks began finding ComfyUI free, the same
#: for every run. It holds back a run whose own wait saw those prompts, and any
#: run that would restart ComfyUI while a run that saw them could start now. A
#: run that is held, paused or waiting for earlier steps is not asked, so what it
#: saw may be long over, and it holds back no other run. An editor that closed
#: or saved does not come back by itself and starts no spell: a run that waited
#: only on one starts once it closes, unless a run that could start now saw
#: prompts and the spell after them is still running.
MEDIA_IDLE_SETTLE_SECONDS = 10.0

#: The most one check adds to a wait's clock: enough for a check that comes
#: late, and all that counts of a stretch in which the run was not asked. A
#: waiting run is asked again every few seconds, even while another job has the
#: device, but not while it is held, paused or waiting for earlier steps.
MEDIA_WAIT_STEP_SECONDS = 10.0

_RESTART_NEEDED = (
    "This request needs ComfyUI restarted with a different set of models and custom nodes."
)


class MediaWorkerBusy(RuntimeError):
    """A run was stopped rather than restart ComfyUI under someone else's work.

    `late` is a refusal at the moment of the restart, which sees only that the
    work is there then: it may have begun while the run was being prepared, or
    have been there all along when the run's check could not tell or did not
    foresee the restart. Otherwise the run waited out its limit. Neither is
    retried automatically: a retry would meet the same work.
    """

    code = "media-worker-busy"

    def __init__(self, use: MediaUse, *, late: bool, limit_seconds: float = 0.0) -> None:
        self.use = use
        self.late = late
        super().__init__(
            _late_message(use) if late else _limit_message(use, limit_seconds),
        )


@dataclass(slots=True)
class MediaReplacementWait:
    """One queued run's wait for someone else's use of ComfyUI to end.

    `counted` is the time between the checks that found ComfyUI in use, each
    gap counted up to `MEDIA_WAIT_STEP_SECONDS`, and `last_busy` when a check
    last found it in use. The run is asked again unless it is held, paused or
    waiting for earlier steps, so time another job spends on the device counts.
    `prompts_seen` is whether a check of this run found someone else's prompts
    rather than only an editor. That holds this run to the quiet spell after
    them, and other runs only while this one could start now: a run that is not
    asked may have seen them long ago. When checks began finding ComfyUI free
    is one time for every waiting run, so it is kept once for all of them rather
    than here. `announced` is the stage the run's conversation shows, if any: a
    stage that could not be written is not recorded, so the next check writes
    it again.
    """

    counted: float = 0.0
    last_busy: float | None = None
    prompts_seen: bool = False
    announced: str | None = None


def _waited(limit_seconds: float) -> str:
    """How long the run waited, as the end of a sentence about the work it met."""

    if limit_seconds <= 0:
        return "and LM Atelier is set not to wait for it"
    if limit_seconds % 60 == 0:
        minutes = int(limit_seconds // 60)
        return f"after {minutes} minute{'' if minutes == 1 else 's'}"
    return f"after {limit_seconds:g} second{'' if limit_seconds == 1 else 's'}"


def _limit_message(use: MediaUse, limit_seconds: float) -> str:
    waited = _waited(limit_seconds)
    if use == "editor":
        return (
            f"{_RESTART_NEEDED} The ComfyUI workflow editor was still open {waited}, so it "
            "was left open and this request was stopped. Save or close the editor, then try "
            "again."
        )
    return (
        f"{_RESTART_NEEDED} ComfyUI was still running work that LM Atelier did not start "
        f"{waited}, so that work was left running and this request was stopped. Try again "
        "once it has finished."
    )


def _late_message(use: MediaUse) -> str:
    if use == "editor":
        return (
            "A ComfyUI workflow editor was open, so ComfyUI was not restarted under it. Save "
            "or close the editor, then try again."
        )
    return (
        "ComfyUI was running work that LM Atelier did not start, so it was not restarted for "
        "this request and that work was left running. Try again once it has finished."
    )
