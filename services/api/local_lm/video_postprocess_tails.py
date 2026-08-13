"""Data-only recognition of admitted video post-processing graph tails.

No node-name substring inference. Unknown, interleaved, branching, or
output-affecting shapes remain unmatched. Catalogue starts empty until exact
public-reviewed tails are admitted by product decision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Exact admitted tails only. Empty until a reviewed catalogue entry lands.
ADMITTED_VIDEO_POSTPROCESS_TAILS: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class VideoPostprocessTailMatch:
    """One exact admitted tail at the end of a linear post-frame chain."""

    stage: str
    class_types: tuple[str, ...]
    producer_class_types: tuple[str, ...]


def analyze_video_postprocess_tail(
    graph: Mapping[str, Any] | None,
) -> VideoPostprocessTailMatch | None:
    """Return a match only for an exact admitted linear tail; otherwise None."""

    if not ADMITTED_VIDEO_POSTPROCESS_TAILS:
        return None
    if not isinstance(graph, Mapping):
        return None
    # Catalogue non-empty path reserved for future admission work.
    _ = graph
    return None
