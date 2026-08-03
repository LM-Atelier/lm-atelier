"""Explicit source links a package author wrote down, validated server-side.

Workflow authors often record where a model came from - in a note node, a
property, anywhere text lives. Those links are the difference between "search
this filename and hope" and "this exact version", because a filename search
cannot find a file inside a repository, and a specialized checkpoint may not
be findable by name at all.

The links are still untrusted input. Nothing here fetches anything: it parses
text, keeps only hosts belonging to a registered catalog source, extracts the
provider's own identifiers, and hands back candidates the normal preflight
path must still resolve into an immutable plan. A browser-supplied URL is
never accepted, and a candidate is a suggestion of what to preflight - never
a download instruction.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse

MAX_CANDIDATES_PER_ASSET = 8
MAX_SCANNED_TEXT = 64_000

_URL = re.compile(r"https?://[^\s\"'<>)\]]+")
_CIVITAI_MODEL = re.compile(r"^/models/(?P<model>[1-9][0-9]{0,11})(?:/|$)")
_HUGGINGFACE_REPO = re.compile(
    r"^/(?P<owner>[A-Za-z0-9][\w.-]{0,95})/(?P<name>[A-Za-z0-9][\w.-]{0,95})"
    r"(?:/(?:resolve|blob)/(?P<revision>[^/]{1,200})/(?P<path>.+))?$"
)


@dataclass(frozen=True)
class SourceCandidate:
    """One explicit source the package author recorded, already validated."""

    provider: str
    remote_id: str
    revision: str | None
    filename: str | None
    url: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "remote_id": self.remote_id,
            "revision": self.revision,
            "filename": self.filename,
            "url": self.url,
        }


def collect_source_candidates(
    workflow: Mapping[str, Any],
    *,
    allowed_hosts: Mapping[str, str],
    asset_filenames: Sequence[str] = (),
) -> dict[str, tuple[SourceCandidate, ...]]:
    """Map each asset filename to the explicit sources written near it.

    `allowed_hosts` maps a hostname to the registered source id that owns it,
    so which hosts are acceptable is a property of what this installation has
    registered - never a constant compiled into the parser.

    A candidate is attached to a filename when the surrounding text names that
    file; anything else is offered under the empty key as a general
    suggestion, because guessing which unnamed link answers which file is
    exactly the substitution this whole path refuses to make.

    In practice most authors write display names ("Lenovo UltraReal") rather
    than filenames, so the general bucket is the common case - and that is
    the point. A list of sources the author actually recorded, for the user
    to assign, beats a filename search that returns nothing, and beats a
    slug-similarity guess that would sometimes be confidently wrong.
    """
    by_filename: dict[str, list[SourceCandidate]] = {}
    for text in _text_fragments(workflow):
        for block in _blocks(text):
            for url in _URL.findall(block):
                candidate = parse_source_url(url, allowed_hosts=allowed_hosts)
                if candidate is None:
                    continue
                key = _matching_filename(block, asset_filenames) or ""
                bucket = by_filename.setdefault(key, [])
                if any(existing.url == candidate.url for existing in bucket):
                    continue
                if len(bucket) < MAX_CANDIDATES_PER_ASSET:
                    bucket.append(candidate)
    return {key: tuple(value) for key, value in by_filename.items()}


def parse_source_url(
    url: str,
    *,
    allowed_hosts: Mapping[str, str],
) -> SourceCandidate | None:
    """Validate one URL into provider identifiers, or refuse it entirely."""
    if len(url) > 2_000:
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme != "https" or parsed.username or parsed.password:
        return None
    host = (parsed.hostname or "").lower()
    provider = allowed_hosts.get(host)
    if not provider:
        return None
    path = parsed.path.rstrip("/") or "/"
    if provider == "civitai":
        match = _CIVITAI_MODEL.match(path)
        if not match:
            return None
        # A CivitAI version is the installable identity; a model link without
        # one names a page, not something that can be installed.
        version = parse_qs(parsed.query).get("modelVersionId", [""])[0]
        if not re.fullmatch(r"[1-9][0-9]{0,11}", version):
            return None
        return SourceCandidate(
            provider=provider,
            remote_id=version,
            revision=version,
            filename=None,
            url=_canonical(parsed.scheme, host, path, parsed.query),
        )
    if provider == "huggingface":
        match = _HUGGINGFACE_REPO.match(path)
        if not match:
            return None
        filename = match.group("path")
        if filename and (".." in filename or filename.startswith("/")):
            return None
        return SourceCandidate(
            provider=provider,
            remote_id=f"{match.group('owner')}/{match.group('name')}",
            revision=match.group("revision"),
            filename=filename,
            url=_canonical(parsed.scheme, host, path, ""),
        )
    return None


def catalog_host_map(sources: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Host -> source id for every registered catalog source.

    Built from what the application actually registered, so a deployment
    serving a different host resolves its own links without this module
    knowing anything about it.
    """
    hosts: dict[str, str] = {}
    for source_id, base_url in sources:
        host = (urlparse(base_url).hostname or "").lower()
        if host:
            hosts[host] = source_id
    return hosts


def _canonical(scheme: str, host: str, path: str, query: str) -> str:
    return f"{scheme}://{host}{path}" + (f"?{query}" if query else "")


def _text_fragments(workflow: Mapping[str, Any]) -> list[str]:
    """Every string the graph carries, bounded so a huge file cannot stall."""
    fragments: list[str] = []
    budget = MAX_SCANNED_TEXT
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return fragments
    for node in nodes:
        if budget <= 0:
            break
        if not isinstance(node, Mapping):
            continue
        for value in _node_strings(node):
            if budget <= 0:
                break
            trimmed = value[:budget]
            budget -= len(trimmed)
            fragments.append(trimmed)
    return fragments


def _node_strings(node: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    widgets = node.get("widgets_values")
    if isinstance(widgets, list):
        values.extend(item for item in widgets if isinstance(item, str))
    elif isinstance(widgets, Mapping):
        values.extend(item for item in widgets.values() if isinstance(item, str))
    properties = node.get("properties")
    if isinstance(properties, Mapping):
        values.extend(item for item in properties.values() if isinstance(item, str))
    title = node.get("title")
    if isinstance(title, str):
        values.append(title)
    return values


def _blocks(text: str) -> list[str]:
    """Split note text where its author separated one subject from the next.

    A single note routinely lists several models, each with its own link. If a
    filename mentioned anywhere in that note could claim every link in it, one
    named file would swallow sources belonging to entirely different models -
    a wrong binding, which is worse than no binding. A blank line is how
    authors mark "new subject", so it is the boundary honored here.
    """
    return [block for block in re.split(r"\n\s*\n", text) if block.strip()]


def _matching_filename(text: str, filenames: Sequence[str]) -> str | None:
    """The asset this text is about, if it names one unambiguously."""
    lowered = text.casefold()
    matches = [name for name in filenames if name and name.casefold() in lowered]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        # A stem match catches "Detailer-KREA2" written without .safetensors.
        stems = [
            name for name in filenames if name and name.rsplit(".", 1)[0].casefold() in lowered
        ]
        if len(stems) == 1:
            return stems[0]
    return None
