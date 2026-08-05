"""What a dependency declared as a source URL actually points at.

A custom-node package can declare a dependency as a URL rather than a name
and version. Every one of those is refused: installing from a URL means
running a build backend nobody reviewed, over code that can change under the
same address tomorrow.

One blanket refusal was enough while the answer was always no. It is not
enough now, because the live cases differ in what would fix them:

- `git+https://github.com/owner/repo@<40-hex>` names an exact immutable
  object. It cannot be installed yet - content-binding a resolved source to a
  reviewed artifact is the next slice - but it is resolvable in principle,
  and saying so is different from saying never.
- `git+https://github.com/owner/repo` and the same with a branch or tag name
  no immutable object at all. Impact Pack declares one of these and WAS 3.0.1
  declares three. No amount of later machinery makes a branch exact; the
  package has to pin it, or the dependency has to be proven unnecessary by
  activating without it and verifying the workflow's required node inventory.
- Anything else - a local path, another host, another version control system
  - is outside what is allowed at all.

This module only classifies. It installs nothing, fetches nothing, and every
path through it still ends in a refusal; what it changes is that the refusal
says which of those three situations the package is in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

#: Git object names are exactly this, and anything shorter is a prefix that
#: can become ambiguous as a repository grows.
_COMMIT = re.compile(r"\A[0-9a-f]{40}\Z")
_ALLOWED_HOSTS = frozenset({"github.com", "www.github.com"})
_GIT_SCHEMES = frozenset({"git+https", "git+ssh", "git"})
MAX_SOURCE_URL_CHARACTERS = 700


@dataclass(frozen=True)
class SourceDependency:
    """A dependency declared as a URL, and what can be said about it."""

    url: str
    #: `owner/repo` when the URL names one on an allowed host, else None.
    repository: str | None
    #: The exact object this is pinned to, when it is pinned to one.
    commit: str | None
    #: The branch or tag it names instead, when it names one.
    reference: str | None

    @property
    def pinned(self) -> bool:
        return self.commit is not None


def classify_source_url(url: str) -> SourceDependency:
    """Read a dependency URL without fetching anything it points at."""
    if not url or len(url) > MAX_SOURCE_URL_CHARACTERS:
        return SourceDependency(url=url, repository=None, commit=None, reference=None)
    split = urlsplit(url)
    if split.scheme not in _GIT_SCHEMES or split.hostname not in _ALLOWED_HOSTS:
        return SourceDependency(url=url, repository=None, commit=None, reference=None)
    # pip spells the revision after `@`, which is also the separator inside a
    # userinfo section, so the path is what carries it and the netloc is not
    # consulted for it.
    path, _, revision = split.path.partition("@")
    parts = [part for part in path.strip("/").split("/") if part]
    if len(parts) != 2:
        return SourceDependency(url=url, repository=None, commit=None, reference=None)
    owner, repo = parts[0], parts[1].removesuffix(".git")
    if not owner or not repo:
        return SourceDependency(url=url, repository=None, commit=None, reference=None)
    repository = f"{owner}/{repo}"
    if _COMMIT.fullmatch(revision.casefold()):
        return SourceDependency(
            url=url, repository=repository, commit=revision.casefold(), reference=None
        )
    return SourceDependency(url=url, repository=repository, commit=None, reference=revision or None)


def source_refusal(source: SourceDependency) -> tuple[str, str]:
    """The code and sentence for refusing one source dependency.

    Every source is refused. The distinction is what the package would have
    to do about it, which is the part that was missing.
    """
    if source.repository is None:
        return (
            "direct_dependency_url",
            "Registry pip dependencies cannot use direct, local, or VCS URLs",
        )
    if not source.pinned:
        named = f" at {source.reference}" if source.reference else ""
        return (
            "unpinned_source_dependency",
            f"This package depends on {source.repository}{named}, which names no exact "
            "commit. A branch can change after review, so it cannot be installed as "
            "declared; the package must pin the dependency to a commit.",
        )
    commit = source.commit or ""
    return (
        "unresolved_source_dependency",
        f"This package depends on {source.repository} pinned to {commit[:12]}. "
        "Installing it would run a build backend that has not been reviewed, so it is "
        "refused until that source can be resolved to a reviewed artifact.",
    )
