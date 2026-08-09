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
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

from packaging.requirements import InvalidRequirement, Requirement

#: Git object names are exactly this, and anything shorter is a prefix that
#: can become ambiguous as a repository grows.
_COMMIT = re.compile(r"\A[0-9a-f]{40}\Z")
_ALLOWED_HOSTS = frozenset({"github.com", "www.github.com"})
_GIT_SCHEMES = frozenset({"git+https", "git+ssh", "git"})
MAX_SOURCE_URL_CHARACTERS = 700
#: A requirements file carries furniture that names no dependency. Matching
#: what the planner already strips, so partition and refusal read one line the
#: same way rather than through two parsers that can drift apart.
_INLINE_COMMENT = re.compile(r"\s+#")


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


def bare_source_url(line: str) -> str | None:
    """A dependency written as a URL alone, with no distribution name.

    pip takes this spelling and PEP 508 does not, so the packaging parser
    refuses it and every reader that asks the parser what a line points at
    hears "nothing". Two of them care: the one that explains a refusal, and
    the one that decides whether an unpinned source may be set aside.

    Deliberately narrow: only a line that is entirely one scheme-bearing URL
    counts. Anything with whitespace is a malformed requirement rather than a
    source, and reading that as a source would set aside a line nobody parsed.
    """
    if len(line.split()) != 1:
        return None
    scheme = line.partition("://")[0]
    return line if scheme and scheme != line else None


def partition_unpinned_sources(
    declarations: Sequence[str], *, authorized: bool
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split declarations into what can be planned and what may be omitted.

    Only an unpinned dependency on an allowed host is ever set aside, and only
    under an authorized workflow: without one there is nothing an omission
    could later be proven against, so the ordinary refusal stands and the
    planner sees the declaration unchanged.

    A pinned source is never omitted. It names an immutable object and its road
    is resolution, not omission - setting it aside would quietly skip a
    dependency that could have been installed exactly.
    """
    if not authorized:
        return tuple(declarations), ()
    installable: list[str] = []
    omitted: list[str] = []
    for declaration in declarations:
        url = declared_source_url(declaration)
        source = classify_source_url(url) if url else None
        if source is not None and source.repository is not None and not source.pinned:
            omitted.append(declaration)
        else:
            installable.append(declaration)
    return tuple(installable), tuple(omitted)


def declared_source_url(declaration: str) -> str | None:
    """The URL a requirement line points at, read by the packaging parser.

    One interpretation of a line, not two. Splitting on "@" by hand looked
    like the same thing and is not: it mistakes an extras marker, a bare URL
    with no name, and a revision separator for each other, and each of those
    appears in real requirements files.
    """
    line = declaration.strip()
    if not line or line.startswith("#") or line.startswith("-"):
        return None
    line = _INLINE_COMMENT.split(line, maxsplit=1)[0].strip()
    if not line:
        return None
    try:
        return Requirement(line).url
    except InvalidRequirement:
        # One spelling the parser refuses is still a source, and plainly so: a
        # line that is nothing but a URL points where it says. pip accepts it,
        # so real packages ship it - four of them among the two this product
        # has to install - and reading it as unparseable meant an unpinned
        # dependency written this way could never be set aside, while the same
        # dependency written `name @ url` could. The spelling is not the
        # question; whether the source names an exact commit is.
        #
        # Anything else here is genuinely unparseable, and nothing is set
        # aside on the strength of a line nobody could read.
        return bare_source_url(line)
