from __future__ import annotations

import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_FILES = {
    ".env.example",
    "docs/assets/application-preview.png",
    "docs/assets/social-preview.png",
}
DENIED_DIRECTORIES = {
    ".private",
    ".venv",
    "artifacts",
    "backups",
    "build",
    "cache",
    "caches",
    "coverage",
    "data",
    "diagnostics",
    "downloads",
    "logs",
    "models",
    "node_modules",
    "outputs",
    "release",
    "temp",
    "tmp",
}
DENIED_BASENAMES = {
    ".npmrc",
    ".pypirc",
    ".netrc",
    "credentials.json",
}
DENIED_SUFFIXES = {
    ".7z",
    ".a",
    ".aac",
    ".age",
    ".apk",
    ".appimage",
    ".appx",
    ".asc",
    ".avi",
    ".bin",
    ".bmp",
    ".bz2",
    ".cab",
    ".cer",
    ".ckpt",
    ".class",
    ".crt",
    ".csr",
    ".db",
    ".db-shm",
    ".db-wal",
    ".deb",
    ".der",
    ".dll",
    ".dmg",
    ".dylib",
    ".egg",
    ".exe",
    ".flac",
    ".gif",
    ".gguf",
    ".gpg",
    ".gz",
    ".ipa",
    ".iso",
    ".jks",
    ".jar",
    ".jpeg",
    ".jpg",
    ".key",
    ".keystore",
    ".kdbx",
    ".lib",
    ".log",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mobileprovision",
    ".msi",
    ".msix",
    ".msu",
    ".node",
    ".nupkg",
    ".o",
    ".obj",
    ".onnx",
    ".ovpn",
    ".p12",
    ".pem",
    ".pfx",
    ".pkcs8",
    ".pkg",
    ".png",
    ".ppk",
    ".pt",
    ".pth",
    ".pyc",
    ".pyd",
    ".pyo",
    ".rar",
    ".rpm",
    ".run",
    ".safetensors",
    ".so",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tgz",
    ".wav",
    ".war",
    ".wasm",
    ".webm",
    ".webp",
    ".whl",
    ".xz",
    ".zip",
    ".zst",
}
SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    # A real key starts at a token boundary; without this, any hyphenated
    # identifier containing "sk-" trips it - "mask-feather-out-of-range" and
    # "task-scheduler-configuration" both matched before.
    re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
)
# Some files carry machine-readable handling flags saying they must never be
# published. Matching the declaration rather than a path means the rule survives
# a rename, a copy, or an excerpt pasted into another document - which is how
# this kind of content actually escapes.
SELF_EXCLUDING_MARKERS = (
    re.compile(r'"never_commit"\s*:\s*true'),
    re.compile(r'"never_publish"\s*:\s*true'),
    re.compile(r'"never_include_in_public_documentation"\s*:\s*true'),
)

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*]\((?P<target>[^)]+)\)")
# A UTF-8 sequence re-read as cp1252 or latin-1 always begins with one of
# these pairs. They are written as escapes so this file stays ASCII and
# cannot itself be mangled, which is the discipline the check asks for.
MOJIBAKE_MARKERS = (
    "\u00e2\u20ac",  # a-circumflex + euro: smart punctuation as cp1252
    "\u00c3\u00a2",  # A-tilde + a-circumflex: the same text mangled twice
    "\u00c2\u00a0",  # A-circumflex + no-break space
)


def candidate_paths() -> list[str]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
    )
    decoded = (item.decode("utf-8") for item in result.stdout.split(b"\0") if item)
    return sorted(
        item for item in decoded if Path(item).is_file() or Path(item).is_symlink()
    )


def unsafe_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    lowered = normalized.lower()
    if lowered in ALLOWED_FILES:
        return False
    path = PurePosixPath(lowered)
    if any(part in DENIED_DIRECTORIES for part in path.parts[:-1]):
        return True
    name = path.name
    if name == ".env" or name.startswith(".env."):
        return True
    if name in DENIED_BASENAMES or (
        name.startswith("service-account") and name.endswith(".json")
    ):
        return True
    return any(name.endswith(suffix) for suffix in DENIED_SUFFIXES)


def contains_secret(path: str) -> bool:
    payload = Path(path).read_bytes()
    if b"\0" in payload:
        return False
    text = payload.decode("utf-8", errors="ignore")
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


# Vendored third-party text is not ours to reformat. A licence in
# particular has to ship byte for byte as its author published it, and one
# of the OFL files carries trailing whitespace upstream.
VENDORED_TEXT = ("apps/web/public/fonts/",)


def declares_it_must_not_ship(path: str) -> bool:
    """Whether a file carries content that declared itself unpublishable."""

    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return any(pattern.search(text) for pattern in SELF_EXCLUDING_MARKERS)


def is_vendored(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized.startswith(VENDORED_TEXT)


def has_trailing_whitespace(path: str) -> bool:
    payload = Path(path).read_bytes()
    if b"\0" in payload:
        return False
    text = payload.decode("utf-8", errors="ignore")
    return any(line.endswith((" ", "\t")) for line in text.splitlines())


def _coordination_patterns() -> tuple[re.Pattern[str], ...]:
    """Build the forbidden patterns from fragments, at run time.

    Spelling any of them out here would make this file match its own rule, and a
    checker that rejects itself is not a checker. Nothing joined below appears
    whole anywhere in tracked source.
    """

    # Joined through a generator so the fragments are never a constant
    # expression. Written as a literal tuple of joins, a linter correctly offers
    # to fold each one into the whole word, and taking that offer would make
    # this file match its own rule.
    sender_fragments = (("co", "dex"), ("cla", "ude"), ("gr", "ok"))
    authority_fragments = (("own", "er"), ("deci", "sion"))
    senders = "|".join("".join(fragment) for fragment in sender_fragments)
    # Ordinary words, written plainly: on their own they match nothing, because
    # the pattern below requires a sender name immediately before one.
    verbs = "adopted|approved|rejected|requested|reviewed|decided|raised|authorized"
    authority = r"\s+".join("".join(fragment) for fragment in authority_fragments)
    return (
        # A sender token followed by an R-number, which is the shape of a private
        # request identifier. A bare name is deliberately not matched: it occurs
        # legitimately inside branch names in tracked design notes.
        re.compile(rf"\b(?:{senders})/R\d+\b", re.IGNORECASE),
        # A sender named as the actor behind a change. The verb set is narrow for
        # the same reason, so that merely mentioning a name is not a match.
        re.compile(rf"\b(?:{senders})\s+(?:{verbs})\b", re.IGNORECASE),
        # The private record of authority, cited as the reason for a change.
        re.compile(rf"\b{authority}s?\b", re.IGNORECASE),
    )


COORDINATION_PATTERNS = _coordination_patterns()


def _forbidden_in(text: str) -> bool:
    return any(pattern.search(text) for pattern in COORDINATION_PATTERNS)


def _reportable(path: str) -> str:
    """A location safe to print, with any forbidden token masked out.

    The refusal is printed into the build log, and a build log is public. A path
    that itself carries an identifier would therefore publish it in the very
    output that exists to keep it unpublished. Masking the match leaves the
    directory and the extension, which is enough to find the file.
    """

    normalised = path.replace("\\", "/")
    if not _forbidden_in(normalised):
        # Nothing to mask, so the location is returned exactly as given. A
        # refusal has to name a file the reader can open, and rewriting every
        # safe location would make the common case worse to serve the rare one.
        return path
    # Masked against the NORMALISED copy, not the original. The identifier shape
    # contains a slash, so a native path can only be matched after normalising;
    # substituting against the original would replace nothing and report the raw
    # token, having correctly detected it.
    safe = normalised
    for pattern in COORDINATION_PATTERNS:
        safe = pattern.sub("[redacted]", safe)
    return safe


def _listed(locations: list[str]) -> str:
    """Join locations for a refusal, with any forbidden token masked.

    Every refusal in this checker goes through here rather than joining raw
    strings, because the build log is public and a path can carry the very token
    these rules exist to keep out of it.
    """

    return "\n- ".join(_reportable(location) for location in locations)


def private_coordination_lines(path: str) -> list[str]:
    """Return safe locations for private coordination text in a candidate.

    Tracked public text must not name the private request identifiers, nor an
    agent as the actor behind a change, nor the private record of authority for
    it. The rule was enforced only by whoever read the diff, so its failure mode
    was a review round trip rather than a red gate. It has been missed three
    times, and twice the bytes were public before a reader caught them.

    The PATH is checked as well as the contents, because a tracked pathname is
    already published in repository metadata: a file whose contents are perfectly
    ordinary still leaks the identifier if it is the name of the file. Separators
    are normalised first, since the identifier shape contains a slash and a
    caller on Windows may hand over a native path.

    Nothing reported here carries the matched text - not the line, and not the
    path when the path is what matched.
    """

    found: list[str] = []
    if _forbidden_in(path.replace("\\", "/")):
        found.append(f"{_reportable(path)}: the path itself")
    payload = Path(path).read_bytes()
    if b"\x00" in payload:
        return found
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return found
    found.extend(
        f"{_reportable(path)}:{number}"
        for number, line in enumerate(text.splitlines(), 1)
        if _forbidden_in(line)
    )
    return found


def mojibake_lines(path: str) -> list[str]:
    """Return lines showing a UTF-8 text decoded as a single-byte codepage.

    Writing a UTF-8 file with a cp1252 tool turns each multi-byte character into
    a run of Latin-1 characters, and the result still decodes as valid UTF-8, so
    nothing else notices. It reached a live regex once already: a bullet in a
    routing pattern became six Latin-1 characters, and the class silently stopped
    matching bulleted lists. These sequences do not occur in correct text.
    """

    payload = Path(path).read_bytes()
    if b"\0" in payload:
        return []
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return []
    found: list[str] = []
    for number, line in enumerate(text.splitlines(), 1):
        if any(marker in line for marker in MOJIBAKE_MARKERS):
            readable = "".join(
                character if character.isascii() else f"<U+{ord(character):04X}>"
                for character in line.strip()
            )
            found.append(f"{path}:{number}: {readable[:120]}")
    return found


def broken_local_links(paths: list[str]) -> list[str]:
    """Return local Markdown links that do not resolve inside the repository."""

    broken: list[str] = []
    for path in paths:
        if Path(path).suffix.lower() != ".md":
            continue
        text = Path(path).read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            target = match.group("target").strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            else:
                target = target.split(maxsplit=1)[0]
            if (
                not target
                or target.startswith(("#", "/", "mailto:"))
                or "://" in target
            ):
                continue
            relative = unquote(target.split("#", 1)[0].split("?", 1)[0])
            if not relative:
                continue
            candidate = (ROOT / Path(path).parent / relative).resolve()
            if not candidate.is_relative_to(ROOT) or not candidate.exists():
                broken.append(f"{path} -> {target}")
    return broken


def main() -> int:
    paths = candidate_paths()

    # This one stays first, because it is the only rule here that decides
    # whether a candidate may be OPENED at all. It reads no file: it answers
    # from the path and a symlink test. Every rule below reads bytes, so a
    # candidate link would be followed and read before this could refuse it, and
    # the private tree would be inspected before the rule that owns excluding it.
    unsafe = [path for path in paths if unsafe_path(path) or Path(path).is_symlink()]
    if unsafe:
        raise SystemExit(
            "Private, generated, executable, or runtime artifacts are in the "
            "candidate:\n- " + _listed(unsafe)
        )

    # First among the rules that read content. The others report raw paths, and
    # the mojibake rule reports the offending line itself, so reaching any of
    # them with a forbidden pathname or a forbidden line would publish the token
    # into the build log before this rule could reduce it to a location.
    # Ordering is the guarantee; `_listed` is the second line of defence for the
    # day someone reorders this.
    coordination = [line for path in paths for line in private_coordination_lines(path)]
    if coordination:
        raise SystemExit(
            "Private coordination text is present in candidate files. Tracked "
            "public text must not carry request identifiers, agent attribution, "
            "or the private record of authority for a change:\n- "
            + _listed(coordination)
        )

    self_excluding = [path for path in paths if declares_it_must_not_ship(path)]
    if self_excluding:
        raise SystemExit(
            "Files that declare they must never be published are in the "
            "candidate:\n- " + _listed(self_excluding)
        )

    secret_paths = [path for path in paths if contains_secret(path)]
    if secret_paths:
        raise SystemExit(
            "Likely credentials are present in candidate files:\n- "
            + _listed(secret_paths)
        )

    whitespace_paths = [
        path
        for path in paths
        if not is_vendored(path) and has_trailing_whitespace(path)
    ]
    if whitespace_paths:
        raise SystemExit(
            "Trailing whitespace is present in candidate files:\n- "
            + _listed(whitespace_paths)
        )

    mangled = [line for path in paths for line in mojibake_lines(path)]
    if mangled:
        raise SystemExit(
            "Text mangled by a codepage round trip is present. Rewrite the line "
            "as UTF-8, and prefer an explicit escape such as \\u2022 over a "
            "literal non-ASCII character inside a regular expression:\n- "
            + _listed(mangled)
        )

    broken_links = broken_local_links(paths)
    if broken_links:
        raise SystemExit(
            "Broken or out-of-repository local Markdown links are present:\n- "
            + _listed(broken_links)
        )

    print(f"Repository hygiene passed for {len(paths)} candidate files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
