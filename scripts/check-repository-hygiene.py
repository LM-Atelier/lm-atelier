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
# Private preference data declares its own handling rules, and the one that
# matters here is `never_commit`. Recognising the declaration rather than a path
# means the check still works when the file is renamed, copied, or pasted into
# something else - which is how this kind of content actually escapes.
PRIVATE_DATA_MARKERS = (
    re.compile(r'"classification"\s*:\s*"private_sensitive_user_preference_data"'),
    re.compile(r'"never_commit"\s*:\s*true'),
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


def contains_private_preference_data(path: str) -> bool:
    """Whether a file carries content that declared itself uncommittable."""

    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return any(pattern.search(text) for pattern in PRIVATE_DATA_MARKERS)


def is_vendored(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized.startswith(VENDORED_TEXT)


def has_trailing_whitespace(path: str) -> bool:
    payload = Path(path).read_bytes()
    if b"\0" in payload:
        return False
    text = payload.decode("utf-8", errors="ignore")
    return any(line.endswith((" ", "\t")) for line in text.splitlines())


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
    unsafe = [path for path in paths if unsafe_path(path) or Path(path).is_symlink()]
    if unsafe:
        raise SystemExit(
            "Private, generated, executable, or runtime artifacts are in the "
            "candidate:\n- " + "\n- ".join(unsafe)
        )

    private_data = [path for path in paths if contains_private_preference_data(path)]
    if private_data:
        raise SystemExit(
            "Private preference data declares never_commit and is in the "
            "candidate:\n- " + "\n- ".join(private_data)
        )

    secret_paths = [path for path in paths if contains_secret(path)]
    if secret_paths:
        raise SystemExit(
            "Likely credentials are present in candidate files:\n- "
            + "\n- ".join(secret_paths)
        )

    whitespace_paths = [
        path
        for path in paths
        if not is_vendored(path) and has_trailing_whitespace(path)
    ]
    if whitespace_paths:
        raise SystemExit(
            "Trailing whitespace is present in candidate files:\n- "
            + "\n- ".join(whitespace_paths)
        )

    mangled = [line for path in paths for line in mojibake_lines(path)]
    if mangled:
        raise SystemExit(
            "Text mangled by a codepage round trip is present. Rewrite the line "
            "as UTF-8, and prefer an explicit escape such as \\u2022 over a "
            "literal non-ASCII character inside a regular expression:\n- "
            + "\n- ".join(mangled)
        )

    broken_links = broken_local_links(paths)
    if broken_links:
        raise SystemExit(
            "Broken or out-of-repository local Markdown links are present:\n- "
            + "\n- ".join(broken_links)
        )

    print(f"Repository hygiene passed for {len(paths)} candidate files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
