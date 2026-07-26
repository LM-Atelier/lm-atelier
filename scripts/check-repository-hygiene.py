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
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*]\((?P<target>[^)]+)\)")


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


def has_trailing_whitespace(path: str) -> bool:
    payload = Path(path).read_bytes()
    if b"\0" in payload:
        return False
    text = payload.decode("utf-8", errors="ignore")
    return any(line.endswith((" ", "\t")) for line in text.splitlines())


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

    secret_paths = [path for path in paths if contains_secret(path)]
    if secret_paths:
        raise SystemExit(
            "Likely credentials are present in candidate files:\n- "
            + "\n- ".join(secret_paths)
        )

    whitespace_paths = [path for path in paths if has_trailing_whitespace(path)]
    if whitespace_paths:
        raise SystemExit(
            "Trailing whitespace is present in candidate files:\n- "
            + "\n- ".join(whitespace_paths)
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
