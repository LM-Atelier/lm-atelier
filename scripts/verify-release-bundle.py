"""Fail-closed verification for release artifacts crossing Actions job boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _platform_files(platform_name: str, version: str) -> set[str]:
    installer = {
        "windows": f"LM-Atelier-Setup-{version}-windows-x86_64.exe",
        "linux": f"LM-Atelier-Setup-{version}-linux-x86_64.run",
    }[platform_name]
    license_archive = {
        "windows": "third-party-licenses-windows.zip",
        "linux": "third-party-licenses-linux.tar.gz",
    }[platform_name]
    return {
        installer,
        f"LICENSE-{platform_name}.txt",
        f"THIRD_PARTY_NOTICES-{platform_name}.md",
        f"release-manifest-{platform_name}.json",
        f"sbom-{platform_name}.cdx.json",
        f"payload-manifest-{platform_name}.json",
        license_archive,
        f"gitleaks-{platform_name}-payload.json",
        f"gitleaks-{platform_name}-metadata.json",
        f"gitleaks-{platform_name}-installer.json",
        f"npm-audit-{platform_name}.json",
        f"pip-audit-{platform_name}.json",
        f"malware-scan-{platform_name}.txt",
        f"SHA256SUMS-{platform_name}",
    }


def _is_link_or_junction(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _flat_files(root: Path) -> set[str]:
    files: set[str] = set()
    for entry in root.iterdir():
        if _is_link_or_junction(entry) or not entry.is_file():
            raise RuntimeError(
                f"Release bundle contains a non-regular entry: {entry.name}"
            )
        files.add(entry.name)
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"Expected a JSON object in {path.name}")
    return document


def _verify_checksums(root: Path, platform_name: str, expected: set[str]) -> None:
    checksum_name = f"SHA256SUMS-{platform_name}"
    checksum_path = root / checksum_name
    expected_payload = expected - {checksum_name}
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        digest, separator, name = line.partition("  ")
        if (
            not separator
            or not SHA256.fullmatch(digest)
            or name not in expected_payload
            or name in entries
        ):
            raise RuntimeError(
                f"{checksum_name}:{line_number}: invalid or duplicate checksum entry"
            )
        entries[name] = digest
    if set(entries) != expected_payload:
        missing = sorted(expected_payload - set(entries))
        unexpected = sorted(set(entries) - expected_payload)
        raise RuntimeError(
            f"{checksum_name} inventory mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, expected_digest in entries.items():
        if _sha256(root / name) != expected_digest:
            raise RuntimeError(f"Checksum verification failed for {name}")


def _source(document: dict[str, Any], label: str) -> dict[str, Any]:
    source = document.get("source")
    if not isinstance(source, dict):
        raise TypeError(f"{label} has no source object")
    return source


def _verify_metadata(
    root: Path,
    platform_name: str,
    version: str,
    source_sha: str,
) -> None:
    tag = f"v{version}"
    release = _read_json(root / f"release-manifest-{platform_name}.json")
    payload = _read_json(root / f"payload-manifest-{platform_name}.json")
    sbom = _read_json(root / f"sbom-{platform_name}.cdx.json")
    release_source = _source(release, "Release manifest")
    payload_source = _source(payload, "Payload manifest")
    expected_source = {
        "sha": source_sha,
        "commit": source_sha,
        "tag": tag,
        "dirty": False,
    }
    if release.get("application") != "LM Atelier" or release.get("version") != version:
        raise RuntimeError(f"{platform_name} release manifest identity is inconsistent")
    if release_source != expected_source:
        raise RuntimeError(f"{platform_name} release manifest source is inconsistent")
    if (
        payload.get("application") != "LM Atelier"
        or payload.get("version") != version
        or payload_source != expected_source
    ):
        raise RuntimeError(f"{platform_name} payload manifest source is inconsistent")
    if payload.get("generated_at") != release.get("generated_at"):
        raise RuntimeError(f"{platform_name} payload generation time is inconsistent")

    metadata = sbom.get("metadata")
    component = metadata.get("component") if isinstance(metadata, dict) else None
    properties = metadata.get("properties") if isinstance(metadata, dict) else None
    if (
        sbom.get("bomFormat") != "CycloneDX"
        or not isinstance(component, dict)
        or component.get("name") != "LM Atelier"
        or component.get("version") != version
        or not isinstance(properties, list)
    ):
        raise RuntimeError(f"{platform_name} SBOM identity is inconsistent")
    property_map = {
        item.get("name"): item.get("value")
        for item in properties
        if isinstance(item, dict)
    }
    if property_map.get("lm-atelier:source-commit") != source_sha:
        raise RuntimeError(f"{platform_name} SBOM source commit is inconsistent")
    if property_map.get("lm-atelier:source-tag") != tag:
        raise RuntimeError(f"{platform_name} SBOM source tag is inconsistent")


def verify_bundle(
    root: Path,
    platforms: tuple[str, ...],
    version: str,
    source_sha: str,
) -> None:
    if _is_link_or_junction(root):
        raise RuntimeError(f"Release bundle is not a regular directory: {root}")
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"Release bundle is not a regular directory: {root}")
    expected: set[str] = set()
    for platform_name in platforms:
        expected.update(_platform_files(platform_name, version))
    actual = _flat_files(root)
    if actual != expected:
        raise RuntimeError(
            "Release bundle inventory mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    for platform_name in platforms:
        platform_files = _platform_files(platform_name, version)
        _verify_checksums(root, platform_name, platform_files)
        _verify_metadata(root, platform_name, version, source_sha)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the exact inventory, checksums, and source identity of a release bundle."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--platform", choices=("windows", "linux", "all"), required=True
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    if (
        re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z][0-9A-Za-z.-]*)?", args.version)
        is None
    ):
        parser.error("--version must be an exact SemVer-style version")
    if (
        SHA256.fullmatch(args.source_sha) is None
        and re.fullmatch(r"[0-9a-f]{40}", args.source_sha) is None
    ):
        parser.error("--source-sha must be a complete Git object ID")
    platforms = ("windows", "linux") if args.platform == "all" else (args.platform,)
    verify_bundle(args.bundle, platforms, args.version, args.source_sha)
    print(
        f"Verified {args.platform} release bundle for "
        f"LM Atelier {args.version} at {args.source_sha}"
    )


if __name__ == "__main__":
    main()
