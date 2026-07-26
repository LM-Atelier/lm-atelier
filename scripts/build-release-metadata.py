from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sqlite3
import ssl
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "services" / "api" / "local_lm" / "__init__.py"
VERSION_PATTERN = re.compile(r'^__version__ = "([^"]+)"$', re.MULTILINE)
LICENSE_FILE_PATTERN = re.compile(
    r"^(license|licence|copying|copyright|notice|authors?)(\..*)?$",
    re.IGNORECASE,
)
KNOWN_LICENSES = {
    "apache 2.0": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    (
        "gplv2-or-later with a special exception which allows to use pyinstaller "
        "to build and distribute non-free programs (including commercial ones)"
    ): "GPL-2.0-or-later WITH Bootloader-exception",
    "isc": "ISC",
    "isc license (iscl)": "ISC",
    "gnu lesser general public license v2 or later (lgplv2+)": "LGPL-2.1-or-later",
    "mit": "MIT",
    "mit license": "MIT",
    "modified bsd license": "BSD-3-Clause",
    "mpl 2.0": "MPL-2.0",
    "mpl-2.0": "MPL-2.0",
    "mozilla public license 2.0 (mpl 2.0)": "MPL-2.0",
    "psf-2.0": "PSF-2.0",
    "psfl": "PSF-2.0",
    "python software foundation license": "PSF-2.0",
    "unlicense": "Unlicense",
}
REVIEWED_LICENSE_EXPRESSIONS = {
    "Apache-2.0",
    "Apache-2.0 OR BSD-2-Clause",
    "Apache-2.0 OR BSD-3-Clause",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "GPL-2.0-or-later WITH Bootloader-exception",
    "ISC",
    "LGPL-2.1-or-later",
    "MIT",
    "MIT AND PSF-2.0",
    "MIT-CMU",
    "MPL-2.0",
    "MPL-2.0 AND MIT",
    "PSF-2.0",
    "Unlicense",
}
PYTHON_LICENSE_OVERRIDES = {
    # colorama 0.4.6 omits its license from core metadata. Its shipped
    # LICENSE.txt contains the three-clause BSD terms.
    ("colorama", "0.4.6"): "BSD-3-Clause",
    # python-dateutil 2.9.0.post0 declares only "Dual License" in core metadata.
    # Its shipped LICENSE file and PyPI classifiers identify Apache-2.0 OR
    # BSD-3-Clause. Any version change must be reviewed again.
    ("python-dateutil", "2.9.0.post0"): "Apache-2.0 OR BSD-3-Clause",
}


def run_git(*arguments: str, required: bool = True) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        if required:
            raise RuntimeError(
                result.stderr.strip() or f"git {' '.join(arguments)} failed"
            )
        return ""
    return result.stdout.strip()


def canonical_version() -> str:
    match = VERSION_PATTERN.search(VERSION_FILE.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError(
            f"Could not read the application version from {VERSION_FILE}"
        )
    return match.group(1)


def normalized_license(value: Any, classifiers: list[str] | None = None) -> str:
    if isinstance(value, list):
        values = [normalized_license(item) for item in value]
        expression = (
            " OR ".join(item for item in values if item and item != "UNKNOWN")
            or "UNKNOWN"
        )
        return expression if expression in REVIEWED_LICENSE_EXPRESSIONS else "UNKNOWN"
    if isinstance(value, dict):
        return normalized_license(value.get("type") or value.get("name"))
    if isinstance(value, str):
        compact = " ".join(value.strip().split())
        if compact:
            candidate = KNOWN_LICENSES.get(compact.lower(), compact)
            if candidate in REVIEWED_LICENSE_EXPRESSIONS:
                return candidate
    for classifier in classifiers or []:
        if not classifier.startswith("License ::"):
            continue
        leaf = classifier.rsplit("::", 1)[-1].strip()
        known = KNOWN_LICENSES.get(leaf.lower())
        if known:
            return known
    return "UNKNOWN"


def declared_license(value: Any) -> str:
    if value is None:
        return "not declared"
    if isinstance(value, str):
        return " ".join(value.strip().split()) or "not declared"
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def safe_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "package"


def copy_license_file(
    source: Path, destination: Path, relative_hint: str
) -> str | None:
    if not source.is_file() or source.stat().st_size > 1_000_000:
        return None
    digest = hashlib.sha256(relative_hint.encode("utf-8")).hexdigest()[:10]
    target = destination / f"{digest}-{safe_segment(source.name)}"
    shutil.copyfile(source, target)
    return target.name


def python_components(licenses_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    components: list[dict[str, Any]] = []
    review: list[str] = []
    distributions = sorted(
        importlib.metadata.distributions(),
        key=lambda item: (item.metadata.get("Name", "").lower(), item.version),
    )
    for distribution in distributions:
        name = distribution.metadata.get("Name") or "unknown-python-package"
        if name.lower().replace("_", "-") == "lm-atelier-api":
            continue
        version = distribution.version or "unknown"
        classifiers = distribution.metadata.get_all("Classifier") or []
        normalized_name = name.lower().replace("_", "-")
        license_override = PYTHON_LICENSE_OVERRIDES.get((normalized_name, version))
        license_name = license_override or normalized_license(
            distribution.metadata.get("License-Expression")
            or distribution.metadata.get("License"),
            classifiers,
        )
        declared = declared_license(
            distribution.metadata.get("License-Expression")
            or distribution.metadata.get("License")
        )
        package_dir = (
            licenses_root / "python" / f"{safe_segment(name)}@{safe_segment(version)}"
        )
        package_dir.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for entry in distribution.files or []:
            candidate = Path(str(entry))
            if not LICENSE_FILE_PATTERN.match(candidate.name):
                continue
            located = Path(distribution.locate_file(entry))
            copied_name = copy_license_file(located, package_dir, str(candidate))
            if copied_name:
                copied.append(copied_name)
        if not copied:
            package_dir.rmdir()
        if license_name == "UNKNOWN":
            review.append(
                f"Python package {name} {version} has unreviewed license metadata: "
                f"{declared}"
            )
        purl = f"pkg:pypi/{quote(normalized_name, safe='')}@{quote(version, safe='')}"
        components.append(
            {
                "type": "library",
                "bom-ref": purl,
                "name": name,
                "version": version,
                "purl": purl,
                "licenses": [{"expression": license_name}],
                "properties": [
                    {"name": "lm-atelier:ecosystem", "value": "python"},
                    {"name": "lm-atelier:declared-license", "value": declared},
                    {
                        "name": "lm-atelier:license-review",
                        "value": (
                            "version-specific override"
                            if license_override
                            else "declared metadata"
                        ),
                    },
                    {
                        "name": "lm-atelier:license-files",
                        "value": ", ".join(copied) if copied else "none located",
                    },
                ],
            }
        )
    return components, review


def node_package_name(lock_path: str, package_document: dict[str, Any]) -> str:
    declared = package_document.get("name")
    if isinstance(declared, str) and declared:
        return declared
    relative = lock_path.rsplit("node_modules/", 1)[-1]
    return relative.replace("\\", "/")


def node_components(licenses_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    lock = json.loads((ROOT / "package-lock.json").read_text(encoding="utf-8"))
    components: list[dict[str, Any]] = []
    review: list[str] = []
    for lock_path, metadata in sorted(lock.get("packages", {}).items()):
        if not lock_path or not isinstance(metadata, dict):
            continue
        if not lock_path.startswith("node_modules/") or metadata.get("dev") is True:
            continue
        package_root = ROOT / Path(lock_path)
        package_document: dict[str, Any] = {}
        package_json = package_root / "package.json"
        if package_json.is_file():
            package_document = json.loads(package_json.read_text(encoding="utf-8"))
        name = node_package_name(lock_path, package_document)
        version = str(
            package_document.get("version") or metadata.get("version") or "unknown"
        )
        license_name = normalized_license(
            package_document.get("license") or metadata.get("license")
        )
        declared = declared_license(
            package_document.get("license") or metadata.get("license")
        )
        package_dir = (
            licenses_root / "npm" / f"{safe_segment(name)}@{safe_segment(version)}"
        )
        package_dir.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        if package_root.is_dir():
            for candidate in sorted(package_root.iterdir()):
                if not LICENSE_FILE_PATTERN.match(candidate.name):
                    continue
                copied_name = copy_license_file(candidate, package_dir, candidate.name)
                if copied_name:
                    copied.append(copied_name)
        if not copied:
            package_dir.rmdir()
        if license_name == "UNKNOWN":
            review.append(
                f"npm package {name} {version} has unreviewed license metadata: "
                f"{declared}"
            )
        encoded_name = quote(name, safe="/")
        purl = f"pkg:npm/{encoded_name}@{quote(version, safe='')}"
        components.append(
            {
                "type": "library",
                "bom-ref": purl,
                "name": name,
                "version": version,
                "purl": purl,
                "licenses": [{"expression": license_name}],
                "properties": [
                    {"name": "lm-atelier:ecosystem", "value": "npm"},
                    {"name": "lm-atelier:declared-license", "value": declared},
                    {
                        "name": "lm-atelier:license-files",
                        "value": ", ".join(copied) if copied else "none located",
                    },
                ],
            }
        )
    return components, review


def runtime_component(
    *,
    name: str,
    version: str,
    license_name: str,
    license_files: list[str],
    purl_name: str,
) -> dict[str, Any]:
    purl = f"pkg:generic/{quote(purl_name, safe='')}@{quote(version, safe='')}"
    return {
        "type": "framework",
        "bom-ref": purl,
        "name": name,
        "version": version,
        "purl": purl,
        "licenses": [{"expression": license_name}],
        "properties": [
            {"name": "lm-atelier:ecosystem", "value": "runtime"},
            {
                "name": "lm-atelier:license-files",
                "value": ", ".join(license_files),
            },
        ],
    }


def runtime_components(licenses_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    runtime_root = licenses_root / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    review: list[str] = []

    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if not python_license.is_file():
        review.append(f"CPython license was not found at {python_license}")
        python_files: list[str] = []
    else:
        destination = runtime_root / "CPython-LICENSE.txt"
        shutil.copyfile(python_license, destination)
        python_files = [destination.name]

    openssl_license = runtime_root / "OpenSSL-Apache-2.0-LICENSE.txt"
    shutil.copyfile(ROOT / "LICENSE", openssl_license)
    sqlite_notice = runtime_root / "SQLite-PUBLIC-DOMAIN.txt"
    sqlite_notice.write_text(
        "SQLite is dedicated to the public domain.\n"
        "See https://www.sqlite.org/copyright.html for the upstream notice.\n",
        encoding="utf-8",
    )

    components = [
        runtime_component(
            name="CPython",
            version=".".join(map(str, sys.version_info[:3])),
            license_name="PSF-2.0",
            license_files=python_files,
            purl_name="cpython",
        ),
        runtime_component(
            name="OpenSSL",
            version=ssl.OPENSSL_VERSION.split()[1],
            license_name="Apache-2.0",
            license_files=[openssl_license.name],
            purl_name="openssl",
        ),
        runtime_component(
            name="SQLite",
            version=sqlite3.sqlite_version,
            license_name="LicenseRef-SQLite-Public-Domain",
            license_files=[sqlite_notice.name],
            purl_name="sqlite",
        ),
    ]
    if sys.platform == "win32":
        msvc_notice = runtime_root / "Microsoft-Visual-C-Runtime-NOTICE.txt"
        msvc_notice.write_text(
            "LM Atelier's Windows build may include Microsoft Visual C++ runtime files.\n"
            "Those files remain subject to Microsoft's redistribution terms.\n"
            "See https://visualstudio.microsoft.com/license-terms/.\n",
            encoding="utf-8",
        )
        components.append(
            runtime_component(
                name="Microsoft Visual C++ Runtime",
                version="runtime selected by CPython/PyInstaller",
                license_name="LicenseRef-Microsoft-Visual-C-Runtime",
                license_files=[msvc_notice.name],
                purl_name="microsoft-visual-c-runtime",
            )
        )
    return components, review


def generated_time() -> datetime:
    source_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_epoch is not None:
        try:
            timestamp = int(source_epoch)
        except ValueError as error:
            raise RuntimeError("SOURCE_DATE_EPOCH must be an integer") from error
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return datetime.now(tz=timezone.utc)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_build_inputs() -> list[dict[str, str]]:
    relative_paths = [
        "package-lock.json",
        "services/api/uv.lock",
    ]
    return [
        {
            "path": relative_path,
            "sha256": sha256_file(ROOT / relative_path),
        }
        for relative_path in relative_paths
    ]


def command_version(command: str, *arguments: str) -> str | None:
    executable = shutil.which(command)
    if executable is None:
        return None
    result = subprocess.run(
        [executable, *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return next(
        (line.strip() for line in result.stdout.splitlines() if line.strip()),
        None,
    )


def toolchain_versions(
    installer_tool: str | None,
    installer_tool_version: str | None,
    installer_tool_sha256: str | None,
) -> dict[str, Any]:
    try:
        pyinstaller_version = importlib.metadata.version("pyinstaller")
    except importlib.metadata.PackageNotFoundError:
        pyinstaller_version = None
    return {
        "python": platform.python_version(),
        "node": command_version("node", "--version"),
        "npm": command_version("npm", "--version"),
        "uv": command_version("uv", "--version"),
        "pyinstaller": pyinstaller_version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "installer": {
            "name": installer_tool,
            "version": installer_tool_version,
            "sha256": installer_tool_sha256,
        },
    }


def sbom_serial(
    *,
    root_ref: str,
    commit: str,
    generated_at: datetime,
    lock_inputs: list[dict[str, str]],
    toolchain: dict[str, Any],
) -> str:
    identity = json.dumps(
        {
            "root_ref": root_ref,
            "commit": commit,
            "generated_at": generated_at.isoformat(),
            "locked_build_inputs": lock_inputs,
            "platform": toolchain["platform"],
            "installer": toolchain["installer"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, identity)}"


def build_notices(components: list[dict[str, Any]]) -> str:
    lines = [
        "# Third-party notices",
        "",
        "LM Atelier includes third-party software. The corresponding license texts are",
        "stored in the adjacent `third-party-licenses` directory. This generated",
        "inventory does not replace the terms in those license files.",
        "",
        "| Ecosystem | Package | Version | Declared license |",
        "| --- | --- | --- | --- |",
    ]
    for component in components:
        ecosystem = next(
            item["value"]
            for item in component["properties"]
            if item["name"] == "lm-atelier:ecosystem"
        )
        license_name = component["licenses"][0]["expression"].replace("|", "\\|")
        name = component["name"].replace("|", "\\|")
        lines.append(
            f"| {ecosystem} | {name} | {component['version']} | {license_name} |"
        )
    lines.extend(
        [
            "",
            "LM Atelier itself is licensed under Apache-2.0; see `LICENSE`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def publish_directory(staging: Path, output: Path) -> None:
    """Publish generated metadata despite short-lived Windows scanner locks."""

    for attempt in range(10):
        try:
            if output.exists():
                shutil.rmtree(output)
            staging.replace(output)
            return
        except PermissionError:
            if not staging.exists() and output.is_dir():
                return
            if attempt == 9:
                raise
            time.sleep(min(0.05 * (2**attempt), 0.5))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate installer licenses, notices, SBOM, and source metadata."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "build" / "release-metadata",
    )
    parser.add_argument(
        "--require-release-tag",
        action="store_true",
        help="Require a clean checkout at an exact v<version> tag.",
    )
    parser.add_argument(
        "--installer-tool",
        help="Identity of the platform installer tool used by the calling build script.",
    )
    parser.add_argument(
        "--installer-tool-version",
        help="Version of the platform installer tool used by the calling build script.",
    )
    parser.add_argument(
        "--installer-tool-sha256",
        help="SHA-256 of the exact platform installer executable.",
    )
    args = parser.parse_args()
    if bool(args.installer_tool) != bool(args.installer_tool_version):
        parser.error(
            "--installer-tool and --installer-tool-version must be supplied together"
        )
    if args.installer_tool_sha256 and not args.installer_tool:
        parser.error("--installer-tool-sha256 requires --installer-tool")
    if args.installer_tool_sha256 and not re.fullmatch(
        r"[0-9a-fA-F]{64}",
        args.installer_tool_sha256,
    ):
        parser.error("--installer-tool-sha256 must be a complete SHA-256 digest")

    output = args.output_dir.resolve()
    build_root = (ROOT / "build").resolve()
    if output != build_root and build_root not in output.parents:
        raise SystemExit(f"Output must remain within {build_root}")

    version = canonical_version()
    commit = run_git("rev-parse", "HEAD")
    tag = run_git("describe", "--tags", "--exact-match", required=False) or None
    dirty = bool(run_git("status", "--porcelain", "--untracked-files=all"))
    if args.require_release_tag:
        expected_tag = f"v{version}"
        if tag != expected_tag:
            raise SystemExit(
                f"Release metadata requires exact tag {expected_tag}; found {tag!r}"
            )
        if dirty:
            raise SystemExit("Release metadata requires a clean tracked worktree")

    staging = output.with_name(f"{output.name}.partial-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    licenses_root = staging / "third-party-licenses"
    licenses_root.mkdir()

    try:
        shutil.copyfile(ROOT / "LICENSE", staging / "LICENSE")
        python, python_review = python_components(licenses_root)
        node, node_review = node_components(licenses_root)
        runtime, runtime_review = runtime_components(licenses_root)
        components = sorted(
            [*python, *node, *runtime],
            key=lambda item: (item["purl"].lower(), item["version"]),
        )
        review = [*python_review, *node_review, *runtime_review]
        if review:
            raise RuntimeError(
                "Dependency license metadata requires review:\n- " + "\n- ".join(review)
            )

        generated_at = generated_time()
        lock_inputs = locked_build_inputs()
        toolchain = toolchain_versions(
            args.installer_tool,
            args.installer_tool_version.strip()
            if args.installer_tool_version
            else None,
            args.installer_tool_sha256.lower() if args.installer_tool_sha256 else None,
        )
        root_ref = f"pkg:generic/lm-atelier@{quote(version, safe='')}"
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "serialNumber": sbom_serial(
                root_ref=root_ref,
                commit=commit,
                generated_at=generated_at,
                lock_inputs=lock_inputs,
                toolchain=toolchain,
            ),
            "version": 1,
            "metadata": {
                "timestamp": generated_at.isoformat().replace("+00:00", "Z"),
                "component": {
                    "type": "application",
                    "bom-ref": root_ref,
                    "name": "LM Atelier",
                    "version": version,
                    "purl": root_ref,
                    "licenses": [{"license": {"id": "Apache-2.0"}}],
                },
                "properties": [
                    {"name": "lm-atelier:source-commit", "value": commit},
                    {"name": "lm-atelier:source-tag", "value": tag or "unreleased"},
                ],
            },
            "components": components,
            "dependencies": [
                {"ref": root_ref, "dependsOn": [item["bom-ref"] for item in components]}
            ],
        }
        write_json(staging / "sbom.cdx.json", sbom)
        (staging / "THIRD_PARTY_NOTICES.md").write_text(
            build_notices(components),
            encoding="utf-8",
        )

        manifest = json.loads(
            (ROOT / "packaging" / "release-manifest.json").read_text(encoding="utf-8")
        )
        manifest.update(
            {
                "version": version,
                "source": {
                    "sha": commit,
                    "commit": commit,
                    "tag": tag,
                    "dirty": dirty,
                },
                "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
                "dependency_locks": [item["path"] for item in lock_inputs],
                "locked_build_inputs": lock_inputs,
                "toolchain": toolchain,
                "license_inventory": "THIRD_PARTY_NOTICES.md",
                "sbom": "sbom.cdx.json",
                "signature_status": (
                    "unsigned-preview"
                    if args.require_release_tag
                    else "unsigned-development-build"
                ),
            }
        )
        write_json(staging / "release-manifest.json", manifest)

        publish_directory(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(
        f"Generated release metadata for LM Atelier {version}: "
        f"{len(components)} dependency components"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
