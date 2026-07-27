from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
from collections.abc import Iterator
from email.parser import Parser
from pathlib import Path
from typing import Any
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = (ROOT / "build").resolve()
PAYLOAD_MANIFEST_RELATIVE = Path("_internal/payload-manifest.json")
METADATA_FILES = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "release-manifest.json",
    "sbom.cdx.json",
)
LICENSE_FILE_PATTERN = re.compile(
    r"^(license|licence|copying|copyright|notice|authors?)(\..*)?$",
    re.IGNORECASE,
)
REVIEWED_FROZEN_LICENSE_EXPRESSIONS = {
    "Apache-2.0",
    "Apache-2.0 OR BSD-2-Clause",
    "Apache-2.0 OR BSD-3-Clause",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "MIT",
    "MIT AND PSF-2.0",
    "MIT-CMU",
    "MPL-2.0",
    "MPL-2.0 AND MIT",
    "PSF-2.0",
    "Unlicense",
}
FROZEN_LICENSE_ALIASES = {
    "apache 2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "isc license (iscl)": "ISC",
    "mit license": "MIT",
    "modified bsd license": "BSD-3-Clause",
    "mozilla public license 2.0 (mpl 2.0)": "MPL-2.0",
    "python software foundation license": "PSF-2.0",
}


def canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def safe_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "package"


def frozen_license_expression(metadata: Any) -> tuple[str, str]:
    raw = metadata.get("License-Expression") or metadata.get("License")
    declared = " ".join(str(raw or "").strip().split()) or "not declared"
    candidate = FROZEN_LICENSE_ALIASES.get(declared.lower(), declared)
    if candidate not in REVIEWED_FROZEN_LICENSE_EXPRESSIONS:
        raise RuntimeError(
            "Frozen distribution has unreviewed license metadata: "
            f"{metadata.get('Name')} {metadata.get('Version')} ({declared})"
        )
    return candidate, declared


def read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return document


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial-{os.getpid()}")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_within(path: Path, parent: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved != parent and parent not in resolved.parents:
        raise RuntimeError(f"{label} must remain within {parent}: {resolved}")
    return resolved


def nested_source_paths(value: Any) -> Iterator[Path]:
    if (
        isinstance(value, tuple)
        and len(value) >= 2
        and isinstance(value[1], str)
        and Path(value[1]).is_absolute()
    ):
        yield Path(value[1])
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from nested_source_paths(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from nested_source_paths(item)


def distribution_file_index() -> tuple[
    dict[tuple[str, str], importlib.metadata.Distribution],
    dict[str, set[tuple[str, str]]],
]:
    distributions: dict[tuple[str, str], importlib.metadata.Distribution] = {}
    files: dict[str, set[tuple[str, str]]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if not name or not version:
            continue
        key = (canonical_name(name), version)
        distributions[key] = distribution
        for entry in distribution.files or []:
            located = Path(distribution.locate_file(entry))
            normalized = os.path.normcase(os.path.realpath(located))
            files.setdefault(normalized, set()).add(key)
    return distributions, files


def analyzed_distributions(analysis_toc: Path) -> set[tuple[str, str]]:
    document = ast.literal_eval(analysis_toc.read_text(encoding="utf-8"))
    distributions, file_index = distribution_file_index()
    detected: set[tuple[str, str]] = set()
    for source in nested_source_paths(document):
        normalized = os.path.normcase(os.path.realpath(source))
        detected.update(file_index.get(normalized, set()))

    pyinstaller = next(
        (key for key in distributions if key[0] == "pyinstaller"),
        None,
    )
    if pyinstaller:
        # The bootloader is part of every frozen payload even though its source
        # executable is produced in PyInstaller's work directory.
        detected.add(pyinstaller)
    return detected


def frozen_metadata_distributions(payload_root: Path) -> set[tuple[str, str]]:
    detected: set[tuple[str, str]] = set()
    for metadata_path in payload_root.rglob("*.dist-info/METADATA"):
        metadata = Parser().parsestr(metadata_path.read_text(encoding="utf-8"))
        name = metadata.get("Name")
        version = metadata.get("Version")
        if name and version:
            detected.add((canonical_name(name), version))
    return detected


def _component_property(component: dict[str, Any], name: str) -> str:
    for property_ in component.get("properties", []):
        if property_.get("name") == name:
            return str(property_.get("value", ""))
    raise RuntimeError(f"SBOM component {component.get('name')} is missing {name}")


def write_notices(path: Path, components: list[dict[str, Any]]) -> None:
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
        ecosystem = _component_property(component, "lm-atelier:ecosystem")
        licenses = component.get("licenses")
        if not isinstance(licenses, list) or not licenses:
            raise RuntimeError(
                f"SBOM component {component.get('name')} has no license expression"
            )
        license_name = str(licenses[0].get("expression", "")).replace("|", "\\|")
        if not license_name:
            raise RuntimeError(
                f"SBOM component {component.get('name')} has no license expression"
            )
        name = str(component.get("name", "")).replace("|", "\\|")
        version = str(component.get("version", "")).replace("|", "\\|")
        lines.append(f"| {ecosystem} | {name} | {version} | {license_name} |")
    lines.extend(
        [
            "",
            "LM Atelier itself is licensed under Apache-2.0; see `LICENSE`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def augment_sbom_with_frozen_metadata(
    sbom: dict[str, Any],
    payload_root: Path,
    metadata_root: Path,
) -> int:
    components = sbom.get("components")
    if not isinstance(components, list):
        raise TypeError("SBOM components must be a list")
    existing = {
        (
            canonical_name(str(component.get("name", ""))),
            str(component.get("version", "")),
        )
        for component in components
        if isinstance(component, dict) and component_ecosystem(component) == "python"
    }
    added = 0
    for metadata_path in sorted(payload_root.rglob("*.dist-info/METADATA")):
        metadata = Parser().parsestr(metadata_path.read_text(encoding="utf-8"))
        name = metadata.get("Name")
        version = metadata.get("Version")
        if not name or not version:
            raise RuntimeError(
                f"Frozen distribution metadata is incomplete: {metadata_path}"
            )
        key = (canonical_name(name), version)
        if key in existing or key[0] == "lm-atelier-api":
            continue

        license_name, declared = frozen_license_expression(metadata)
        license_files = sorted(
            path
            for path in metadata_path.parent.rglob("*")
            if path.is_file() and LICENSE_FILE_PATTERN.match(path.name)
        )
        if not license_files:
            raise RuntimeError(
                f"Frozen distribution has no copied license file: {name} {version}"
            )
        destination = (
            metadata_root
            / "third-party-licenses"
            / "python"
            / f"{safe_segment(name)}@{safe_segment(version)}"
        )
        destination.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for source in license_files:
            relative = source.relative_to(payload_root).as_posix()
            digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:10]
            target = destination / f"{digest}-{safe_segment(source.name)}"
            shutil.copyfile(source, target)
            copied.append(target.name)

        relative_metadata = metadata_path.relative_to(payload_root)
        properties = [
            {"name": "lm-atelier:ecosystem", "value": "python"},
            {"name": "lm-atelier:declared-license", "value": declared},
            {
                "name": "lm-atelier:license-review",
                "value": "frozen vendored metadata",
            },
            {
                "name": "lm-atelier:license-files",
                "value": ", ".join(copied),
            },
            {
                "name": "lm-atelier:frozen-metadata-path",
                "value": relative_metadata.as_posix(),
            },
        ]
        parts = relative_metadata.parts
        if "_vendor" in parts:
            vendor_index = parts.index("_vendor")
            if vendor_index > 0:
                properties.append(
                    {
                        "name": "lm-atelier:vendored-by",
                        "value": parts[vendor_index - 1],
                    }
                )
        purl = f"pkg:pypi/{quote(key[0], safe='')}@{quote(version, safe='')}"
        components.append(
            {
                "type": "library",
                "bom-ref": purl,
                "name": name,
                "version": version,
                "purl": purl,
                "licenses": [{"expression": license_name}],
                "properties": properties,
            }
        )
        existing.add(key)
        added += 1

    if added:
        components.sort(
            key=lambda component: (
                str(component.get("purl", "")).lower(),
                str(component.get("version", "")),
            )
        )
        write_notices(metadata_root / "THIRD_PARTY_NOTICES.md", components)
    return added


def bundled_node_packages(payload_root: Path) -> set[str]:
    """Read npm package identities from the Vite production source maps."""

    web_root = payload_root / "_internal" / "web"
    source_maps = sorted(web_root.rglob("*.js.map"))
    if not source_maps:
        raise RuntimeError(
            "The frozen web payload has no JavaScript source maps for SBOM reconciliation"
        )
    packages: set[str] = set()
    for source_map in source_maps:
        document = read_json(source_map)
        sources = document.get("sources")
        if not isinstance(sources, list):
            raise TypeError(f"Vite source map has no sources list: {source_map}")
        for source in sources:
            if not isinstance(source, str):
                raise TypeError(
                    f"Vite source map contains a non-string source: {source_map}"
                )
            normalized = source.replace("\\", "/")
            if "node_modules/" not in normalized:
                continue
            relative = normalized.rsplit("node_modules/", 1)[1]
            parts = [part for part in relative.split("/") if part]
            if not parts:
                continue
            if parts[0].startswith("@"):
                if len(parts) < 2:
                    raise RuntimeError(
                        f"Vite source map has an incomplete scoped package path: {source}"
                    )
                packages.add(f"{parts[0]}/{parts[1]}")
            else:
                packages.add(parts[0])
    if not packages:
        raise RuntimeError(
            "The Vite source maps do not identify any bundled npm packages"
        )
    return packages


def component_ecosystem(component: dict[str, Any]) -> str | None:
    for item in component.get("properties", []):
        if item.get("name") == "lm-atelier:ecosystem":
            return str(item.get("value"))
    return None


def set_property(component: dict[str, Any], name: str, value: str) -> None:
    properties = [
        item for item in component.get("properties", []) if item.get("name") != name
    ]
    properties.append({"name": name, "value": value})
    component["properties"] = properties


def reconcile_sbom(
    sbom: dict[str, Any],
    included_python: set[tuple[str, str]],
    included_npm: set[str],
) -> tuple[int, int, int, int]:
    python_components: dict[tuple[str, str], dict[str, Any]] = {}
    npm_components: dict[str, dict[str, Any]] = {}
    required_refs: list[str] = []
    included_python_count = 0
    excluded_python_count = 0
    included_npm_count = 0
    excluded_npm_count = 0

    components = sbom.get("components")
    if not isinstance(components, list):
        raise TypeError("SBOM components must be a list")
    for raw_component in components:
        if not isinstance(raw_component, dict):
            raise TypeError("SBOM component entries must be objects")
        component = raw_component
        ecosystem = component_ecosystem(component)
        if ecosystem == "python":
            key = (
                canonical_name(str(component.get("name", ""))),
                str(component.get("version", "")),
            )
            python_components[key] = component
            if key in included_python:
                component["scope"] = "required"
                set_property(
                    component,
                    "lm-atelier:payload-status",
                    "included in frozen application",
                )
                included_python_count += 1
                required_refs.append(str(component["bom-ref"]))
            else:
                component["scope"] = "excluded"
                set_property(
                    component,
                    "lm-atelier:payload-status",
                    "build environment only",
                )
                excluded_python_count += 1
        elif ecosystem == "npm":
            name = str(component.get("name", ""))
            npm_components[name] = component
            if name in included_npm:
                component["scope"] = "required"
                set_property(
                    component,
                    "lm-atelier:payload-status",
                    "embedded in compiled web bundle",
                )
                included_npm_count += 1
                required_refs.append(str(component["bom-ref"]))
            else:
                component["scope"] = "excluded"
                set_property(
                    component,
                    "lm-atelier:payload-status",
                    "not present after production web tree-shaking",
                )
                excluded_npm_count += 1
        else:
            component["scope"] = "required"
            set_property(
                component,
                "lm-atelier:payload-status",
                "included runtime component",
            )
            required_refs.append(str(component["bom-ref"]))

    missing = sorted(
        key
        for key in included_python
        if key not in python_components and key[0] != "lm-atelier-api"
    )
    if missing:
        details = ", ".join(f"{name}=={version}" for name, version in missing)
        raise RuntimeError(
            "Frozen Python distributions are missing from the SBOM: " + details
        )
    missing_npm = sorted(included_npm - npm_components.keys())
    if missing_npm:
        raise RuntimeError(
            "Bundled npm packages are missing from the SBOM: " + ", ".join(missing_npm)
        )

    root_ref = str(sbom["metadata"]["component"]["bom-ref"])
    sbom["dependencies"] = [{"ref": root_ref, "dependsOn": sorted(required_refs)}]
    metadata = sbom["metadata"]
    set_property(
        metadata,
        "lm-atelier:payload-python-components",
        str(included_python_count),
    )
    set_property(
        metadata,
        "lm-atelier:excluded-build-python-components",
        str(excluded_python_count),
    )
    set_property(
        metadata,
        "lm-atelier:payload-npm-components",
        str(included_npm_count),
    )
    set_property(
        metadata,
        "lm-atelier:excluded-npm-components",
        str(excluded_npm_count),
    )
    set_property(
        metadata,
        "lm-atelier:payload-manifest",
        "payload-manifest.json",
    )
    return (
        included_python_count,
        excluded_python_count,
        included_npm_count,
        excluded_npm_count,
    )


def sync_release_metadata(metadata_root: Path, payload_root: Path) -> None:
    internal_entry = payload_root / "_internal"
    if internal_entry.is_symlink() or bool(
        getattr(internal_entry, "is_junction", lambda: False)()
    ):
        raise RuntimeError(
            f"Frozen payload _internal must be a regular directory: {internal_entry}"
        )
    internal = ensure_within(internal_entry, payload_root, "Metadata destination")
    if not internal.is_dir():
        raise RuntimeError(f"Frozen payload has no _internal directory: {payload_root}")
    for name in METADATA_FILES:
        source = metadata_root / name
        if not source.is_file():
            raise RuntimeError(f"Release metadata is missing {source}")
        destination = ensure_within(
            internal / name,
            payload_root,
            f"{name} destination",
        )
        shutil.copyfile(source, destination)

    source_licenses = metadata_root / "third-party-licenses"
    destination_licenses = internal / "third-party-licenses"
    if not source_licenses.is_dir():
        raise RuntimeError(f"Release licenses are missing: {source_licenses}")
    ensure_within(destination_licenses, payload_root, "License destination")
    if destination_licenses.exists():
        shutil.rmtree(destination_licenses)
    shutil.copytree(source_licenses, destination_licenses)


def allowed_installer_extra(relative: str, platform_name: str | None) -> bool:
    if platform_name == "linux":
        return relative in {
            ".lm-atelier-install",
            "lm-atelier.png",
            "uninstall.sh",
        }
    if platform_name == "windows":
        return bool(re.fullmatch(r"unins[0-9]{3}\.(dat|exe|msg)", relative))
    return False


def payload_entries(
    payload_root: Path,
    *,
    installer_extras: str | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(payload_root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(payload_root).as_posix()
        if relative == PAYLOAD_MANIFEST_RELATIVE.as_posix():
            continue
        if allowed_installer_extra(relative, installer_extras):
            continue
        if bool(getattr(path, "is_junction", lambda: False)()):
            raise RuntimeError(f"Payload junctions are prohibited: {path}")
        if path.is_symlink():
            target = os.readlink(path)
            resolved_target = (path.parent / target).resolve()
            ensure_within(resolved_target, payload_root, "Payload symlink target")
            encoded_target = target.encode("utf-8")
            entries.append(
                {
                    "path": relative,
                    "type": "symlink",
                    "target": target,
                    "size": len(encoded_target),
                    "sha256": hashlib.sha256(encoded_target).hexdigest(),
                }
            )
        elif path.is_file():
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        elif not path.is_dir():
            raise RuntimeError(f"Unsupported payload entry: {path}")
    return entries


def create_inventory(
    payload_root: Path,
    analysis_toc: Path,
    metadata_root: Path,
) -> dict[str, Any]:
    sbom_path = metadata_root / "sbom.cdx.json"
    manifest_path = metadata_root / "release-manifest.json"
    sbom = read_json(sbom_path)
    release_manifest = read_json(manifest_path)

    # Reject junctions and escaping symlinks before reading metadata from the
    # just-built payload. The final inventory is computed again after the
    # reconciled metadata is copied into place.
    payload_entries(payload_root)
    frozen_components_added = augment_sbom_with_frozen_metadata(
        sbom,
        payload_root,
        metadata_root,
    )
    included_python = analyzed_distributions(analysis_toc)
    included_python.update(frozen_metadata_distributions(payload_root))
    included_npm = bundled_node_packages(payload_root)
    (
        included_python_count,
        excluded_python_count,
        included_npm_count,
        excluded_npm_count,
    ) = reconcile_sbom(sbom, included_python, included_npm)
    write_json(sbom_path, sbom)

    release_manifest["payload_manifest"] = "payload-manifest.json"
    release_manifest["payload_inventory"] = {
        "coverage": "all frozen application files except the manifest itself",
        "python_components_included": included_python_count,
        "python_build_components_excluded": excluded_python_count,
        "frozen_metadata_components_added": frozen_components_added,
        "npm_components_included": included_npm_count,
        "npm_tree_shaken_components_excluded": excluded_npm_count,
        "sbom_reconciliation": (
            "verified from PyInstaller analysis, frozen metadata, and Vite source maps"
        ),
    }
    write_json(manifest_path, release_manifest)
    sync_release_metadata(metadata_root, payload_root)

    internal_manifest = payload_root / PAYLOAD_MANIFEST_RELATIVE
    if internal_manifest.exists():
        internal_manifest.unlink()
    entries = payload_entries(payload_root)
    inventory = {
        "schema_version": 1,
        "application": release_manifest["application"],
        "version": release_manifest["version"],
        "source": release_manifest["source"],
        "generated_at": release_manifest["generated_at"],
        "hash_algorithm": "sha256",
        "root": ".",
        "self_excluded": PAYLOAD_MANIFEST_RELATIVE.as_posix(),
        "file_count": len(entries),
        "total_bytes": sum(int(item["size"]) for item in entries),
        "python_distributions": [
            {"name": name, "version": version}
            for name, version in sorted(included_python)
            if name != "lm-atelier-api"
        ],
        "npm_packages": sorted(included_npm),
        "files": entries,
    }
    write_json(metadata_root / "payload-manifest.json", inventory)
    write_json(internal_manifest, inventory)
    return inventory


def verify_inventory(
    payload_root: Path,
    *,
    installer_extras: str | None = None,
) -> dict[str, Any]:
    manifest_path = payload_root / PAYLOAD_MANIFEST_RELATIVE
    inventory = read_json(manifest_path)
    expected = inventory.get("files")
    if not isinstance(expected, list):
        raise TypeError("Payload manifest files must be a list")
    actual = payload_entries(payload_root, installer_extras=installer_extras)
    if actual != expected:
        expected_by_path = {str(item.get("path")): item for item in expected}
        actual_by_path = {str(item.get("path")): item for item in actual}
        missing = sorted(set(expected_by_path) - set(actual_by_path))
        unexpected = sorted(set(actual_by_path) - set(expected_by_path))
        changed = sorted(
            path
            for path in set(expected_by_path) & set(actual_by_path)
            if expected_by_path[path] != actual_by_path[path]
        )
        raise RuntimeError(
            "Frozen payload does not match its manifest: "
            f"missing={missing}, unexpected={unexpected}, changed={changed}"
        )
    if inventory.get("file_count") != len(actual):
        raise RuntimeError("Payload manifest file_count is inconsistent")
    if inventory.get("total_bytes") != sum(int(item["size"]) for item in actual):
        raise RuntimeError("Payload manifest total_bytes is inconsistent")
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create or verify the exact frozen-application payload inventory."
    )
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--analysis-toc", type=Path)
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=ROOT / "build" / "release-metadata",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--installer-extras",
        choices=("linux", "windows"),
        help="Allow only the named installer's known root-level management files.",
    )
    args = parser.parse_args()

    payload_root = args.payload_root.resolve()
    if not payload_root.is_dir():
        raise SystemExit(f"Frozen payload does not exist: {payload_root}")

    if args.verify_only:
        inventory = verify_inventory(
            payload_root,
            installer_extras=args.installer_extras,
        )
        print(
            f"Verified {inventory['file_count']} frozen payload files for "
            f"LM Atelier {inventory['version']}"
        )
        return 0

    if args.installer_extras:
        parser.error("--installer-extras is valid only with --verify-only")
    payload_root = ensure_within(payload_root, BUILD_ROOT, "Payload root")
    if args.analysis_toc is None:
        parser.error("--analysis-toc is required when creating an inventory")
    analysis_toc = ensure_within(
        args.analysis_toc.resolve(),
        BUILD_ROOT,
        "PyInstaller analysis TOC",
    )
    if not analysis_toc.is_file():
        raise SystemExit(f"PyInstaller analysis TOC does not exist: {analysis_toc}")
    metadata_root = ensure_within(
        args.metadata_root.resolve(),
        BUILD_ROOT,
        "Release metadata root",
    )
    inventory = create_inventory(payload_root, analysis_toc, metadata_root)
    print(
        f"Inventoried {inventory['file_count']} frozen payload files for "
        f"LM Atelier {inventory['version']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
