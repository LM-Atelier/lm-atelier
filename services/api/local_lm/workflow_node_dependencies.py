"""Resolve the node packages a workflow revision says it needs.

Two subsystems install ComfyUI nodes and only one of them was ever reachable
from a run. `CustomNodeInstall` holds a package cloned from git; a
`ComfyRegistryInstall` holds one prepared from the ComfyUI Registry, with an
archive hash, a reviewed node inventory and a trust decision. The run path
resolved the first and had never heard of the second, so a revision recording a
Registry package would have been refused forever - which is why the workflows
imported so far declare no dependencies at all and run on whatever the shared
runtime happens to have loaded.

The two are kept in separate dependency keys rather than merged. They are
different things: a git install is identified by a URL and a revision, a
Registry package by an identifier and a version, and only the second carries a
reviewed inventory. Overloading one key would make the identity of an installed
package depend on which subsystem happened to install it.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import ComfyRegistryInstall, CustomNodeInstall

CUSTOM_NODE_KEY = "custom_nodes"
REGISTRY_PACKAGE_KEY = "registry_packages"


def node_dependency_errors(session: Session, dependencies: object) -> list[str]:
    """Report why a revision's declared node packages are not ready, if they are not.

    `dependencies` is the whole `dependencies_json`, so both keys are read from
    one place and a revision that declares neither costs nothing.
    """

    if not isinstance(dependencies, dict):
        return []
    errors = custom_node_dependency_errors(session, dependencies.get(CUSTOM_NODE_KEY))
    errors.extend(
        registry_package_dependency_errors(session, dependencies.get(REGISTRY_PACKAGE_KEY))
    )
    return errors


def custom_node_dependency_errors(session: Session, dependencies: object) -> list[str]:
    if not isinstance(dependencies, list):
        return [] if dependencies in (None, []) else ["custom node dependencies must be a list"]
    installs = session.scalars(
        select(CustomNodeInstall).where(CustomNodeInstall.active.is_(True))
    ).all()
    errors: list[str] = []
    for dependency in dependencies:
        revision: str | None = None
        if isinstance(dependency, dict):
            identifier = (
                dependency.get("id") or dependency.get("name") or dependency.get("source_url")
            )
            revision_value = dependency.get("revision")
            revision = str(revision_value) if revision_value else None
        else:
            identifier = dependency
        match = next(
            (
                install
                for install in installs
                if str(identifier) in {install.id, install.name, install.source_url}
            ),
            None,
        )
        if not match:
            errors.append(f"missing custom node dependency: {identifier}")
        elif not match.trusted:
            errors.append(f"custom node dependency is not trusted: {match.name}")
        elif revision and revision.lower() != match.revision.lower():
            errors.append(
                f"custom node revision mismatch for {match.name}: expected {revision}, "
                f"found {match.revision}"
            )
    return errors


def registry_package_dependency_errors(session: Session, dependencies: object) -> list[str]:
    """Report why declared Registry packages are not ready to run.

    A package answers only when it is both trusted and active. Trust is the
    review decision; active is whether this runtime is currently carrying it,
    and a re-provision can clear the second without touching the first.
    """

    if not isinstance(dependencies, list):
        return (
            [] if dependencies in (None, []) else ["registry package dependencies must be a list"]
        )
    installs = session.scalars(select(ComfyRegistryInstall)).all()
    errors: list[str] = []
    for dependency in dependencies:
        if not isinstance(dependency, dict) or not dependency.get("package_id"):
            errors.append(f"malformed registry package dependency: {dependency!r:.60}")
            continue
        package_id = str(dependency["package_id"])
        version = dependency.get("package_version")
        candidates = [install for install in installs if install.package_id == package_id]
        if version:
            candidates = [
                install for install in candidates if install.package_version == str(version)
            ]
        if not candidates:
            named = f"{package_id} {version}" if version else package_id
            errors.append(f"missing registry package dependency: {named}")
            continue
        if not any(install.trusted for install in candidates):
            errors.append(f"registry package dependency is not trusted: {package_id}")
        elif not any(install.trusted and install.active for install in candidates):
            errors.append(
                f"registry package dependency is not active in this runtime: {package_id}"
            )
    return errors
