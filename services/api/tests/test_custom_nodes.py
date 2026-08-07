from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from local_lm.custom_nodes import CustomNodeManager
from local_lm.db import Base, SessionLocal
from local_lm.domain import new_id
from local_lm.models import CustomNodeInstall, Job


def test_custom_node_sources_and_revisions_are_strictly_pinned(settings) -> None:  # type: ignore[no-untyped-def]
    manager = CustomNodeManager(settings)
    assert (
        manager.normalize_source("https://github.com/example/comfy-node")
        == "https://github.com/example/comfy-node.git"
    )
    assert manager.normalize_revision("A" * 40) == "a" * 40
    for unsafe in (
        "http://github.com/example/node",
        "https://example.com/example/node",
        "https://github.com/example/node?ref=main",
        "file:///tmp/node",
    ):
        with pytest.raises(ValueError):
            manager.normalize_source(unsafe)
    with pytest.raises(ValueError):
        manager.normalize_revision("main")


async def test_custom_node_timeout_kills_and_reaps_git_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HangingProcess:
        returncode: int | None = None
        killed = False
        waited = False

        async def communicate(self):  # type: ignore[no-untyped-def]
            await asyncio.Event().wait()

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        async def wait(self) -> int:
            self.waited = True
            return self.returncode or 0

    process = HangingProcess()

    async def create_process(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    with pytest.raises(RuntimeError, match="timed out"):
        await CustomNodeManager._run("git", "--version", timeout=0.01)
    assert process.killed is True
    assert process.waited is True


async def test_custom_node_git_does_not_inherit_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class CompletedProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"git version", b""

    async def create_process(*_args: str, **kwargs: object) -> CompletedProcess:
        captured.update(kwargs)
        return CompletedProcess()

    monkeypatch.setenv("GITHUB_TOKEN", "github_private")
    monkeypatch.setenv("GH_TOKEN", "gh_private")
    monkeypatch.setenv("SSH_AUTH_SOCK", "private-agent")
    monkeypatch.setenv("GIT_ASKPASS", "credential-helper")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    assert await CustomNodeManager._run("git", "--version") == "git version"

    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"]
    assert environment["GIT_ASKPASS"] == ""
    assert environment["SSH_ASKPASS"] == ""
    assert "GITHUB_TOKEN" not in environment
    assert "GH_TOKEN" not in environment
    assert "SSH_AUTH_SOCK" not in environment


async def test_custom_node_lifecycle_and_workflow_trust_gate(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def install(
        _manager: CustomNodeManager,
        session: Session,
        *,
        name: str,
        source_url: str,
        revision: str,
    ) -> CustomNodeInstall:
        record = CustomNodeInstall(
            id=new_id("node"),
            name=name,
            source_url=source_url,
            revision=revision,
            installed_path="lm-atelier-node_test",
            tree_hash="b" * 40,
            trusted=False,
            active=True,
            security_json={"review_required": True, "pinned_commit": True},
        )
        session.add(record)
        session.flush()
        return record

    async def verify(_manager: CustomNodeManager, _install: CustomNodeInstall) -> None:
        return None

    async def update(_manager: CustomNodeManager, record: CustomNodeInstall, revision: str) -> None:
        record.previous_revision = record.revision
        record.revision = revision
        record.trusted = False

    async def rollback(_manager: CustomNodeManager, record: CustomNodeInstall) -> None:
        assert record.previous_revision
        record.revision, record.previous_revision = record.previous_revision, record.revision
        record.trusted = False

    monkeypatch.setattr(CustomNodeManager, "install", install)
    monkeypatch.setattr(CustomNodeManager, "verify", verify)
    monkeypatch.setattr(CustomNodeManager, "update", update)
    monkeypatch.setattr(CustomNodeManager, "rollback", rollback)
    monkeypatch.setattr(CustomNodeManager, "remove", lambda _manager, _install: None)

    revision = "a" * 40
    installed = await client.post(
        "/api/custom-nodes",
        json={
            "name": "Reviewed node",
            "source_url": "https://github.com/example/comfy-node",
            "revision": revision,
        },
    )
    assert installed.status_code == 201
    node = installed.json()
    assert node["trusted"] is False
    assert node["source_url"].endswith(".git")

    workflow = await client.post(
        "/api/workflows",
        json={
            "name": "Node workflow",
            "operation": "text_to_image",
            "api_graph": {"node": {"class_type": "ReviewedNode"}},
            "ui_graph": {"last_node_id": 1, "nodes": []},
            "dependencies": {"custom_nodes": [{"id": node["id"], "revision": revision}]},
            "trusted": True,
        },
    )
    validation = await client.post(f"/api/workflows/{workflow.json()['id']}/validate")
    assert "not trusted" in validation.json()["errors"][0]
    open_target = await client.get(f"/api/workflows/{workflow.json()['id']}/open-target")
    assert open_target.status_code == 200
    assert open_target.json()["url"] == "http://127.0.0.1:8188"
    assert open_target.json()["ui_graph"]["last_node_id"] == 1

    trusted = await client.post(f"/api/custom-nodes/{node['id']}/trust", json={"trusted": True})
    assert trusted.status_code == 200
    validation = await client.post(f"/api/workflows/{workflow.json()['id']}/validate")
    assert validation.json()["valid"] is True

    next_revision = "c" * 40
    updated = await client.patch(
        f"/api/custom-nodes/{node['id']}", json={"revision": next_revision}
    )
    assert updated.json()["trusted"] is False
    assert updated.json()["previous_revision"] == revision
    rolled_back = await client.post(f"/api/custom-nodes/{node['id']}/rollback")
    assert rolled_back.json()["revision"] == revision

    removed = await client.delete(f"/api/custom-nodes/{node['id']}")
    assert removed.status_code == 204
    assert (await client.get("/api/custom-nodes")).json() == []


async def test_custom_node_change_rechecks_media_queue_inside_compute_lease(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = False

    async def verify(_manager: CustomNodeManager, _install: CustomNodeInstall) -> None:
        nonlocal verified
        verified = True

    monkeypatch.setattr(CustomNodeManager, "verify", verify)
    with SessionLocal() as session:
        session.add(
            CustomNodeInstall(
                id="node_lease_race",
                name="Lease race",
                source_url="https://github.com/example/lease-race.git",
                revision="a" * 40,
                installed_path="lm-atelier-node_lease-race",
                tree_hash="b" * 40,
                trusted=False,
                active=True,
                security_json={"review_required": True},
            )
        )
        session.commit()

    async with app.state.services.scheduler.lease("primary"):
        trust = asyncio.create_task(
            client.post(
                "/api/custom-nodes/node_lease_race/trust",
                json={"trusted": True},
            )
        )
        await asyncio.sleep(0.03)
        assert trust.done() is False
        with SessionLocal() as session:
            session.add(
                Job(
                    id="job_node_lease_race",
                    kind="image",
                    status="queued",
                    phase="queued",
                )
            )
            session.commit()

    response = await asyncio.wait_for(trust, timeout=2)
    assert response.status_code == 409
    assert "active or queued job" in response.json()["detail"]
    assert verified is False
    with SessionLocal() as session:
        node = session.get(CustomNodeInstall, "node_lease_race")
        assert node
        assert node.trusted is False


def test_a_registry_package_dependency_resolves_against_registry_installs() -> None:
    """The run path could only see git installs, so a revision naming a Registry
    package would have been refused forever - which is why imported workflows
    declare nothing and run on whatever the shared runtime happens to carry."""

    from local_lm.models import ComfyRegistryInstall
    from local_lm.workflow_node_dependencies import node_dependency_errors

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            [
                ComfyRegistryInstall(
                    package_id="rgthree-comfy",
                    registry_record_id="rec-rgthree",
                    repository_url="https://github.com/example/rgthree-comfy",
                    download_url="https://example.invalid/rgthree-comfy.zip",
                    archive_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    manifest_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    package_version="1.2.3",
                    installed_path="lm-atelier-registry_aaa",
                    trusted=True,
                    active=True,
                    node_types_json=[],
                    pip_dependencies_json=[],
                    review_json={},
                ),
                ComfyRegistryInstall(
                    package_id="comfyui-impact-pack",
                    registry_record_id="rec-impact",
                    repository_url="https://github.com/example/comfyui-impact-pack",
                    download_url="https://example.invalid/comfyui-impact-pack.zip",
                    archive_sha256="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    manifest_sha256="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    package_version="8.0.0",
                    installed_path="lm-atelier-registry_bbb",
                    trusted=True,
                    active=False,
                    node_types_json=[],
                    pip_dependencies_json=[],
                    review_json={},
                ),
                ComfyRegistryInstall(
                    package_id="was-ns",
                    registry_record_id="rec-was",
                    repository_url="https://github.com/example/was-ns",
                    download_url="https://example.invalid/was-ns.zip",
                    archive_sha256="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                    manifest_sha256="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                    package_version="3.0.1",
                    installed_path="lm-atelier-registry_ccc",
                    trusted=False,
                    active=False,
                    node_types_json=[],
                    pip_dependencies_json=[],
                    review_json={},
                ),
            ]
        )
        session.commit()

        ready = {"registry_packages": [{"package_id": "rgthree-comfy", "package_version": "1.2.3"}]}
        assert node_dependency_errors(session, ready) == []

        # Trusted but not carried by this runtime - what a re-provision leaves.
        inactive = node_dependency_errors(
            session, {"registry_packages": [{"package_id": "comfyui-impact-pack"}]}
        )
        assert inactive == [
            "registry package dependency is not active in this runtime: comfyui-impact-pack"
        ]

        untrusted = node_dependency_errors(
            session, {"registry_packages": [{"package_id": "was-ns"}]}
        )
        assert untrusted == ["registry package dependency is not trusted: was-ns"]

        absent = node_dependency_errors(
            session, {"registry_packages": [{"package_id": "never-installed"}]}
        )
        assert absent == ["missing registry package dependency: never-installed"]

        wrong_version = node_dependency_errors(
            session,
            {"registry_packages": [{"package_id": "rgthree-comfy", "package_version": "9.9.9"}]},
        )
        assert wrong_version == ["missing registry package dependency: rgthree-comfy 9.9.9"]

        # A revision that declares neither key costs nothing, which is every
        # revision stored before this existed.
        assert node_dependency_errors(session, {}) == []
