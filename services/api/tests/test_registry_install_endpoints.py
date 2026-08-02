"""Trust and activation over HTTP: explicit decisions with stable refusals."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.config import Settings

pytestmark = pytest.mark.asyncio


def _seed_install(*, trusted: bool = False, active: bool = False) -> str:
    from local_lm.db import SessionLocal
    from local_lm.models import ComfyRegistryInstall

    with SessionLocal() as session:
        install = ComfyRegistryInstall(
            package_id="comfyui-example-node",
            package_version="1.2.3",
            registry_record_id="record-123",
            repository_url="https://github.com/example/comfyui-example-node.git",
            download_url="https://cdn.comfy.org/example/1.2.3.zip",
            archive_sha256="a" * 64,
            manifest_sha256="b" * 64,
            installed_path="lm-atelier-registry_example",
            node_types_json=["ExampleNode"],
            pip_dependencies_json=[],
            review_json={"review_required": True},
            wheel_closure_sha256="c" * 64,
            wheel_environment_sha256="d" * 64,
            wheel_environment_path=f"registry-wheels-{'c' * 64}",
            trusted=trusted,
            active=active,
        )
        session.add(install)
        session.commit()
        return install.id


def _configure_runtime(settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"")
    monkeypatch.setattr(settings, "comfy_executable", executable)
    monkeypatch.setattr(settings, "comfy_directory", tmp_path / "ComfyUI")


async def test_the_install_list_reports_both_pending_decisions(client: AsyncClient) -> None:
    install_id = _seed_install()

    response = await client.get("/api/workflows/packages/installs")

    assert response.status_code == 200
    (install,) = response.json()
    assert install["id"] == install_id
    assert install["package_id"] == "comfyui-example-node"
    assert install["package_version"] == "1.2.3"
    assert install["node_types"] == ["ExampleNode"]
    assert install["archive_sha256"] == "a" * 64
    assert install["trusted"] is False
    assert install["active"] is False
    assert install["reviewed_at"] is None
    assert install["activated_at"] is None


async def test_revoking_trust_needs_no_runtime_verification(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Revocation must always be possible; only granting trust verifies files."""

    _configure_runtime(settings, monkeypatch, tmp_path)
    install_id = _seed_install(trusted=True, active=True)

    response = await client.post(
        f"/api/workflows/packages/installs/{install_id}/review",
        json={"trusted": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["trusted"] is False
    assert body["active"] is False
    assert body["reviewed_at"]


async def test_granting_trust_runs_the_full_launch_verification(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _configure_runtime(settings, monkeypatch, tmp_path)
    install_id = _seed_install()

    import local_lm.comfy_registry_activation as activation_module
    from local_lm.comfy_registry_installs import ComfyRegistryLaunchContract

    def verified(session, **_kwargs: object) -> ComfyRegistryLaunchContract:
        from local_lm.models import ComfyRegistryInstall

        install = session.get(ComfyRegistryInstall, install_id)
        assert install is not None
        return ComfyRegistryLaunchContract((install.installed_path,), (), ("ExampleNode",))

    monkeypatch.setattr(activation_module, "trusted_comfy_registry_launch_contract", verified)

    response = await client.post(
        f"/api/workflows/packages/installs/{install_id}/review",
        json={"trusted": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["trusted"] is True
    # Trust never activates; that is a separate explicit decision.
    assert body["active"] is False
    assert body["reviewed_at"]


async def test_typed_refusals_keep_their_codes_and_statuses(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _configure_runtime(settings, monkeypatch, tmp_path)

    missing = await client.post(
        "/api/workflows/packages/installs/absent/review",
        json={"trusted": False},
    )
    assert missing.status_code == 404
    assert missing.json()["code"] == "registry_install_not_found"

    install_id = _seed_install(trusted=False)
    untrusted = await client.post(f"/api/workflows/packages/installs/{install_id}/activate")
    assert untrusted.status_code == 409
    assert untrusted.json()["code"] == "registry_install_untrusted"


async def test_a_running_media_worker_refuses_activation_changes(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """The stopped-worker requirement is told truthfully, not assumed."""

    _configure_runtime(settings, monkeypatch, tmp_path)
    install_id = _seed_install(trusted=True, active=True)
    services = app.state.services
    monkeypatch.setattr(
        services.processes,
        "statuses",
        lambda: [SimpleNamespace(name="media", running=True, state="running")],
    )

    response = await client.post(
        f"/api/workflows/packages/installs/{install_id}/review",
        json={"trusted": False},
    )

    assert response.status_code == 409
    assert response.json()["code"] == "media_worker_running"


async def test_an_unconfigured_runtime_refuses_review_before_touching_rows(
    client: AsyncClient,
) -> None:
    install_id = _seed_install()

    response = await client.post(
        f"/api/workflows/packages/installs/{install_id}/review",
        json={"trusted": True},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "managed_runtime_unavailable"


async def test_deactivation_restarts_the_runtime_without_the_package(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_id = _seed_install(trusted=True, active=True)
    services = app.state.services
    restarted = []

    async def fake_start_media(*args: object, **kwargs: object) -> object:
        restarted.append(True)
        return SimpleNamespace(name="media", running=True, state="running")

    monkeypatch.setattr(services.processes, "start_media", fake_start_media)

    response = await client.post(f"/api/workflows/packages/installs/{install_id}/deactivate")

    assert response.status_code == 200
    body = response.json()
    assert body["active"] is False
    # Trust survives deactivation; only revocation removes it.
    assert body["trusted"] is True
    assert restarted == [True]
