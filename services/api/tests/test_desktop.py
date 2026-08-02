from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest
from httpx2 import ASGITransport, AsyncClient

from local_lm import desktop
from local_lm.config import Settings
from local_lm.downloads import download_worker_command
from local_lm.instance_identity import INSTANCE_ID_HEADER, InstanceIdentityError
from local_lm.main import create_app
from local_lm.runtime_config import configure_persisted_runtime, runtime_config_path


def test_default_data_dir_uses_windows_local_app_data(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Tester\AppData\Local")

    expected = Path(r"C:\Users\Tester\AppData\Local") / "LMAtelier" / "data"
    assert desktop.default_data_dir() == expected


def test_default_data_dir_uses_xdg_data_home(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", "/tmp/xdg-data")

    assert desktop.default_data_dir() == Path("/tmp/xdg-data/lm-atelier")


def test_download_worker_uses_frozen_executable_dispatch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/opt/lm-atelier/lm-atelier")

    assert download_worker_command() == ["/opt/lm-atelier/lm-atelier", "--download-worker"]


def test_first_run_setup_flag_lands_the_browser_in_setup() -> None:
    """The installer's hand-off: the flag becomes a query the web app reads."""

    base = "http://127.0.0.1:12310"
    assert desktop.launch_url(base, ["--first-run-setup"]) == f"{base}/?firstRunSetup=1"
    assert desktop.launch_url(base, []) == base
    # An unrelated flag must not trigger setup.
    assert desktop.launch_url(base, ["--download-worker"]) == base


def test_desktop_console_script_uses_persistence_aware_launcher() -> None:
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert 'lm-atelier = "local_lm.desktop:main"' in project.read_text(encoding="utf-8")


def test_runtime_configuration_survives_desktop_relaunch(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    first_launch = {
        "LOCAL_LM_CHAT_ENGINE": "llama.cpp",
        "LOCAL_LM_LLAMA_EXECUTABLE": str(tmp_path / "llama-server"),
        "LOCAL_LM_MEDIA_ENGINE": "comfyui",
        "LOCAL_LM_COMFY_EXECUTABLE": str(tmp_path / "python"),
        "LOCAL_LM_COMFY_DIRECTORY": str(tmp_path / "ComfyUI"),
    }

    configure_persisted_runtime(data_dir, first_launch)
    second_launch: dict[str, str] = {}
    configure_persisted_runtime(data_dir, second_launch)

    assert second_launch == first_launch
    payload = runtime_config_path(data_dir).read_text(encoding="utf-8")
    assert "TOKEN" not in payload


def test_saved_startup_limit_reaches_settings_on_the_next_launch(
    monkeypatch,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    data_dir = tmp_path / "data"
    configure_persisted_runtime(data_dir, {"LOCAL_LM_WORKER_STARTUP_SECONDS": "240"})

    relaunch: dict[str, str] = {}
    configure_persisted_runtime(data_dir, relaunch)
    assert relaunch == {"LOCAL_LM_WORKER_STARTUP_SECONDS": "240"}

    # At launch the restored environment is what Settings reads.
    monkeypatch.setenv(
        "LOCAL_LM_WORKER_STARTUP_SECONDS", relaunch["LOCAL_LM_WORKER_STARTUP_SECONDS"]
    )
    assert Settings(data_dir=data_dir).worker_startup_seconds == 240


def test_explicit_runtime_configuration_overrides_saved_value(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    configure_persisted_runtime(
        data_dir,
        {"LOCAL_LM_LLAMA_EXECUTABLE": str(tmp_path / "old-llama-server")},
    )
    environment = {"LOCAL_LM_LLAMA_EXECUTABLE": str(tmp_path / "new-llama-server")}

    configure_persisted_runtime(data_dir, environment)

    assert environment["LOCAL_LM_LLAMA_EXECUTABLE"].endswith("new-llama-server")
    assert "new-llama-server" in runtime_config_path(data_dir).read_text(encoding="utf-8")


def test_desktop_defaults_to_real_engines_without_hidden_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    data_dir = tmp_path / "data"
    monkeypatch.setenv("LOCAL_LM_DATA_DIR", str(data_dir))
    monkeypatch.delenv("LOCAL_LM_CHAT_ENGINE", raising=False)
    monkeypatch.delenv("LOCAL_LM_MEDIA_ENGINE", raising=False)
    monkeypatch.setenv("LOCAL_LM_OPEN_BROWSER", "false")

    desktop.configure_desktop_environment()

    assert desktop.os.environ["LOCAL_LM_CHAT_ENGINE"] == "llama.cpp"
    assert desktop.os.environ["LOCAL_LM_MEDIA_ENGINE"] == "comfyui"
    saved = runtime_config_path(data_dir).read_text(encoding="utf-8")
    assert '"LOCAL_LM_CHAT_ENGINE": "llama.cpp"' in saved
    assert '"LOCAL_LM_MEDIA_ENGINE": "comfyui"' in saved


def test_explicit_mock_engine_remains_available_for_development(
    monkeypatch,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LOCAL_LM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LOCAL_LM_CHAT_ENGINE", "mock")
    monkeypatch.setenv("LOCAL_LM_MEDIA_ENGINE", "mock")

    desktop.configure_desktop_environment()

    assert desktop.os.environ["LOCAL_LM_CHAT_ENGINE"] == "mock"
    assert desktop.os.environ["LOCAL_LM_MEDIA_ENGINE"] == "mock"


def test_source_launcher_honors_documented_environment_file(
    monkeypatch,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            (
                "LOCAL_LM_DATA_DIR=./developer-data",
                "LOCAL_LM_CHAT_ENGINE=mock",
                "LOCAL_LM_MEDIA_ENGINE=mock",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delenv("LOCAL_LM_DATA_DIR", raising=False)
    monkeypatch.delenv("LOCAL_LM_CHAT_ENGINE", raising=False)
    monkeypatch.delenv("LOCAL_LM_MEDIA_ENGINE", raising=False)

    desktop.configure_desktop_environment()

    assert desktop.os.environ["LOCAL_LM_DATA_DIR"] == "developer-data"
    assert desktop.os.environ["LOCAL_LM_CHAT_ENGINE"] == "mock"
    assert desktop.os.environ["LOCAL_LM_MEDIA_ENGINE"] == "mock"


def test_frozen_launcher_ignores_source_environment_file(
    monkeypatch,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "LOCAL_LM_CHAT_ENGINE=mock\nLOCAL_LM_MEDIA_ENGINE=mock\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("LOCAL_LM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("LOCAL_LM_CHAT_ENGINE", raising=False)
    monkeypatch.delenv("LOCAL_LM_MEDIA_ENGINE", raising=False)

    desktop.configure_desktop_environment()

    assert desktop.os.environ["LOCAL_LM_CHAT_ENGINE"] == "llama.cpp"
    assert desktop.os.environ["LOCAL_LM_MEDIA_ENGINE"] == "comfyui"


def test_desktop_health_probe_ignores_proxy_environment(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}
    expected_identity = "a" * 64

    class FakeOpener:
        def open(self, url: str, *, timeout: int) -> io.BytesIO:
            captured["url"] = url
            captured["timeout"] = timeout
            response = io.BytesIO(f'{{"version":"{desktop.__version__}"}}'.encode())
            response.headers = {INSTANCE_ID_HEADER: expected_identity}  # type: ignore[attr-defined]
            return response

    def build_opener(handler: object) -> FakeOpener:
        captured["proxies"] = getattr(handler, "proxies", None)
        return FakeOpener()

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")
    monkeypatch.setattr(desktop.urllib.request, "build_opener", build_opener)

    assert desktop._health_matches("http://127.0.0.1:12340", expected_identity)
    assert captured == {
        "proxies": {},
        "url": "http://127.0.0.1:12340/api/ready",
        "timeout": 2,
    }


def test_instance_identity_is_stable_opaque_and_bound_to_data_root(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = desktop.load_or_create_instance_identity(first_root)
    repeated = desktop.load_or_create_instance_identity(first_root)
    seed = (first_root / "state" / "desktop-instance-seed").read_text(encoding="ascii")
    (second_root / "state").mkdir(parents=True)
    (second_root / "state" / "desktop-instance-seed").write_text(seed, encoding="ascii")
    copied = desktop.load_or_create_instance_identity(second_root)

    assert repeated == first
    assert len(first) == 64
    assert first != seed
    assert copied != first


def test_invalid_instance_identity_is_not_silently_replaced(tmp_path: Path) -> None:
    seed = tmp_path / "data" / "state" / "desktop-instance-seed"
    seed.parent.mkdir(parents=True)
    seed.write_text("not-an-instance-seed", encoding="ascii")

    with pytest.raises(InstanceIdentityError, match="invalid"):
        desktop.load_or_create_instance_identity(tmp_path / "data")

    assert seed.read_text(encoding="ascii") == "not-an-instance-seed"


def test_existing_instance_seed_permissions_are_restricted(tmp_path: Path) -> None:
    seed = tmp_path / "data" / "state" / "desktop-instance-seed"
    seed.parent.mkdir(parents=True)
    seed.write_text("a" * 64, encoding="ascii")
    seed.chmod(0o666)

    identity = desktop.load_or_create_instance_identity(tmp_path / "data")

    assert len(identity) == 64
    if os.name != "nt":
        assert seed.stat().st_mode & 0o777 == 0o600


def test_instance_identity_rejects_linked_state_directory(tmp_path: Path) -> None:
    data = tmp_path / "data"
    outside = tmp_path / "outside-state"
    data.mkdir()
    outside.mkdir()
    try:
        (data / "state").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("filesystem links are unavailable in this test environment")

    with pytest.raises(InstanceIdentityError, match="state folder"):
        desktop.load_or_create_instance_identity(data)

    assert not (outside / "desktop-instance-seed").exists()


def test_instance_identity_rejects_non_file_seed(tmp_path: Path) -> None:
    seed = tmp_path / "data" / "state" / "desktop-instance-seed"
    seed.mkdir(parents=True)

    with pytest.raises(InstanceIdentityError, match="regular file"):
        desktop.load_or_create_instance_identity(tmp_path / "data")


@pytest.mark.parametrize("identity", [None, "b" * 64])
def test_same_version_service_without_expected_identity_is_a_conflict(
    monkeypatch,
    identity: str | None,
) -> None:  # type: ignore[no-untyped-def]
    class FakeOpener:
        def open(self, _url: str, *, timeout: int) -> io.BytesIO:
            assert timeout == 2
            response = io.BytesIO(f'{{"version":"{desktop.__version__}"}}'.encode())
            response.headers = (  # type: ignore[attr-defined]
                {} if identity is None else {INSTANCE_ID_HEADER: identity}
            )
            return response

    monkeypatch.setattr(
        desktop.urllib.request,
        "build_opener",
        lambda _handler: FakeOpener(),
    )

    probe = desktop._probe_instance("http://127.0.0.1:12340", "a" * 64)

    assert probe.state == "conflict"
    assert "different data folder" in probe.reason


def test_unresponsive_occupied_port_is_reported_as_a_conflict(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class FakeOpener:
        def open(self, _url: str, *, timeout: int) -> io.BytesIO:
            assert timeout == 2
            raise desktop.urllib.error.URLError("timed out")

    monkeypatch.setattr(
        desktop.urllib.request,
        "build_opener",
        lambda _handler: FakeOpener(),
    )
    monkeypatch.setattr(desktop, "_port_accepts_connections", lambda _url: True)

    probe = desktop._probe_instance("http://127.0.0.1:12340", "a" * 64)

    assert probe.state == "conflict"
    assert "accepting connections" in probe.reason


def _configure_main_test(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Settings:
    from local_lm import config

    settings = Settings(
        data_dir=tmp_path / "data",
        dev=False,
        chat_engine="mock",
        media_engine="mock",
    )
    settings.prepare()
    monkeypatch.setattr(sys, "argv", ["lm-atelier"])
    monkeypatch.setattr(desktop, "configure_desktop_environment", lambda: None)
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(
        desktop,
        "load_or_create_instance_identity",
        lambda _data_dir: "a" * 64,
    )
    return settings


def test_desktop_reuses_only_the_owned_same_version_instance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_main_test(monkeypatch, tmp_path)
    opened: list[str] = []
    monkeypatch.setenv("LOCAL_LM_OPEN_BROWSER", "true")
    monkeypatch.setattr(
        desktop,
        "_probe_instance",
        lambda _url, _identity: desktop.InstanceProbe("owned"),
    )
    monkeypatch.setattr(desktop.webbrowser, "open", opened.append)

    assert desktop.main() == 0
    assert opened == ["http://127.0.0.1:12340"]


def test_desktop_reports_identity_conflict_without_starting_or_opening(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _configure_main_test(monkeypatch, tmp_path)
    monkeypatch.setenv("LOCAL_LM_OPEN_BROWSER", "true")
    monkeypatch.setattr(
        desktop,
        "_probe_instance",
        lambda _url, _identity: desktop.InstanceProbe(
            "conflict",
            "LM Atelier is using a different data folder",
        ),
    )
    monkeypatch.setattr(
        desktop.webbrowser,
        "open",
        lambda _url: pytest.fail("a conflicting service must never be opened"),
    )

    assert desktop.main() == 2
    assert "Close the other LM Atelier/service" in capsys.readouterr().err


def test_desktop_starts_when_the_owned_service_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from local_lm import main as main_module

    _configure_main_test(monkeypatch, tmp_path)
    calls: list[bool] = []
    monkeypatch.setenv("LOCAL_LM_OPEN_BROWSER", "false")
    monkeypatch.setattr(
        desktop,
        "_probe_instance",
        lambda _url, _identity: desktop.InstanceProbe("unavailable"),
    )
    monkeypatch.setattr(main_module, "run", lambda: calls.append(True))

    assert desktop.main() == 0
    assert calls == [True]


async def test_ready_probe_exposes_the_selected_data_root_identity(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        dev=True,
        chat_engine="mock",
        media_engine="mock",
    )
    app = create_app(settings)
    expected = desktop.load_or_create_instance_identity(settings.data_dir)

    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client,
    ):
        response = await client.get("/api/ready")

    assert response.status_code == 200
    assert response.json() == {"version": desktop.__version__}
    assert response.headers[INSTANCE_ID_HEADER] == expected
