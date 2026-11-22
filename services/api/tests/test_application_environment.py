"""A test leaves the application's settings in the environment as it found them.

Whatever wrote them is put right afterwards, so no test runs with a setting an
earlier test in the same process left behind.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from application_environment import application_environment_kept

from local_lm import desktop

SETTING = "LOCAL_LM_CONSTRUCTED_SETTING"
OTHER_SETTING = "LOCAL_LM_OTHER_CONSTRUCTED_SETTING"
LEFT_BEHIND = "LOCAL_LM_CONSTRUCTED_LEFT_BEHIND"
LAUNCHER_SETTINGS = ("LOCAL_LM_CHAT_ENGINE", "LOCAL_LM_MEDIA_ENGINE", "LOCAL_LM_DEV")


def test_what_the_launcher_configures_is_gone_afterwards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOCAL_LM_DATA_DIR", str(tmp_path / "data"))
    for name in LAUNCHER_SETTINGS:
        # Absent, so clearing it records nothing for monkeypatch to undo.
        monkeypatch.delenv(name, raising=False)
    # The whole environment, not just the part the restore looks after, so a
    # name the restore failed to cover still shows up as left behind.
    before = dict(os.environ)

    with application_environment_kept():
        desktop.configure_desktop_environment()
        assert os.environ["LOCAL_LM_CHAT_ENGINE"] == "llama.cpp"
        assert os.environ["LOCAL_LM_MEDIA_ENGINE"] == "comfyui"
        assert os.environ["LOCAL_LM_DEV"] == "false"

    for name in LAUNCHER_SETTINGS:
        assert name not in os.environ
    assert dict(os.environ) == before


def test_a_changed_or_removed_setting_is_put_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SETTING, "before")
    monkeypatch.setenv(OTHER_SETTING, "kept")

    with application_environment_kept():
        os.environ[SETTING] = "after"
        del os.environ[OTHER_SETTING]

    assert os.environ[SETTING] == "before"
    assert os.environ[OTHER_SETTING] == "kept"


def test_a_setting_written_in_lower_case_is_removed_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SETTING, raising=False)

    with application_environment_kept():
        os.environ[SETTING.lower()] = "written"

    assert SETTING.lower() not in os.environ
    assert SETTING not in os.environ


def test_names_the_application_does_not_read_are_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONSTRUCTED_UNRELATED_NAME", "before")

    with application_environment_kept():
        os.environ["CONSTRUCTED_UNRELATED_NAME"] = "after"

    assert os.environ["CONSTRUCTED_UNRELATED_NAME"] == "after"


def test_a_test_may_write_a_setting_and_leave_it() -> None:
    # Nothing here puts it back. The next test checks that the restore every
    # test runs under did.
    os.environ[LEFT_BEHIND] = "written"


def test_a_setting_an_earlier_test_left_is_gone(request: pytest.FixtureRequest) -> None:
    # Run after the test above, as the tests in one file run, this sees what
    # the restore between them did. Selected alone, it still sees the restore
    # in place.
    assert "application_environment_restored" in request.fixturenames
    assert LEFT_BEHIND not in os.environ
