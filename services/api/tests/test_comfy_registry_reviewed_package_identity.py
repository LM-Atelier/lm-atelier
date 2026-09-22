from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy.orm import Session
from test_comfy_registry_reviewed_wheel_staging import (
    source_review_context as source_review_context,
)
from test_comfy_registry_source_artifacts import DECLARATION
from test_workflow_reviewed_package_plan import _close, _inputs, _prepare

from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_activation import ComfyRegistryActivationError
from local_lm.comfy_registry_activation_batches import _row
from local_lm.models import ComfyRegistryInstall
from local_lm.workflow_offer_packages import _existing
from local_lm.workflow_package_execution_plan import plan_workflow_package_execution


@pytest.mark.parametrize("commit", [False, True])
@pytest.mark.parametrize("inactive", [False, True])
async def test_batch_and_offer_reuse_bind_the_full_reviewed_declaration_identity(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    commit: bool,
    inactive: bool,
) -> None:
    inputs = _inputs(source_review_context, tmp_path, commit=commit, inactive=inactive)
    try:
        plan = await plan_workflow_package_execution(**inputs["plan_arguments"])
        prepared = await _prepare(inputs, plan)
        with inputs["factory"]() as session:
            row = session.get(ComfyRegistryInstall, prepared.install_id)
            assert row is not None
            assert _row(session, prepared, plan) is row
            assert _existing(session, plan) == replace(prepared, reused_wheel_environment=True)
            assert not row.trusted and not row.active
            original = list(row.pip_dependencies_json)
            for changed in ([], ["alpha==1.0"], [DECLARATION + ' ; python_version >= "0"']):
                row.pip_dependencies_json = changed
                with pytest.raises(ComfyRegistryActivationError) as error:
                    _row(session, prepared, plan)
                assert error.value.code == "registry_batch_identity_changed"
                with pytest.raises(ComfyRegistryActivationError) as error:
                    _existing(session, plan)
                assert error.value.code == "registry_batch_identity_changed"
            row.pip_dependencies_json = original
            assert _row(session, prepared, plan) is row
            assert _existing(session, plan) == replace(prepared, reused_wheel_environment=True)
            assert not row.trusted and not row.active
    finally:
        await _close(inputs)
