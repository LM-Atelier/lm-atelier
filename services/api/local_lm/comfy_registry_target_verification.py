"""Bind explicit package verification to the interpreter that will load it."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from .artifacts import ArtifactStore
from .comfy_registry_installs import ComfyRegistryInstallError
from .comfy_registry_launch_verification import (
    VerifiedComfyRegistryLaunch,
    _verify_comfy_registry_launch,
)
from .comfy_registry_reviewed_inputs import ComfyRegistryReviewedInputContext
from .comfy_registry_runtime import canonical_comfy_registry_runtime_distributions
from .models import ComfyRegistryInstall


@dataclass(frozen=True)
class ComfyRegistryVerificationTarget:
    session_factory: Callable[[], Session]
    python_executable: Path
    custom_node_root: Path
    environment_root: Path
    source_store: ArtifactStore | None = None
    check_configuration: Callable[[], None] | None = None

    def require_configuration(self) -> None:
        if self.check_configuration is not None:
            self.check_configuration()

    async def verify(
        self,
        install_ids: Sequence[str],
        *,
        include_active: bool = True,
        prepare_installs: Callable[[list[ComfyRegistryInstall]], None] | None = None,
    ) -> VerifiedComfyRegistryLaunch:
        """Verify explicit and active packages using fresh target and source evidence."""
        from .comfy_registry_interpreter import probe_comfy_registry_runtime_target

        self.require_configuration()
        identifiers = tuple(install_ids)

        def verify(
            reviewed: ComfyRegistryReviewedInputContext | None = None,
        ) -> VerifiedComfyRegistryLaunch:
            return _verify_comfy_registry_launch(
                self.session_factory,
                identifiers,
                include_active=include_active,
                custom_node_root=self.custom_node_root,
                environment_root=self.environment_root,
                reviewed_inputs=reviewed,
                prepare_installs=prepare_installs,
            )

        proof = None
        try:
            proof = await _verification_worker(verify)
        except ComfyRegistryInstallError as exc:
            if exc.code != "source_review_context_required" or self.source_store is None:
                raise
        if proof is not None and not proof.contract.runtime_distributions:
            self.require_configuration()
            return proof
        try:
            environment, tags, distributions = await probe_comfy_registry_runtime_target(
                self.python_executable
            )
            runtime = canonical_comfy_registry_runtime_distributions(distributions)
        except (ValueError, OSError) as exc:
            raise ComfyRegistryInstallError(
                "The managed media runtime package baseline could not be verified.",
                code="registry_runtime_verification_failed",
            ) from exc
        if proof is None:
            assert self.source_store is not None
            reviewed = ComfyRegistryReviewedInputContext(
                self.session_factory, self.source_store, environment, tags
            )
            proof = await _verification_worker(lambda: verify(reviewed))
        if proof.contract.runtime_distributions != runtime:
            raise ComfyRegistryInstallError(
                "The managed media runtime changed after its dependencies were prepared.",
                code="registry_runtime_changed",
            )
        self.require_configuration()
        return proof


async def _verification_worker[T](operation: Callable[[], T]) -> T:
    """Drain inspection failures without allowing them to replace cancellation."""
    from .comfy_registry_activation_batches import _worker

    def outcome() -> tuple[T] | Exception:
        try:
            return (operation(),)
        except Exception as exc:
            return exc

    result = await _worker(outcome)
    if isinstance(result, Exception):
        raise result
    return result[0]
