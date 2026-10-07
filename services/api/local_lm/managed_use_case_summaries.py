"""Keep description suggestions on the current managed chat worker."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from sqlalchemy.orm import Session

from .models import ModelInstall, ModelProfile
from .use_case_summaries import UseCaseSummaryError, suggest_use_case_summary

if TYPE_CHECKING:
    from .main import Services

MANAGED_SUMMARY_TIMEOUT_SECONDS = 20.0


async def suggest_managed_use_case_summary(
    services: Services, session: Session, description: str
) -> str:
    """Check the loaded model while holding its lease, then return an unsaved suggestion."""
    try:
        async with asyncio.timeout(MANAGED_SUMMARY_TIMEOUT_SECONDS):
            async with services.scheduler.lease("primary"):
                session.expire_all()
                workers = [item for item in services.processes.statuses() if item.name == "chat"]
                if len(workers) != 1:
                    raise UseCaseSummaryError
                worker = workers[0]
                if (
                    not worker.managed
                    or not worker.running
                    or worker.state != "ready"
                    or not worker.pid
                    or worker.pid <= 0
                    or not worker.profile_id
                ):
                    raise UseCaseSummaryError
                profile = session.get(ModelProfile, worker.profile_id)
                if (
                    profile is None
                    or profile.role != "chat"
                    or profile.engine not in {"llama.cpp", "vllm"}
                    or profile.engine != services.settings.chat_engine
                    or not profile.model_install_id
                ):
                    raise UseCaseSummaryError
                install = session.get(ModelInstall, profile.model_install_id)
                if (
                    install is None
                    or not install.active
                    or install.role != "chat"
                    or install.engine != profile.engine
                ):
                    raise UseCaseSummaryError
                try:
                    origin = services.settings.validate_worker_url(services.settings.llama_url)
                    target = urlsplit(origin)
                    for flag, expected in (
                        ("--host", target.hostname),
                        ("--port", str(target.port or 80)),
                    ):
                        if worker.command.count(flag) != 1:
                            raise ValueError
                        offset = worker.command.index(flag) + 1
                        if offset >= len(worker.command) or worker.command[offset] != expected:
                            raise ValueError
                except (TypeError, ValueError):
                    raise UseCaseSummaryError from None
                return await suggest_use_case_summary(origin, description)
    except TimeoutError:
        raise UseCaseSummaryError from None
