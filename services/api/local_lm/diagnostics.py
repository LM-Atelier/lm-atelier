from __future__ import annotations

import json
import platform
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from . import __version__
from .artifacts import ArtifactStore
from .config import Settings
from .domain import ArtifactKind, utcnow
from .hardware import collect_system_info
from .models import Artifact, Chat, Job, ModelInstall, Project, Run, WorkflowDefinition
from .processes import ProcessSupervisor


class DiagnosticBundleBuilder:
    def __init__(
        self,
        settings: Settings,
        artifacts: ArtifactStore,
        processes: ProcessSupervisor,
    ) -> None:
        self.settings = settings
        self.artifacts = artifacts
        self.processes = processes

    def create(self, session: Session) -> Artifact:
        model_rows = session.execute(select(ModelInstall.role, ModelInstall.engine)).all()
        workflow_engines = Counter(session.scalars(select(WorkflowDefinition.operation)).all())
        job_statuses = Counter(session.scalars(select(Job.status)).all())
        log_files = [path for path in self.settings.log_dir.iterdir() if path.is_file()]
        payload = {
            "format": "lm-atelier-diagnostics",
            "version": 1,
            "created_at": utcnow().isoformat(),
            "application": {
                "version": __version__,
                "python": platform.python_version(),
                "development_mode": self.settings.dev,
            },
            "system": collect_system_info(self.settings).model_dump(mode="json"),
            "configuration": {
                "host_scope": "loopback",
                "chat_engine": self.settings.chat_engine,
                "media_engine": self.settings.media_engine,
                "auto_unload_chat_for_media": self.settings.auto_unload_chat_for_media,
                "comfy_inactivity_seconds": self.settings.comfy_inactivity_seconds,
                "artifact_retention_days": self.settings.artifact_retention_days,
                "temporary_retention_hours": self.settings.temporary_retention_hours,
                "backup_daily_count": self.settings.backup_daily_count,
                "backup_weekly_count": self.settings.backup_weekly_count,
            },
            "database": {
                "integrity": session.execute(text("PRAGMA integrity_check")).scalar_one(),
                "projects": session.scalar(select(func.count(Project.id))) or 0,
                "chats": session.scalar(select(func.count(Chat.id))) or 0,
                "runs": session.scalar(select(func.count(Run.id))) or 0,
                "artifacts": session.scalar(select(func.count(Artifact.id))) or 0,
            },
            "models": {
                "count": len(model_rows),
                "role_engine_counts": dict(
                    Counter(f"{role}:{engine}" for role, engine in model_rows)
                ),
            },
            "workflows": {
                "count": sum(workflow_engines.values()),
                "operation_counts": dict(workflow_engines),
            },
            "jobs": {"status_counts": dict(job_statuses)},
            # statuses() sanitizes before it returns: commands and tails have
            # local paths replaced, so nothing here weakens the privacy notes.
            "workers": [
                {
                    "name": status.name,
                    "state": status.state,
                    "managed": status.managed,
                    "running": status.running,
                    "exit_code": status.exit_code,
                    "failure_code": status.failure_code,
                    "failure_detail": status.failure_detail,
                    "startup_duration_ms": status.startup_duration_ms,
                    "current_memory_bytes": status.current_memory_bytes,
                    "peak_memory_bytes": status.peak_memory_bytes,
                    "stderr_tail": status.stderr_tail,
                }
                for status in self.processes.statuses()
            ],
            "logs": {
                "file_count": len(log_files),
                "total_bytes": sum(path.stat().st_size for path in log_files),
                "contents_included": False,
            },
            "privacy": {
                "prompts_included": False,
                "media_included": False,
                "tokens_included": False,
                "absolute_paths_included": False,
            },
        }
        with tempfile.NamedTemporaryFile(
            dir=self.settings.export_dir, suffix=".lm-atelier-diagnostics.zip", delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(
                    "diagnostics.json", json.dumps(payload, indent=2, ensure_ascii=False)
                )
                archive.writestr(
                    "README.txt",
                    "LM Atelier redacted diagnostics\n\n"
                    "This bundle excludes prompts, chat content, generated media, credentials, "
                    "log contents, and absolute filesystem paths.\n"
                    "Worker entries include the same redacted failure output the application "
                    "shows on screen, with local paths already replaced.\n",
                )
            return self.artifacts.ingest_path(
                session,
                temporary,
                kind=ArtifactKind.EXPORT,
                media_type="application/zip",
                original_name="lm-atelier-diagnostics.zip",
                metadata={"format": "lm-atelier-diagnostics", "version": 1},
            )
        finally:
            temporary.unlink(missing_ok=True)
