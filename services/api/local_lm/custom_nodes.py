from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from .config import Settings
from .domain import new_id
from .models import CustomNodeInstall
from .subprocess_env import git_subprocess_environment

_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
MAX_REVIEWED_CUSTOM_NODE_TYPES = 4_096
MAX_REVIEWED_CUSTOM_NODE_TYPE_LENGTH = 200


def reviewed_custom_node_types(value: object, *, required: bool = False) -> tuple[str, ...]:
    """Return the exact node inventory explicitly reviewed for one pinned source."""

    if not isinstance(value, dict):
        if required:
            raise ValueError("Custom node has no reviewed node type evidence")
        return ()
    raw = value.get("node_types")
    if raw is None and not required:
        return ()
    if (
        not isinstance(raw, list)
        or len(raw) > MAX_REVIEWED_CUSTOM_NODE_TYPES
        or (required and not raw)
    ):
        raise ValueError("Custom node has no reviewed node type evidence")
    result: list[str] = []
    folded: set[str] = set()
    for item in raw:
        if (
            not isinstance(item, str)
            or not item
            or len(item) > MAX_REVIEWED_CUSTOM_NODE_TYPE_LENGTH
            or any(character < " " or ord(character) == 127 for character in item)
            or item.casefold() in folded
        ):
            raise ValueError("Custom node has invalid reviewed node type evidence")
        folded.add(item.casefold())
        result.append(item)
    return tuple(sorted(result, key=lambda item: (item.casefold(), item)))


class CustomNodeManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.custom_node_dir.resolve()

    def normalize_source(self, value: str) -> str:
        return self._source(value)

    def normalize_revision(self, value: str) -> str:
        return self._commit(value)

    async def install(
        self, session: Session, *, name: str, source_url: str, revision: str
    ) -> CustomNodeInstall:
        source = self._source(source_url)
        commit = self._commit(revision)
        node_id = new_id("node")
        directory_name = f"lm-atelier-{node_id}"
        destination = self._destination(directory_name)
        staging = Path(tempfile.mkdtemp(prefix="node-install-", dir=self.root))
        try:
            await self._run(
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--",
                source,
                str(staging),
            )
            await self._run("git", "-C", str(staging), "checkout", "--detach", commit)
            resolved = await self._run("git", "-C", str(staging), "rev-parse", "HEAD")
            if resolved.lower() != commit:
                raise ValueError("the custom node checkout did not resolve to the pinned commit")
            tree_hash = await self._run("git", "-C", str(staging), "rev-parse", "HEAD^{tree}")
            security = self._inspect(staging, source)
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        install = CustomNodeInstall(
            id=node_id,
            name=name.strip(),
            source_url=source,
            revision=commit,
            installed_path=directory_name,
            tree_hash=tree_hash,
            trusted=False,
            active=True,
            security_json=security,
        )
        session.add(install)
        session.flush()
        return install

    async def update(self, install: CustomNodeInstall, revision: str) -> None:
        commit = self._commit(revision)
        destination = self._destination(install.installed_path, require_exists=True)
        previous = install.revision
        await self._run(
            "git", "-C", str(destination), "fetch", "--filter=blob:none", "origin", commit
        )
        try:
            await self._run("git", "-C", str(destination), "checkout", "--detach", commit)
            resolved = await self._run("git", "-C", str(destination), "rev-parse", "HEAD")
            if resolved.lower() != commit:
                raise ValueError("the custom node update did not resolve to the pinned commit")
        except Exception:
            await self._run("git", "-C", str(destination), "checkout", "--detach", previous)
            raise
        install.previous_revision = previous
        install.revision = commit
        install.tree_hash = await self._run(
            "git", "-C", str(destination), "rev-parse", "HEAD^{tree}"
        )
        install.security_json = self._inspect(destination, install.source_url)
        install.trusted = False

    async def rollback(self, install: CustomNodeInstall) -> None:
        if not install.previous_revision:
            raise ValueError("this custom node has no previous pinned revision")
        current = install.revision
        previous = install.previous_revision
        destination = self._destination(install.installed_path, require_exists=True)
        await self._run("git", "-C", str(destination), "checkout", "--detach", previous)
        install.revision = previous
        install.previous_revision = current
        install.tree_hash = await self._run(
            "git", "-C", str(destination), "rev-parse", "HEAD^{tree}"
        )
        install.security_json = self._inspect(destination, install.source_url)
        install.trusted = False

    async def verify(self, install: CustomNodeInstall) -> None:
        destination = self._destination(install.installed_path, require_exists=True)
        revision = await self._run("git", "-C", str(destination), "rev-parse", "HEAD")
        tree_hash = await self._run("git", "-C", str(destination), "rev-parse", "HEAD^{tree}")
        if revision.lower() != install.revision.lower() or tree_hash != install.tree_hash:
            raise ValueError("custom node files no longer match the recorded pinned revision")

    def remove(self, install: CustomNodeInstall) -> None:
        destination = self._destination(install.installed_path)
        shutil.rmtree(destination, ignore_errors=True)

    def _destination(self, name: str, *, require_exists: bool = False) -> Path:
        if Path(name).name != name or not name.startswith("lm-atelier-node_"):
            raise ValueError("invalid managed custom node path")
        destination = (self.root / name).resolve()
        if destination.parent != self.root:
            raise ValueError("custom node path escapes the managed directory")
        if require_exists and not destination.is_dir():
            raise FileNotFoundError("managed custom node directory is missing")
        return destination

    @staticmethod
    def _source(value: str) -> str:
        parsed = urlparse(value.strip())
        parts = [part for part in parsed.path.split("/") if part]
        if (
            parsed.scheme != "https"
            or parsed.hostname != "github.com"
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.query
            or parsed.fragment
            or len(parts) != 2
        ):
            raise ValueError("custom node sources must be canonical HTTPS GitHub repositories")
        repository = parts[1][:-4] if parts[1].endswith(".git") else parts[1]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", parts[0]) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+", repository
        ):
            raise ValueError("invalid GitHub repository path")
        return f"https://github.com/{parts[0]}/{repository}.git"

    @staticmethod
    def _commit(value: str) -> str:
        if not _COMMIT.fullmatch(value.strip()):
            raise ValueError("custom nodes require a full 40-character commit SHA")
        return value.strip().lower()

    @staticmethod
    async def _run(*command: str, timeout: float = 180) -> str:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=git_subprocess_environment(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError as exc:
            await CustomNodeManager._terminate(process)
            raise RuntimeError("custom node Git operation timed out") from exc
        except asyncio.CancelledError:
            await CustomNodeManager._terminate(process)
            raise
        if process.returncode:
            detail = stderr.decode("utf-8", errors="replace").strip()[-1000:]
            raise RuntimeError(detail or "custom node Git operation failed")
        return stdout.decode("utf-8", errors="replace").strip()

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
        await process.wait()

    @staticmethod
    def _inspect(path: Path, source_url: str) -> dict[str, object]:
        files = [item for item in path.rglob("*") if item.is_file() and ".git" not in item.parts]
        manifest = "\n".join(
            f"{item.relative_to(path).as_posix()}:{item.stat().st_size}" for item in sorted(files)
        )
        return {
            "source_host": "github.com",
            "pinned_commit": True,
            "file_count": len(files),
            "python_file_count": sum(item.suffix == ".py" for item in files),
            "has_install_script": any(
                item.name.lower() in {"install.py", "setup.py", "requirements.txt"}
                for item in files
            ),
            "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
            "source_url": source_url,
            "review_required": True,
        }
