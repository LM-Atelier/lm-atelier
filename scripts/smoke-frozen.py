from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def request_json(url: str) -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return None
    return payload if isinstance(payload, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("application", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--port", type=int, default=12440)
    parser.add_argument(
        "--pre-manifest-release",
        action="store_true",
        help=(
            "Accept the reduced runtime self-test contract of releases that "
            "predate the payload manifest."
        ),
    )
    args = parser.parse_args()

    application = args.application.resolve()
    if not application.is_file():
        raise SystemExit(f"Frozen application not found: {application}")
    health_url = f"http://127.0.0.1:{args.port}/api/health"
    if request_json(health_url) is not None:
        raise RuntimeError(f"Smoke-test port {args.port} is already serving LM Atelier")

    with tempfile.TemporaryDirectory(prefix="lm-atelier-frozen-smoke-") as data_dir:
        environment = os.environ.copy()
        environment.update(
            {
                "LOCAL_LM_DATA_DIR": data_dir,
                "LOCAL_LM_CHAT_ENGINE": "mock",
                "LOCAL_LM_MEDIA_ENGINE": "mock",
                "LOCAL_LM_OPEN_BROWSER": "false",
                "LOCAL_LM_PORT": str(args.port),
            }
        )
        process = subprocess.Popen([str(application)], env=environment)
        try:
            deadline = time.monotonic() + 45
            health: dict[str, object] | None = None
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Frozen application exited with code {process.returncode} before startup"
                    )
                health = request_json(health_url)
                if health is not None:
                    break
                time.sleep(0.25)
            if health is None:
                raise RuntimeError("Frozen application did not become healthy within 45 seconds")
            if health.get("version") != args.version:
                raise RuntimeError(f"Unexpected health response: {health}")

            document = ""
            for _attempt in range(3):
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.port}/", timeout=2
                    ) as response:
                        document = response.read().decode("utf-8")
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.25)
            if "LM Atelier" not in document:
                raise RuntimeError("Frozen application did not serve the bundled web interface")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            shutdown_deadline = time.monotonic() + 10
            while time.monotonic() < shutdown_deadline:
                if request_json(health_url) is None:
                    break
                time.sleep(0.25)
            else:
                raise RuntimeError(
                    f"Frozen application still responds on port {args.port} after shutdown"
                )
        worker = subprocess.run(
            [str(application), "--download-worker"],
            input=b"{}",
            capture_output=True,
            timeout=15,
            check=False,
            env=environment,
        )
        if worker.returncode == 0 or b"repo_id" not in worker.stderr:
            raise RuntimeError("Frozen download-worker dispatch did not run as expected")
        runtime_environment = environment.copy()
        runtime_environment["LOCAL_LM_DATA_DIR"] = str(Path(data_dir) / "default-runtime")
        for key in (
            "LOCAL_LM_CHAT_ENGINE",
            "LOCAL_LM_MEDIA_ENGINE",
            "LOCAL_LM_LLAMA_EXECUTABLE",
            "LOCAL_LM_LLAMA_URL",
            "LOCAL_LM_COMFY_EXECUTABLE",
            "LOCAL_LM_COMFY_DIRECTORY",
            "LOCAL_LM_COMFY_URL",
        ):
            runtime_environment.pop(key, None)
        runtime = subprocess.run(
            [str(application), "--runtime-self-test"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env=runtime_environment,
        )
        try:
            runtime_result = json.loads(runtime.stdout)
        except ValueError as exc:
            raise RuntimeError(
                f"Frozen runtime self-test returned invalid output: {runtime.stderr}"
            ) from exc
        engine_report_ok = args.pre_manifest_release or (
            runtime_result.get("chat_engine") == "llama.cpp"
            and runtime_result.get("media_engine") == "comfyui"
            and runtime_result.get("engine_manifest_available") is True
        )
        if (
            runtime.returncode != 0
            or runtime_result.get("version") != args.version
            or not engine_report_ok
        ):
            raise RuntimeError(f"Frozen runtime self-test failed: {runtime_result}")
    print(f"Frozen application smoke test passed: {application}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
