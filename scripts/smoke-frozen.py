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
    args = parser.parse_args()

    application = args.application.resolve()
    if not application.is_file():
        raise SystemExit(f"Frozen application not found: {application}")

    with tempfile.TemporaryDirectory(prefix="lm-atelier-frozen-smoke-") as data_dir:
        environment = os.environ.copy()
        environment.update(
            {
                "LOCAL_LM_DATA_DIR": data_dir,
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
                health = request_json(f"http://127.0.0.1:{args.port}/api/health")
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
    worker = subprocess.run(
        [str(application), "--download-worker"],
        input=b"{}",
        capture_output=True,
        timeout=15,
        check=False,
    )
    if worker.returncode == 0 or b"repo_id" not in worker.stderr:
        raise RuntimeError("Frozen download-worker dispatch did not run as expected")
    runtime = subprocess.run(
        [str(application), "--runtime-self-test"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    try:
        runtime_result = json.loads(runtime.stdout)
    except ValueError as exc:
        raise RuntimeError(
            f"Frozen runtime self-test returned invalid output: {runtime.stderr}"
        ) from exc
    if runtime.returncode != 0 or runtime_result.get("version") != args.version:
        raise RuntimeError(f"Frozen runtime self-test failed: {runtime_result}")
    print(f"Frozen application smoke test passed: {application}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
