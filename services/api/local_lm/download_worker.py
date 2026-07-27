from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from huggingface_hub import hf_hub_download

_MAX_DOWNLOAD_FILES = 512


def main() -> int:
    payload: dict[str, Any] = json.load(sys.stdin.buffer)
    raw_files = payload.get("files")
    if isinstance(raw_files, list):
        files = [str(item) for item in raw_files]
        if not files or len(files) > _MAX_DOWNLOAD_FILES:
            raise ValueError("download worker file batch is outside the supported bounds")
        workers = min(max(int(payload.get("max_workers") or 1), 1), 4, len(files))

        def download(filename: str) -> tuple[str, str]:
            return (
                filename,
                hf_hub_download(
                    repo_id=str(payload["repo_id"]),
                    filename=filename,
                    revision=str(payload["revision"]),
                    local_dir=str(payload["local_dir"]),
                    token=payload.get("token"),
                ),
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            paths = dict(executor.map(download, files))
        json.dump({"paths": paths}, sys.stdout)
        return 0
    path = hf_hub_download(
        repo_id=str(payload["repo_id"]),
        filename=str(payload["filename"]),
        revision=str(payload["revision"]),
        local_dir=str(payload["local_dir"]),
        token=payload.get("token"),
    )
    json.dump({"path": path}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
