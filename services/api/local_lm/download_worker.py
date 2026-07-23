from __future__ import annotations

import json
import sys
from typing import Any

from huggingface_hub import hf_hub_download


def main() -> int:
    payload: dict[str, Any] = json.load(sys.stdin.buffer)
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
