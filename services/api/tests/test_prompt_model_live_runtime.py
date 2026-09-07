"""Opt-in acceptance against an explicitly provisioned local CPU model fixture."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from local_lm.adapters.llama_cpp import LlamaCppAdapter
from local_lm.prompt_model_invocation import (
    PromptModelInvocationData,
    PromptModelInvocationItem,
    invoke_prompt_model_values,
)
from local_lm.prompt_model_values import prompt_model_slot_contract
from local_lm.prompt_templates import parse_prompt_template_contract

pytestmark = pytest.mark.skipif(
    not os.environ.get("MODEL_SLOT_RUNTIME_RECEIPT"),
    reason="requires an explicitly provisioned disposable local runtime",
)


async def test_local_model_fills_complete_mixed_scope_batches() -> None:
    receipt = json.loads(Path(os.environ["MODEL_SLOT_RUNTIME_RECEIPT"]).read_text(encoding="utf-8"))
    template = parse_prompt_template_contract(
        {
            "schema_version": 1,
            "operation": "text_to_image",
            "body": "A {{color}} {{subject}} with {{texture}} texture.",
            "slots": [
                {
                    "name": name,
                    "mode": "model",
                    "variation_scope": scope,
                    "guidance": guidance,
                }
                for name, scope, guidance in (
                    ("color", "batch", "one ordinary color word"),
                    ("subject", "item", "a different everyday object, one or two words"),
                    ("texture", "item", "one ordinary texture word"),
                )
            ],
            "resource_policy": {"mode": "inherited"},
        }
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    process = subprocess.Popen(
        [
            receipt["executable"],
            "--model",
            receipt["model"],
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--jinja",
            "--ctx-size",
            "8192",
            "--n-gpu-layers",
            "0",
            "--threads",
            "4",
            "--log-disable",
        ],
        cwd=receipt["data_dir"],
        env={
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("LLAMA_", "LOCAL_LM_"))
            and key not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    adapter = LlamaCppAdapter(f"http://127.0.0.1:{port}")
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=1) as client:
            async with asyncio.timeout(30):
                while True:
                    assert process.poll() is None, "the disposable runtime exited during startup"
                    try:
                        health = await client.get(f"http://127.0.0.1:{port}/health")
                        if health.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(0.1)
        for count in (1, 8, 16):
            contract = prompt_model_slot_contract(template, item_count=count)
            data = PromptModelInvocationData(
                template_text=template.body,
                batch_values=(),
                items=tuple(
                    PromptModelInvocationItem(ordinal=i, values=()) for i in range(1, count + 1)
                ),
            )
            started = time.monotonic()
            result = await invoke_prompt_model_values(adapter, contract=contract, data=data)
            assert len(result.values.items) == count
            assert len(result.values.batch_values) == 1
            assert all(len(item.values) == 2 for item in result.values.items)
            assert len(result.attempts) == 1
            print(
                json.dumps(
                    {
                        "requested_items": count,
                        "accepted_items": len(result.values.items),
                        "attempts": len(result.attempts),
                        "events": result.attempts[0].event_count,
                        "argument_fragments": result.attempts[0].argument_fragment_count,
                        "aggregate_bytes": result.attempts[0].aggregate_bytes,
                        "seconds": round(time.monotonic() - started, 3),
                    }
                ),
                flush=True,
            )
    finally:
        await adapter.close()
        if process.poll() is None:
            process.terminate()
        await asyncio.to_thread(process.wait, timeout=15)
