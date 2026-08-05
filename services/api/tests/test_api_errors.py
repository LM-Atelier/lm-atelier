"""Typed API errors: additive codes, forbidden unknown fields, a ratchet."""

from __future__ import annotations

import re
from pathlib import Path

from httpx2 import AsyncClient

API_SOURCE = (Path(__file__).resolve().parents[1] / "local_lm" / "api.py").read_text(
    encoding="utf-8"
)

# Lower this every time a bare HTTPException is converted to api_error; it
# must never rise. The eslint test ceilings use the same one-way ratchet.
BARE_HTTP_EXCEPTIONS_CEILING = 175


async def test_a_typed_error_keeps_detail_and_adds_a_stable_code(
    client: AsyncClient,
) -> None:
    response = await client.post("/api/workers/other/stop")
    assert response.status_code == 422
    body = response.json()
    # Existing clients keep reading detail; new clients branch on code.
    assert body["detail"] == "worker must be chat or media"
    assert body["code"] == "worker-unknown"


async def test_a_busy_worker_error_carries_its_job_count(client: AsyncClient) -> None:
    from local_lm.db import SessionLocal
    from local_lm.models import Job

    with SessionLocal() as session:
        session.add(Job(kind="chat", status="queued", phase="queued", payload_json={}))
        session.commit()

    response = await client.post("/api/workers/chat/stop")
    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "worker-busy"
    assert body["busy_jobs"] == 1
    assert "cancel or wait" in body["detail"]


async def test_an_unknown_request_field_is_refused_not_defaulted(
    client: AsyncClient,
) -> None:
    """A client typo must be a 422, not a silently applied default."""

    response = await client.put(
        "/api/workers/settings",
        json={"worker_startup_seconds": 90, "worker_startup_second": 90},
    )
    assert response.status_code == 422


def test_bare_http_exceptions_only_ever_decrease() -> None:
    bare = len(re.findall(r"HTTPException\(", API_SOURCE))
    assert bare <= BARE_HTTP_EXCEPTIONS_CEILING, (
        f"api.py has {bare} bare HTTPException raises, above the recorded "
        f"ceiling of {BARE_HTTP_EXCEPTIONS_CEILING}. New errors must use "
        "api_error with a stable code."
    )


def test_error_codes_are_kebab_case_slugs() -> None:
    # The same code may recur when call sites share one condition (all four
    # worker endpoints refuse an unknown name with "worker-unknown"); the
    # format contract is what must hold everywhere.
    codes = re.findall(r"api_error\(\s*\d+,\s*\"([^\"]+)\"", API_SOURCE)
    assert codes, "expected at least one typed api_error in api.py"
    for code in codes:
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", code), code
