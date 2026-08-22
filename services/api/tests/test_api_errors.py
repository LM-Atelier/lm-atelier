"""Typed API errors: additive codes, forbidden unknown fields, a ratchet."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from httpx2 import AsyncClient

API_SOURCE = (Path(__file__).resolve().parents[1] / "local_lm" / "api.py").read_text(
    encoding="utf-8"
)

# Lower this every time a bare HTTPException is converted to api_error; it
# must never rise. The eslint test ceilings use the same one-way ratchet.
#
# It cannot reach zero, and the floor is not four raises' worth of laziness.
# Four sites already put a "code" in their detail dict alongside a plan or an
# estimate, so they are programmatically identifiable already - which is the
# thing this ratchet exists to get. Two of those codes,
# route_confirmation_required and ordered_plan_confirmation_required, are
# matched by name in apps/web/src/api.ts, so converting them to api_error's
# kebab case would break the composer's confirmation flow. Counting them is an
# artefact of measuring by regex; converting them would be a regression.
#
# The ratchet is now AT that floor. Every remaining raise is one of those four,
# so this number should not move again. A rise means a new untyped error was
# added; the fix is api_error with a code, never a higher ceiling.
#: Four raises remain and this is the floor, not a rung.
#:
#: They are not the defect the ratchet was built for. That defect was 204 raises
#: carrying a bare string, where the status code was the only thing a client
#: could act on. These four each carry a `code` already, plus the payload the
#: caller needs to resolve the condition - the plan and its estimate, or the
#: pin's project, revision, role and reason.
#:
#: They stay on `HTTPException` because their codes are snake_case and the
#: browser reads two of them literally: `api.ts:312` matches
#: `ordered_plan_confirmation_required` and `:338` matches
#: `route_confirmation_required`. Moving them to `api_error` would either
#: rename the codes and break those two call sites, or keep snake_case and
#: violate `test_error_codes_are_kebab_case_slugs` below.
#:
#: So driving this to zero is a client-visible rename, not a cleanup. If that is
#: wanted, change the browser and these together in one commit; do not convert
#: the raises alone and expect the suite to protect you, because it will not -
#: the browser's comparison is a string literal and nothing checks it against
#: this file.
BARE_HTTP_EXCEPTIONS_CEILING = 4


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


async def test_selecting_an_unknown_revision_says_which_refusal_it_was(
    client: AsyncClient,
) -> None:
    """Two 404s reach this route; only the code says which one happened."""

    response = await client.post("/api/messages/msg_missing/revisions/rev_missing/select")
    assert response.status_code == 404
    assert response.json()["code"] == "response-revision-not-found"


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


def test_no_code_means_two_different_things() -> None:
    """A code may recur; it may not recur at two different statuses.

    Reuse across call sites is deliberate - four worker endpoints refusing an
    unknown name all mean the same thing and share `worker-unknown`. But the
    same code at a 409 in one place and a 422 in another is two conditions
    wearing one name, and a client branching on it would take the wrong action
    for one of them. That is the failure typed codes exist to prevent, so
    reintroducing it through a copied slug should fail here rather than in
    somebody's error handling.
    """
    statuses: dict[str, set[str]] = {}
    for status, code in re.findall(r'api_error\(\s*(\d+),\s*"([^"]+)"', API_SOURCE):
        statuses.setdefault(code, set()).add(status)

    ambiguous = {code: sorted(seen) for code, seen in statuses.items() if len(seen) > 1}
    assert ambiguous == {}, f"codes used at more than one status: {ambiguous}"


def test_the_browser_matches_the_codes_these_raises_actually_send() -> None:
    """The four exempt raises are a contract with the browser, so check it.

    Their codes are snake_case and the browser compares them as string
    literals - `detail?.code === "route_confirmation_required"`. Nothing held
    the two ends together, so renaming a code here would leave the browser
    silently failing to recognise a confirmation it is supposed to act on: the
    request would look like an ordinary error and the user would never see the
    confirm dialog.

    Only the codes the browser actually matches are checked. The server may
    raise conditions the browser has no branch for; that is a missing feature
    rather than a broken contract, and asserting the reverse would fail every
    time somebody adds a refusal before its UI.
    """

    browser = Path(__file__).resolve().parents[3] / "apps" / "web" / "src" / "api.ts"
    if not browser.is_file():
        pytest.skip("the browser client is not present in this checkout")

    matched = set(
        re.findall(r'detail\?\.code === "([a-z_]+)"', browser.read_text(encoding="utf-8"))
    )
    assert matched, "expected the browser to match at least one detail code"

    raised = set(re.findall(r'"code": "([a-z_]+)"', API_SOURCE))
    missing = matched - raised
    assert not missing, (
        f"the browser matches {sorted(missing)}, which api.py no longer raises; "
        "the confirmation it gates would never appear"
    )
