from __future__ import annotations

from httpx2 import ASGITransport, AsyncClient

from local_lm.config import Settings
from local_lm.main import create_app


async def test_non_dev_api_requires_cookie_and_csrf(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=tmp_path / "secure", dev=False)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            assert (await client.get("/api/projects")).status_code == 401
            session = await client.post("/api/session")
            assert session.status_code == 200
            csrf = session.json()["csrf_token"]
            assert (await client.get("/api/projects")).status_code == 200
            denied = await client.post("/api/projects", json={"name": "Denied"})
            assert denied.status_code == 403
            allowed = await client.post(
                "/api/projects",
                json={"name": "Allowed"},
                headers={"x-local-lm-csrf": csrf},
            )
            assert allowed.status_code == 201
