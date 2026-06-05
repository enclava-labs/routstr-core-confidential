from __future__ import annotations

import secrets
import time
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import AsyncClient

from routstr.core.admin import admin_sessions
from routstr.core.db import ApiKey, AsyncSession


@pytest_asyncio.fixture
async def admin_client(
    integration_client: AsyncClient,
) -> AsyncGenerator[AsyncClient, None]:
    token = secrets.token_urlsafe(24)
    admin_sessions[token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {token}"
    try:
        yield integration_client
    finally:
        admin_sessions.pop(token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_can_create_prepaid_api_key(
    admin_client: AsyncClient,
    integration_client: AsyncClient,
    integration_session: AsyncSession,
) -> None:
    response = await admin_client.post(
        "/admin/api/apikeys",
        json={"balance_msats": 12345},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["api_key"].startswith("sk-")
    assert body["hashed_key"] == body["api_key"][3:]
    assert body["balance"] == 12345
    assert body["reserved_balance"] == 0
    assert body["total_spent"] == 0
    assert body["total_requests"] == 0

    stored = await integration_session.get(ApiKey, body["hashed_key"])
    assert stored is not None
    assert stored.balance == 12345

    integration_client.headers["Authorization"] = f"Bearer {body['api_key']}"
    balance = await integration_client.get("/v1/balance/info")
    assert balance.status_code == 200
    assert balance.json()["balance"] == 12345


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_create_prepaid_api_key_requires_admin(
    integration_client: AsyncClient,
) -> None:
    response = await integration_client.post(
        "/admin/api/apikeys",
        json={"balance_msats": 12345},
    )

    assert response.status_code == 403
