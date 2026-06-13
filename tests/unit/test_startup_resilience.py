import asyncio
import time

import pytest

from routstr import proxy
from routstr.core import main


@pytest.mark.asyncio
async def test_startup_dependencies_continue_fail_closed_when_initializer_hangs() -> (
    None
):
    """Startup must not wait forever on provider/model initialization."""
    started = asyncio.Event()

    async def hangs() -> None:
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(hangs(), name="hanging-startup-dependency")

    try:
        pending = await asyncio.wait_for(
            main._await_startup_dependencies(
                {"initialize_upstreams": task},
                timeout_seconds=0.01,
            ),
            timeout=0.5,
        )
        assert started.is_set()
        assert pending == [task]
        assert not task.done()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_info_confidentiality_lookup_timeout_fails_closed(monkeypatch) -> None:
    """Readiness must not hang when confidential status collection blocks."""

    def stuck_confidentiality_status() -> dict[str, object]:
        time.sleep(0.2)
        return {
            "mode": "required",
            "required": True,
            "end_to_end_ready": True,
            "routable_with_full_attestation": {
                "tinfoil": ["deepseek-v4-pro"],
            },
            "routstr_tee": {
                "ready": True,
                "required": True,
                "client_confidentiality": {
                    "mode": "attested-tls-termination",
                    "tls_terminates_in_attested_tee": True,
                    "inbound_ehbp_ohttp_request_decryption": False,
                },
            },
        }

    monkeypatch.setattr(proxy, "get_confidentiality_status", stuck_confidentiality_status)
    monkeypatch.setattr(main, "INFO_CONFIDENTIALITY_TIMEOUT_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(
        main.global_settings,
        "confidential_routing_mode",
        "required",
        raising=False,
    )

    started = time.monotonic()
    response = await main.info()
    elapsed = time.monotonic() - started

    confidentiality = response["confidentiality"]
    assert elapsed < 0.1
    assert confidentiality["mode"] == "required"
    assert confidentiality["required"] is True
    assert confidentiality["end_to_end_ready"] is False
    assert confidentiality["routable_with_full_attestation"] == {
        "tinfoil": [],
        "ppq-private": [],
        "privatemode": [],
    }
