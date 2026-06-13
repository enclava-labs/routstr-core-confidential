import asyncio

import pytest

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
