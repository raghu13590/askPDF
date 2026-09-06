import asyncio

import pytest

from langgraph_runtime.workflows.deep_research_execution import run_cancellable


class Token:
    def __init__(self):
        self.cancelled = False

    async def requested(self):
        return self.cancelled


@pytest.mark.asyncio
async def test_run_cancellable_stops_blocking_work():
    token = Token()
    stopped = asyncio.Event()

    async def work():
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    operation = asyncio.create_task(run_cancellable(work(), token, poll_seconds=0.001))
    await asyncio.sleep(0)
    token.cancelled = True
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_run_cancellable_cleans_up_on_timeout():
    stopped = asyncio.Event()

    async def work():
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    with pytest.raises(asyncio.TimeoutError):
        await run_cancellable(work(), Token(), timeout_seconds=0.001, poll_seconds=0.001)
    assert stopped.is_set()
