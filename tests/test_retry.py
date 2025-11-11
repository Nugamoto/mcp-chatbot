import time

import pytest

from src.utils.retry import retry_call, async_retry


def test_retry_call_succeeds_after_failures():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("temporary error")
        return "ok"

    start = time.time()
    result = retry_call(flaky, tries=3, backoff=0.01, jitter=0.0, exceptions=(RuntimeError,))
    dur = time.time() - start
    assert result == "ok"
    assert attempts["n"] == 3
    assert dur >= 0.01  # had to back off at least once


@pytest.mark.asyncio
async def test_async_retry_decorator():
    calls = {"n": 0}

    @async_retry(tries=2, backoff=0.01, jitter=0.0, exceptions=(ValueError,))
    async def sometimes():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("boom")
        return 42

    res = await sometimes()
    assert res == 42
    assert calls["n"] == 2
