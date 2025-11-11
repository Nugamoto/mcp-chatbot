import random
import time
import asyncio
from typing import Any, Callable, Iterable, Tuple, Type


def retry_call(
    func: Callable[[], Any],
    tries: int = 3,
    backoff: float = 0.5,
    jitter: float = 0.25,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> Any:
    """Call a sync function with retry, exponential backoff, and jitter."""
    attempt = 0
    delay = backoff
    while True:
        try:
            return func()
        except exceptions as e:
            attempt += 1
            if attempt >= tries:
                raise
            sleep_for = delay + random.uniform(0, jitter)
            time.sleep(sleep_for)
            delay *= 2


def async_retry(
    tries: int = 3,
    backoff: float = 0.5,
    jitter: float = 0.25,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
):
    """Decorator for retrying async functions with exponential backoff and jitter."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            delay = backoff
            while True:
                try:
                    return await fn(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= tries:
                        raise
                    sleep_for = delay + random.uniform(0, jitter)
                    await asyncio.sleep(sleep_for)
                    delay *= 2
        return wrapper

    return decorator
