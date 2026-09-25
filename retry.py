import asyncio
import logging
from functools import wraps
from typing import Callable, Tuple, Type

logger = logging.getLogger(__name__)

# Exceptions that should trigger a retry
RETRYABLE = (TimeoutError, ConnectionError, OSError)

def async_retry(
    max_retries: int = 3,
    base_delay:  float = 1.0,
    max_delay:   float = 60.0,
    retryable:   Tuple[Type[Exception], ...] = RETRYABLE,
):
    def decorator(fn: Callable):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except retryable as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        break
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        "[Retry %d/%d] %s — sleeping %.1fs",
                        attempt + 1, max_retries, exc, delay
                    )
                    await asyncio.sleep(delay)
            raise last_exc  # re-raise after exhausting retries
        return wrapper
    return decorator