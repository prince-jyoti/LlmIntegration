import asyncio
import time

class RateLimiter:
    """
    Token-bucket rate limiter.
    Allows `rate` requests per `period` seconds.
    """
    def __init__(self, rate: int, period: float = 60.0):
        self._rate      = rate          # max tokens in bucket
        self._period    = period
        self._tokens    = float(rate)
        self._last_refill = time.monotonic()
        self._lock      = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            self._refill()
            if self._tokens < 1:
                wait = (1 - self._tokens) * (self._period / self._rate)
                await asyncio.sleep(wait)
                self._refill()
            self._tokens -= 1

    def _refill(self) -> None:
        now     = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            float(self._rate),
            self._tokens + elapsed * (self._rate / self._period)
        )
        self._last_refill = now