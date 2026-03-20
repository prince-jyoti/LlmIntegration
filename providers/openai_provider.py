import asyncio
import time
from typing import AsyncIterator, List

import httpx

from base import BaseLLMProvider, LLMConfig, LLMResponse, Message
from rate_limiter import RateLimiter
from retry import async_retry

class OpenAIProvider(BaseLLMProvider):
    BASE_URL = "https://api.openai.com/v1"

    def __init__(self, api_key: str, config: LLMConfig):
        super().__init__(api_key, config)
        self._limiter = RateLimiter(rate=config.requests_per_minute)
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _build_body(self, messages: List[Message]) -> dict:
        return {
            "model":       self.config.model,
            "max_tokens":  self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages":    [{"role": m.role, "content": m.content} for m in messages],
        }

    @async_retry()
    async def complete(self, messages: List[Message]) -> LLMResponse:
        await self._limiter.acquire()
        t0 = time.monotonic()

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            resp = await client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self._headers,
                json=self._build_body(messages),
            )
            resp.raise_for_status()

        data    = resp.json()
        choice  = data["choices"][0]["message"]["content"]
        usage   = data.get("usage", {})

        return LLMResponse(
            content       = choice,
            model         = data["model"],
            provider      = "openai",
            input_tokens  = usage.get("prompt_tokens", 0),
            output_tokens = usage.get("completion_tokens", 0),
            latency_ms    = (time.monotonic() - t0) * 1000,
        )

    async def stream(self, messages: List[Message]) -> AsyncIterator[str]:
        await self._limiter.acquire()
        body = {**self._build_body(messages), "stream": True}

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            async with client.stream("POST",
                f"{self.BASE_URL}/chat/completions",
                headers=self._headers, json=body
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        import json
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # r = await client.get(f"{self.BASE_URL}/models", headers=self._headers)
                # return r.status_code == 200
                r = await client.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=self._headers,
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,  # cheapest possible check
                    },
                )
                return r.status_code == 200
        except httpx.RequestError:
            return False