import time
from typing import AsyncIterator, List

import httpx

from base import BaseLLMProvider, LLMConfig, LLMResponse, Message, Role
from rate_limiter import RateLimiter
from retry import async_retry

class AnthropicProvider(BaseLLMProvider):
    BASE_URL = "https://api.anthropic.com/v1"

    def __init__(self, api_key: str, config: LLMConfig):
        super().__init__(api_key, config)
        self._limiter = RateLimiter(rate=config.requests_per_minute)
        self._headers = {
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type":      "application/json",
        }

    def _split_system(self, messages: List[Message]):
        system   = next((m.content for m in messages if m.role == Role.SYSTEM), None)
        filtered = [m for m in messages if m.role != Role.SYSTEM]
        return system, filtered

    def _build_body(self, messages: List[Message]) -> dict:
        system, msgs = self._split_system(messages)
        body = {
            "model":       self.config.model,
            "max_tokens":  self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages":    [{"role": m.role, "content": m.content} for m in msgs],
        }
        if system:
            body["system"] = system
        return body

    @async_retry()
    async def complete(self, messages: List[Message]) -> LLMResponse:
        await self._limiter.acquire()
        t0 = time.monotonic()

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            resp = await client.post(
                f"{self.BASE_URL}/messages",
                headers=self._headers,
                json=self._build_body(messages),
            )
            resp.raise_for_status()

        data  = resp.json()
        usage = data.get("usage", {})

        return LLMResponse(
            content       = data["content"][0]["text"],
            model         = data["model"],
            provider      = "anthropic",
            input_tokens  = usage.get("input_tokens", 0),
            output_tokens = usage.get("output_tokens", 0),
            latency_ms    = (time.monotonic() - t0) * 1000,
        )

    async def stream(self, messages: List[Message]) -> AsyncIterator[str]:
        await self._limiter.acquire()
        body = {**self._build_body(messages), "stream": True}

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            async with client.stream("POST",
                f"{self.BASE_URL}/messages",
                headers=self._headers, json=body
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        import json
                        event = json.loads(line[5:].strip())
                        if event.get("type") == "content_block_delta":
                            yield event["delta"].get("text", "")

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.BASE_URL}/models", headers=self._headers)
                return r.status_code == 200
        except httpx.RequestError:
            return False