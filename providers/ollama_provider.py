import json
import time
from typing import AsyncIterator, List

import httpx

from base import BaseLLMProvider, LLMConfig, LLMResponse, Message
from retry import async_retry

class OllamaProvider(BaseLLMProvider):
    """Local Ollama instance — no API key needed."""

    def __init__(self, config: LLMConfig, base_url: str = "http://localhost:11434"):
        super().__init__(api_key=None, config=config)
        self._base_url = base_url.rstrip("/")

    @async_retry()
    async def complete(self, messages: List[Message]) -> LLMResponse:
        t0   = time.monotonic()
        body = {
            "model":    self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream":   False,
            "options":  {
                "num_predict": self.config.max_tokens,
                "temperature": self.config.temperature,
            },
        }
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=body)
            resp.raise_for_status()

        data = resp.json()
        return LLMResponse(
            content       = data["message"]["content"],
            model         = data.get("model", self.config.model),
            provider      = "ollama",
            input_tokens  = data.get("prompt_eval_count", 0),
            output_tokens = data.get("eval_count", 0),
            latency_ms    = (time.monotonic() - t0) * 1000,
        )

    async def stream(self, messages: List[Message]) -> AsyncIterator[str]:
        body = {
            "model":    self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream":   True,
        }
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            async with client.stream("POST",
                f"{self._base_url}/api/chat", json=body
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        yield chunk["message"]["content"]
                        if chunk.get("done"):
                            break

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{self._base_url}/api/tags")
                return r.status_code == 200
        except httpx.RequestError:
            return False