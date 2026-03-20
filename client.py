import os
import logging
from typing import AsyncIterator, Dict, Literal, Optional

from base import LLMConfig, LLMResponse, Message, Role
from providers.openai_provider    import OpenAIProvider
from providers.anthropic_provider import AnthropicProvider
from providers.ollama_provider    import OllamaProvider

logger = logging.getLogger(__name__)

ProviderName = Literal["openai", "anthropic", "ollama"]

class LLMClient:
    """
    Unified async interface for OpenAI, Anthropic, and Ollama.

    Usage:
        client = LLMClient.from_env()
        response = await client.complete("openai", "GPT-4o", "Explain recursion.")
    """

    def __init__(
        self,
        openai_key:    Optional[str] = None,
        anthropic_key: Optional[str] = None,
        ollama_url:    str           = "http://localhost:11434",
    ):
        self._keys = {
            "openai":    openai_key,
            "anthropic": anthropic_key,
        }
        self._ollama_url = ollama_url

    @classmethod
    def from_env(cls) -> "LLMClient":
        """Load API keys from environment variables."""
        return cls(
            openai_key    = os.getenv("OPENAI_API_KEY"),
            anthropic_key = os.getenv("ANTHROPIC_API_KEY"),
            ollama_url    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    def _get_provider(self, provider: ProviderName, model: str, **kwargs):
        cfg = LLMConfig(model=model, **kwargs)
        if provider == "openai":
            if not self._keys["openai"]:
                raise ValueError("OPENAI_API_KEY not set")
            return OpenAIProvider(self._keys["openai"], cfg)
        elif provider == "anthropic":
            if not self._keys["anthropic"]:
                raise ValueError("ANTHROPIC_API_KEY not set")
            return AnthropicProvider(self._keys["anthropic"], cfg)
        elif provider == "ollama":
            return OllamaProvider(cfg, base_url=self._ollama_url)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def complete(
        self,
        provider:    ProviderName,
        model:       str,
        prompt:      str,
        system:      Optional[str] = None,
        **cfg_kwargs,
    ) -> LLMResponse:
        msgs = []
        if system:
            msgs.append(Message(role=Role.SYSTEM, content=system))
        msgs.append(Message(role=Role.USER, content=prompt))

        p = self._get_provider(provider, model, **cfg_kwargs)
        logger.info("→ %s / %s", provider, model)
        return await p.complete(msgs)

    async def stream(
        self,
        provider:    ProviderName,
        model:       str,
        prompt:      str,
        system:      Optional[str] = None,
        **cfg_kwargs,
    ) -> AsyncIterator[str]:
        msgs = []
        if system:
            msgs.append(Message(role=Role.SYSTEM, content=system))
        msgs.append(Message(role=Role.USER, content=prompt))

        p = self._get_provider(provider, model, **cfg_kwargs)
        async for token in p.stream(msgs):
            yield token

    async def health_check_all(self) -> Dict[str, bool]:
        import asyncio
        results = {}
        checks  = {
            "openai":    self._get_provider("openai",    "gpt-4o-mini"),
            "anthropic": self._get_provider("anthropic", "claude-haiku-4-5-20251001"),
            "ollama":    self._get_provider("ollama",    "llama3"),
        }
        for name, provider in checks.items():
            try:
                results[name] = await provider.health_check()
            except OSError:
                results[name] = False
        return results
