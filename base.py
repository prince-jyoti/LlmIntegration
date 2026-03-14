# base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional
from enum import Enum

class Role(str, Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"

@dataclass
class Message:
    role: Role
    content: str

@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    input_tokens: int  = 0
    output_tokens: int = 0
    latency_ms: float  = 0.0

@dataclass
class LLMConfig:
    model: str
    max_tokens: int          = 1024
    temperature: float       = 0.7
    timeout_seconds: float   = 30.0
    max_retries: int         = 3
    requests_per_minute: int = 60

class BaseLLMProvider(ABC):
    def __init__(self, api_key: Optional[str], config: LLMConfig):
        self.api_key = api_key
        self.config  = config

    @abstractmethod
    async def complete(self, messages: list[Message]) -> LLMResponse:
        """Send messages and return a single completion."""
        ...

    @abstractmethod
    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Stream tokens as they arrive."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable."""
        ...