"""Normalized request/response types for the LLM gateway.

Every provider adapter takes a `ChatRequest` and returns a `ChatResult`, regardless
of the wire format underneath (OpenAI-compatible, Gemini-native, Cohere). Keeping the
gateway's public surface in these types is what lets us swap or add providers without
touching callers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str

    def as_openai(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatRequest:
    messages: list[ChatMessage]
    # Optional caller hint. When set, the task-router biases model/provider choice
    # toward models tagged for this kind of work (see taskrouter.py). This is the
    # seam the future benchmark-cluster routing plugs into.
    task: Optional[str] = None
    model: Optional[str] = None            # force a specific model id (skips task routing)
    provider: Optional[str] = None         # force a specific provider id (skips fallback ordering)
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    # Providers/models the caller wants excluded (used by the gateway during fallback
    # to avoid retrying something that just failed).
    exclude_providers: list[str] = field(default_factory=list)

    @property
    def prompt_text(self) -> str:
        """Flattened text of the request, for logging/verification heuristics."""
        return "\n".join(m.content for m in self.messages)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResult:
    text: str
    provider: str            # provider id that served the request
    model: str               # model id that served the request
    usage: Usage = field(default_factory=Usage)
    # Providers that were tried and failed before this one succeeded (fallback trail).
    fallback_from: list[str] = field(default_factory=list)
    raw: Optional[dict] = None

    def __str__(self) -> str:  # convenience for CLI
        return self.text


class LLMError(Exception):
    """Base class for gateway errors."""


class ProviderError(LLMError):
    """A single provider failed. Carries whether the failure is a rate-limit so the
    gateway can put the provider on cooldown vs. treat it as a hard error."""

    def __init__(self, provider: str, message: str, *, rate_limited: bool = False,
                 status: Optional[int] = None):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.rate_limited = rate_limited
        self.status = status


class AllProvidersFailed(LLMError):
    """Every eligible provider was exhausted without a success."""

    def __init__(self, attempts: list[tuple[str, str]]):
        self.attempts = attempts  # [(provider, error), ...]
        detail = "; ".join(f"{p}: {e}" for p, e in attempts) or "no eligible providers"
        super().__init__(f"all providers failed ({detail})")
