"""ag23-llm: a task-routed gateway over many free LLM providers.

Public surface (this is the whole importable library — pure, no side effects on import):
    from ag23_llm import chat
    chat("write a binary search in python", task="coding")

Under the hood: providers.json (registry) → LiteLLM Router (transport, fallback,
rate-limit cooldowns) → Semantic Router / benchmark clusters (which model per task).

The provider *scout* — which browses the web and rewrites the registry — is deliberately
NOT in here. It's a separate tool (`../scout`, run via `python -m scout`) that imports
this package and operates on it. Importing ag23_llm never triggers any of that.
"""
__version__ = "0.1.0"

from .gateway import chat, reset_router
from .types import ChatMessage, ChatRequest, ChatResult, LLMError

__all__ = ["chat", "reset_router", "ChatMessage", "ChatRequest", "ChatResult", "LLMError"]
