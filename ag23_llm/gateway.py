"""The gateway: a thin, typed wrapper over `litellm.Router`.

Callers use `chat(...)` and never touch LiteLLM directly. LiteLLM owns transport,
fallback, retries, and rate-limit cooldowns across every free provider; this module
owns the *public surface* (ChatMessage/ChatResult) and the routing decision:

  - `task="coding"`  → task-router maps it to tags → LiteLLM tag-filters the `free`
    group to models good at that task (the benchmark-cluster seam).
  - `provider=`/`model=` → force one specific deployment (skips the group/fallback).
  - neither → route across all configured free models.
"""
from __future__ import annotations

from functools import lru_cache
from time import perf_counter
from typing import Optional, Union

from . import config_gen, registry, telemetry   # telemetry is stdlib-only + no-op when off
from .config import get_config
from .config_gen import GROUP
from .types import (AllProvidersFailed, ChatMessage, ChatResult, LLMError,
                    Usage)

MessagesInput = Union[str, list[ChatMessage], list[dict]]

# Model ids discovered from a local provider's /models endpoint, keyed by provider id.
# Populated lazily on the first forced call to a local provider with no default_model.
_discovered_model_cache: dict[str, str] = {}


def _discover_local_model(prov: "registry.Provider") -> str:
    """Resolve a local (e.g. vLLM) provider's served model from its OpenAI-compatible
    GET {base_url}/models endpoint (data[0].id). Cached by the caller per provider id."""
    import httpx

    base = prov.resolved_base_url().rstrip("/")
    resp = httpx.get(f"{base}/models", timeout=10.0)
    resp.raise_for_status()
    data = resp.json().get("data") or []
    if not data or not data[0].get("id"):
        raise LLMError(
            f"provider '{prov.id}' returned no models from {base}/models"
        )
    return data[0]["id"]


def _normalize_messages(messages: MessagesInput) -> list[dict]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    out: list[dict] = []
    for m in messages:
        if isinstance(m, ChatMessage):
            out.append(m.as_openai())
        elif isinstance(m, dict):
            out.append({"role": m["role"], "content": m["content"]})
        else:
            raise TypeError(f"unsupported message type: {type(m)}")
    return out


@lru_cache(maxsize=1)
def get_router():
    """Build (once) the in-process LiteLLM Router from the current registry."""
    try:
        from litellm import Router
    except ImportError as e:  # pragma: no cover
        raise LLMError(
            "litellm is not installed. `pip install -r requirements.txt` "
            "(added litellm) in the backend venv."
        ) from e

    kwargs = config_gen.build_router_kwargs()
    if not kwargs["model_list"]:
        raise LLMError(
            "no providers are configured. Set at least one provider key (e.g. "
            "GROQ_API_KEY) in backend/.env — see .env.example."
        )
    return Router(**kwargs)


def reset_router() -> None:
    """Drop the cached Router so the next call rebuilds from a changed registry
    (e.g. after the scout adds a provider, or a key is set)."""
    get_router.cache_clear()


def _result_from_response(resp, *, requested_provider: Optional[str]) -> ChatResult:
    choice = resp.choices[0]
    text = getattr(choice.message, "content", None) or ""
    usage = getattr(resp, "usage", None)
    hidden = getattr(resp, "_hidden_params", {}) or {}
    # A forced provider is the registry id the caller asked for — keep it. LiteLLM's
    # custom_llm_provider is the wire format ("openai"), which misattributes stats
    # for any OpenAI-compatible provider (e.g. local-vllm showed up as "openai").
    provider = (requested_provider
                or hidden.get("custom_llm_provider")
                or (hidden.get("model_info") or {}).get("provider")
                or "unknown")
    return ChatResult(
        text=text,
        provider=str(provider),
        model=getattr(resp, "model", "") or "",
        usage=Usage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        ),
        raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
    )


def chat(
    messages: MessagesInput,
    *,
    task: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    tags: Optional[list[str]] = None,
) -> ChatResult:
    """Run a chat completion over the free-model gateway.

    task:     capability hint ("coding", "reasoning", …). Resolved to tags by the
              task-router; LiteLLM then routes within the `free` group by those tags.
              Ignored when config.task_routing is False (plain load-balancing then).
    provider: force a specific provider id from the registry (skips group routing).
    model:    force a raw model id (used with `provider`, or as a full LiteLLM string).
    tags:     pass tags directly, bypassing the task-router.
    """
    msgs = _normalize_messages(messages)
    common = {"messages": msgs, "temperature": temperature}
    if max_tokens:
        common["max_tokens"] = max_tokens

    cluster: Optional[str] = None          # recorded in telemetry
    requested_provider = provider

    # ── decide how to execute (build the call thunk) ────────────────────────
    if provider or (model and "/" not in (model or "")):
        # forced single provider/model
        import litellm
        prov = registry.get_provider(provider) if provider else None
        if provider and not prov:
            raise LLMError(f"unknown provider '{provider}' (see providers.json)")
        target_model = model or (prov.default_model if prov else None)
        discovered = False
        if not target_model and prov and prov.local:
            # Local provider (e.g. vLLM) that doesn't pin a model: discover the
            # served model from its /models endpoint, cached per provider id.
            target_model = _discovered_model_cache.get(prov.id)
            if not target_model:
                target_model = _discover_local_model(prov)
                _discovered_model_cache[prov.id] = target_model
            discovered = True
        if not target_model:
            raise LLMError(f"provider '{provider}' has no default_model; pass model=")
        params = config_gen.litellm_model_params(prov, target_model, use_env_ref=False)
        if discovered:
            # The served id from /models is passed as-is; custom_llm_provider (not a
            # "openai/" model prefix) tells LiteLLM to route it to the local endpoint.
            params["model"] = target_model
            params["custom_llm_provider"] = "openai"
        label = provider or "forced"

        def _call():
            return litellm.completion(**{**params, **common})

    elif model:
        # forced full LiteLLM model string (e.g. "groq/llama-3.1-8b-instant")
        import litellm
        label = model

        def _call():
            return litellm.completion(model=model, **common)

    else:
        # route across the `free` group. task_routing=False → no tags → load-balance.
        cfg = get_config()
        resolved_tags = tags
        if resolved_tags is None and task and cfg.task_routing:
            from . import taskrouter
            resolved_tags = taskrouter.tags_for_task(task, semantic=cfg.semantic_router) or None
        cluster = resolved_tags[0] if resolved_tags else None
        router = get_router()
        call = {"model": GROUP, **common}
        if resolved_tags:
            call["metadata"] = {"tags": resolved_tags}
        label = GROUP

        def _call():
            return router.completion(**call)

    # ── execute once: timed, with opt-in telemetry (no-op when disabled) ─────
    started = perf_counter()
    try:
        resp = _call()
    except Exception as e:  # noqa: BLE001 — surface as AllProvidersFailed
        telemetry.record(provider=requested_provider or label, model=model or label,
                         task=task, cluster=cluster, ok=False, error=type(e).__name__,
                         latency_ms=(perf_counter() - started) * 1000)
        raise AllProvidersFailed([(label, str(e))]) from e

    result = _result_from_response(resp, requested_provider=requested_provider)
    telemetry.record(provider=result.provider, model=result.model, task=task,
                     cluster=cluster, ok=True, total_tokens=result.usage.total_tokens,
                     latency_ms=(perf_counter() - started) * 1000)
    return result
