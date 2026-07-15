"""Generate a LiteLLM Router config from the provider registry.

`providers.json` is the source of truth; this turns the *configured* providers into
LiteLLM deployments. Every model becomes one deployment in a single `free` model group,
and each model's `strengths` become LiteLLM **tags** — which is exactly the hook that
lets the task-router (Semantic Router → cluster → tag) steer calls, and later lets a
benchmark-driven custom strategy do the same. See taskrouter.py and gateway.py.

Two outputs from the same data:
  - `build_router_kwargs()` → kwargs for an in-process `litellm.Router(...)`.
  - `to_yaml()` → a proxy-server `config.yaml` (for `litellm --config`), for inspection
    or if you ever run the standalone proxy.
"""
from __future__ import annotations

from typing import Optional

from . import registry
from .registry import Provider

# The single group callers address. Tags (strengths) narrow it per task.
GROUP = "free"

# Providers LiteLLM supports natively — use the native prefix so its own cost/token
# accounting works. Everything else is called through LiteLLM's OpenAI-compatible
# passthrough (`openai/<model>` + api_base), which covers every OpenAI-shaped endpoint.
LITELLM_NATIVE_PREFIX = {
    "groq": "groq",
    "cerebras": "cerebras",
    "google-ai-studio": "gemini",
    "openrouter": "openrouter",
    "mistral": "mistral",
    "nvidia-nim": "nvidia_nim",
    "cohere": "cohere_chat",
}


def _model_ids(provider: Provider) -> list[str]:
    ids = [m.id for m in provider.models]
    if provider.default_model and provider.default_model not in ids:
        ids.insert(0, provider.default_model)
    return ids or ([provider.default_model] if provider.default_model else [])


def _strengths_for(provider: Provider, model_id: str) -> list[str]:
    for m in provider.models:
        if m.id == model_id:
            return list(m.strengths)
    return []


def litellm_model_params(provider: Provider, model_id: str, *, use_env_ref: bool) -> dict:
    """Build the `litellm_params` for one (provider, model) deployment.

    `use_env_ref=True` emits `os.environ/VAR` strings (for YAML the proxy resolves);
    `False` resolves the real key now (for an in-process Router)."""
    prefix = LITELLM_NATIVE_PREFIX.get(provider.id)

    def key_value() -> Optional[str]:
        if provider.auth_scheme == "none":
            return "EMPTY"
        if not provider.auth_env:
            return None
        return f"os.environ/{provider.auth_env[0]}" if use_env_ref else provider.api_key()

    params: dict = {}
    # Native providers whose base_url is the vendor default → use the native prefix.
    if prefix and "${" not in provider.base_url and provider.wire_format != "openai":
        # gemini / cohere: native, no api_base needed.
        params["model"] = f"{prefix}/{model_id}"
    elif prefix and provider.id in ("groq", "cerebras", "openrouter", "mistral", "nvidia-nim") \
            and "${" not in provider.base_url:
        params["model"] = f"{prefix}/{model_id}"
    else:
        # OpenAI-compatible passthrough for the long tail (+ any templated base_url).
        params["model"] = f"openai/{model_id}"
        params["api_base"] = provider.base_url if use_env_ref else provider.resolved_base_url()

    kv = key_value()
    if kv is not None:
        params["api_key"] = kv
    return params


def build_model_list(*, use_env_ref: bool = False,
                     providers: Optional[list[Provider]] = None) -> list[dict]:
    provs = providers if providers is not None else registry.configured_providers()
    deployments: list[dict] = []
    for p in provs:
        for model_id in _model_ids(p):
            if not model_id:
                continue
            tags = _strengths_for(p, model_id) + [f"provider:{p.id}"]
            deployments.append({
                "model_name": GROUP,
                "litellm_params": {
                    **litellm_model_params(p, model_id, use_env_ref=use_env_ref),
                    "tags": tags,
                },
                "model_info": {
                    "id": f"{p.id}:{model_id}",
                    "provider": p.id,
                    "base_model": model_id,
                    # rpm/tpm feed LiteLLM's rate-limit-aware routing + cooldowns.
                    **({"rpm": p.rpm()} if p.rpm() else {}),
                },
            })
    return deployments


def build_fallbacks() -> list[dict]:
    # Single group today, so within-group fallback is automatic. Kept as a seam for
    # when clusters become distinct model groups.
    return []


def build_router_kwargs() -> dict:
    """kwargs for `litellm.Router(**build_router_kwargs())` (in-process use)."""
    return {
        "model_list": build_model_list(use_env_ref=False),
        "enable_tag_filtering": True,
        # rate-limit-aware avoids a provider that's out of headroom, then cools it down.
        "routing_strategy": "usage-based-routing-v2",
        "num_retries": 3,
        "retry_after": 5,
        "allowed_fails": 2,
        "cooldown_time": 60,
    }


def to_yaml() -> str:
    """Proxy-server config.yaml (env refs preserved). Requires PyYAML."""
    import yaml  # local import: only needed for the proxy path

    doc = {
        "model_list": build_model_list(use_env_ref=True),
        "router_settings": {
            "enable_tag_filtering": True,
            "routing_strategy": "usage-based-routing-v2",
            "num_retries": 3,
            "cooldown_time": 60,
        },
    }
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
