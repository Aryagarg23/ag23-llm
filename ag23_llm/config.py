"""Runtime configuration — all optional, with env-var defaults.

Deliberately dependency-free: importing or changing config never pulls in litellm,
semantic-router, or fastembed. Those load lazily, only when the feature that needs them
actually runs. So a base install and an idle import stay cheap.

    import ag23_llm
    ag23_llm.configure(task_routing=False, telemetry=True)   # plain load-balancer + tracking
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    # Honor task hints and route by capability. False → ignore `task=` and just
    # load-balance across all providers (LiteLLM's built-in balancing). When False,
    # nothing in the task-router runs and semantic-router is never touched.
    task_routing: bool = True
    # For free-text task classification, use Semantic Router (embeddings). False →
    # keyword classification only, which never imports semantic-router/fastembed.
    # (Named clusters like "coding" map directly and need neither.)
    semantic_router: bool = True
    # Opt-in telemetry: record per-call provider/model/latency/success — never content.
    telemetry: bool = False
    # Optional path to append telemetry events as JSONL (in addition to the logger).
    telemetry_file: Optional[str] = None


_config = Config(
    task_routing=_env_bool("AG23_LLM_TASK_ROUTING", True),
    semantic_router=_env_bool("AG23_LLM_SEMANTIC_ROUTER", True),
    telemetry=_env_bool("AG23_LLM_TELEMETRY", False),
    telemetry_file=os.environ.get("AG23_LLM_TELEMETRY_FILE"),
)


def get_config() -> Config:
    return _config


def configure(**kwargs) -> Config:
    """Set options at runtime, e.g. configure(task_routing=False, telemetry=True)."""
    for key, value in kwargs.items():
        if not hasattr(_config, key):
            raise AttributeError(f"unknown config option: {key!r}")
        setattr(_config, key, value)
    return _config
