"""Acceptance test for issue #1: local-vllm should auto-discover the served
model instead of requiring the caller to pass model= explicitly.

A registry provider marked local: true with default_model: null should have
its model resolved at call time from GET {base_url}/models — cached so repeat
calls don't re-probe. Providers that are neither local nor have a
default_model still raise, unchanged.
"""
from unittest.mock import patch

import pytest

from ag23_llm import gateway, registry
from ag23_llm.types import LLMError


def _local_provider(**overrides):
    fields = dict(
        id="local-vllm", name="Local vLLM", wire_format="openai",
        base_url="http://localhost:8000/v1", auth_scheme="none", auth_env=[],
        default_model=None, models=[], rate_limits={}, priority=100,
        status="active", local=True,
    )
    fields.update(overrides)
    return registry.Provider(**fields)


@pytest.fixture(autouse=True)
def _clear_discovery_cache():
    gateway._discovered_model_cache.clear()
    yield
    gateway._discovered_model_cache.clear()


def test_local_provider_without_default_model_discovers_at_call_time(monkeypatch):
    prov = _local_provider()
    monkeypatch.setattr(registry, "get_provider", lambda pid: prov)
    monkeypatch.setattr(gateway, "_discover_local_model", lambda p: "Qwen3-Coder-30B")

    captured = {}
    class FakeResp:
        choices = [type("C", (), {"message": type("M", (), {"content": "ok"})()})]
        usage = None
        model = "Qwen3-Coder-30B"
        _hidden_params = {}

    with patch("litellm.completion", lambda **kw: captured.update(kw) or FakeResp()):
        result = gateway.chat("hi", provider="local-vllm")

    assert captured["model"] == "Qwen3-Coder-30B"
    assert result.model == "Qwen3-Coder-30B"


def test_local_provider_discovery_is_cached(monkeypatch):
    prov = _local_provider()
    monkeypatch.setattr(registry, "get_provider", lambda pid: prov)
    calls = []
    monkeypatch.setattr(gateway, "_discover_local_model", lambda p: calls.append(1) or "M")

    class FakeResp:
        choices = [type("C", (), {"message": type("M", (), {"content": "ok"})()})]
        usage = None
        model = "M"
        _hidden_params = {}

    with patch("litellm.completion", lambda **kw: FakeResp()):
        gateway.chat("hi", provider="local-vllm")
        gateway.chat("hi", provider="local-vllm")

    assert len(calls) == 1  # second call served from cache, not re-probed


def test_non_local_provider_without_default_model_still_raises(monkeypatch):
    prov = _local_provider(local=False)
    monkeypatch.setattr(registry, "get_provider", lambda pid: prov)
    with pytest.raises(LLMError, match="no default_model"):
        gateway.chat("hi", provider="local-vllm")


def test_explicit_model_bypasses_discovery(monkeypatch):
    prov = _local_provider()
    monkeypatch.setattr(registry, "get_provider", lambda pid: prov)

    def boom(p):
        raise AssertionError("should not probe when model= is given explicitly")
    monkeypatch.setattr(gateway, "_discover_local_model", boom)

    class FakeResp:
        choices = [type("C", (), {"message": type("M", (), {"content": "ok"})()})]
        usage = None
        model = "manual"
        _hidden_params = {}

    with patch("litellm.completion", lambda **kw: FakeResp()):
        gateway.chat("hi", provider="local-vllm", model="manual-model")
