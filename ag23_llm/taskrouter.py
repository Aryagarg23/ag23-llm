"""Task → cluster → tags. The benchmark-cluster seam.

Two jobs:
  1. Turn a task hint into LiteLLM tags the gateway filters on (`tags_for_task`).
  2. Classify free-text into a cluster when the caller doesn't name one
     (`classify`), via Semantic Router over LOCAL embeddings (keyless, no LLM call),
     with a keyword fallback when Semantic Router isn't installed.

The per-cluster model *ranking* lives in benchmarks.json (seeded from strengths now,
your own evals later). Today the gateway routes by tag + LiteLLM's rate-limit-aware
strategy. When you want strict benchmark ordering, `BenchmarkClusterStrategy` below is
the LiteLLM `CustomRoutingStrategyBase` hook to plug that ranking straight into the
Router — that is where measured evals ultimately take over.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
BENCHMARKS_PATH = HERE / "benchmarks.json"

# Clusters are also the strength tags used in providers.json, so a cluster name *is*
# a valid LiteLLM tag. Synonyms map free-form task words onto a canonical cluster.
CLUSTER_SYNONYMS = {
    "code": "coding", "coding": "coding", "programming": "coding", "debug": "coding",
    "reason": "reasoning", "reasoning": "reasoning", "logic": "reasoning", "analysis": "reasoning",
    "math": "math", "arithmetic": "math", "calculation": "math",
    "long": "long-context", "long-context": "long-context", "document": "long-context",
    "rag": "rag", "retrieval": "rag", "grounded": "rag",
    "vision": "vision", "image": "vision", "multimodal": "vision",
    "fast": "fast-cheap", "cheap": "fast-cheap", "fast-cheap": "fast-cheap", "quick": "fast-cheap",
    "general": "general", "chat": "general", "summarize": "general", "write": "general",
}

# Example utterances per cluster for Semantic Router's embedding match.
CLUSTER_UTTERANCES = {
    "coding": ["write a python function", "fix this bug", "refactor this code",
               "what's wrong with this stack trace", "implement a REST endpoint"],
    "reasoning": ["explain step by step why", "what's the logical flaw here",
                  "reason about the tradeoffs", "prove this claim"],
    "math": ["solve this equation", "what is the integral of", "compute the probability",
             "simplify this expression"],
    "long-context": ["summarize this long document", "answer questions about this 40 page pdf",
                     "find the clause in this contract"],
    "rag": ["answer using the provided sources", "cite the documents",
            "ground your answer in the retrieved passages"],
    "vision": ["what's in this image", "describe this screenshot", "read the text in this photo"],
    "fast-cheap": ["quick one-liner", "just a short reply", "classify this into a label"],
    "general": ["tell me about", "help me write an email", "what do you think about"],
}


def canonical_cluster(task: str) -> Optional[str]:
    """Map a task word to a canonical cluster, if it names one directly."""
    t = task.strip().lower()
    if t in CLUSTER_SYNONYMS:
        return CLUSTER_SYNONYMS[t]
    if t in CLUSTER_UTTERANCES:
        return t
    return None


def tags_for_task(task: str, *, semantic: bool = True) -> list[str]:
    """Tags the gateway should filter the `free` group by for this task.

    A named cluster maps straight to its tag; free-text is classified first.
    semantic=False keeps it to keyword classification (never imports semantic-router)."""
    cluster = canonical_cluster(task) or classify(task, semantic=semantic)
    return [cluster] if cluster else []


# ── classification ───────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _semantic_router():
    """Build a Semantic Router over local (keyless) embeddings, or None if unavailable."""
    try:
        from semantic_router import Route
        from semantic_router.routers import SemanticRouter
        try:
            from semantic_router.encoders import FastEmbedEncoder
            encoder = FastEmbedEncoder()
        except Exception:
            from semantic_router.encoders import HuggingFaceEncoder
            encoder = HuggingFaceEncoder()
        routes = [Route(name=name, utterances=utts)
                  for name, utts in CLUSTER_UTTERANCES.items()]
        return SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")
    except Exception:
        return None


def _keyword_classify(query: str) -> Optional[str]:
    q = query.lower()
    for word, cluster in CLUSTER_SYNONYMS.items():
        if word in q:
            return cluster
    return None


def classify(query: str, *, semantic: bool = True) -> Optional[str]:
    """Classify free text into a cluster. semantic=True uses Semantic Router when
    installed (importing it lazily), then keywords; semantic=False is keyword-only and
    never imports semantic-router/fastembed."""
    if semantic:
        router = _semantic_router()
        if router is not None:
            try:
                choice = router(query)
                if choice and getattr(choice, "name", None):
                    return choice.name
            except Exception:
                pass
    return _keyword_classify(query)


# ── benchmark cluster table ──────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _benchmarks() -> dict:
    with BENCHMARKS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def ranked_ids_for(cluster: str) -> list[str]:
    """Best→worst `<provider>:<model>` ids for a cluster (from benchmarks.json)."""
    return list(_benchmarks().get("clusters", {}).get(cluster, []))


def clusters() -> list[str]:
    return sorted(CLUSTER_UTTERANCES.keys())


# ── the benchmark → LiteLLM seam (not wired by default) ──────────────────────
def build_strategy():
    """Return a LiteLLM `CustomRoutingStrategyBase` that orders deployments by the
    benchmark cluster table, or None if LiteLLM isn't installed.

    NOTE: intentionally a stub. Today the gateway routes by tag + rate-limit-aware
    strategy, which is enough. Flip this on once benchmarks.json holds *measured*
    rankings you trust — then `router.set_custom_routing_strategy(build_strategy())`
    makes the Router pick strictly by your evals, per cluster.
    """
    try:
        from litellm.router_strategy.base_routing_strategy import CustomRoutingStrategyBase
    except Exception:
        return None

    class BenchmarkClusterStrategy(CustomRoutingStrategyBase):  # pragma: no cover
        async def async_get_available_deployment(self, model, messages=None, input=None,
                                                 specific_deployment=None, request_kwargs=None):
            return self._pick(model, request_kwargs)

        def get_available_deployment(self, model, messages=None, input=None,
                                     specific_deployment=None, request_kwargs=None):
            return self._pick(model, request_kwargs)

        def _pick(self, model, request_kwargs):
            # TODO: read tags from request_kwargs -> cluster -> ranked_ids_for(cluster),
            # then return the highest-ranked deployment that isn't on cooldown.
            raise NotImplementedError(
                "benchmark cluster strategy not enabled yet — see taskrouter.build_strategy()"
            )

    return BenchmarkClusterStrategy()
