"""Lightweight, opt-in telemetry — stdlib only.

Records one structured event per gateway call: which provider/model served it, the task/
cluster it routed to, latency, and success or error type. It NEVER records prompts or
completions. Aggregates (call count, error rate, average latency, per-provider breakdown)
are kept in memory and exposed via `stats()`.

Cost when disabled: a single boolean check and an early return — nothing is built, no file
is touched, no logger fires. Enable with `configure(telemetry=True)` or AG23_LLM_TELEMETRY=1.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("ag23_llm")


@dataclass
class _Agg:
    calls: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    by_provider: dict = field(default_factory=dict)


_agg = _Agg()
_lock = threading.Lock()


def record(*, provider: str, model: str, latency_ms: float, ok: bool,
           task: Optional[str] = None, cluster: Optional[str] = None,
           error: Optional[str] = None, total_tokens: int = 0) -> None:
    """Record one call. No-op (cheap) unless telemetry is enabled."""
    from .config import get_config
    cfg = get_config()
    if not cfg.telemetry:
        return

    event = {
        "ts": round(time.time(), 3),
        "provider": provider,
        "model": model,
        "task": task,
        "cluster": cluster,
        "latency_ms": round(latency_ms, 1),
        "ok": ok,
        "error": error,
        "total_tokens": total_tokens,
    }

    with _lock:
        _agg.calls += 1
        _agg.total_latency_ms += latency_ms
        if not ok:
            _agg.errors += 1
        p = _agg.by_provider.setdefault(provider, {"calls": 0, "errors": 0, "latency_ms": 0.0})
        p["calls"] += 1
        p["latency_ms"] += latency_ms
        if not ok:
            p["errors"] += 1

    (logger.error if not ok else logger.info)("ag23_llm.call %s", event)

    if cfg.telemetry_file:
        try:
            with open(cfg.telemetry_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:  # never let telemetry break a call
            pass


def stats() -> dict:
    """Aggregate view: totals, error rate, average latency, per-provider breakdown."""
    with _lock:
        avg = (_agg.total_latency_ms / _agg.calls) if _agg.calls else 0.0
        by_provider = {
            name: {
                "calls": v["calls"],
                "errors": v["errors"],
                "avg_latency_ms": round(v["latency_ms"] / v["calls"], 1) if v["calls"] else 0.0,
            }
            for name, v in _agg.by_provider.items()
        }
        return {
            "calls": _agg.calls,
            "errors": _agg.errors,
            "error_rate": round(_agg.errors / _agg.calls, 3) if _agg.calls else 0.0,
            "avg_latency_ms": round(avg, 1),
            "by_provider": by_provider,
        }


def reset() -> None:
    """Clear the in-memory aggregates."""
    global _agg
    with _lock:
        _agg = _Agg()


def aggregate(events: list[dict]) -> dict:
    """Same summary shape as stats(), computed over raw JSONL events (for `ag23-llm stats`)."""
    calls = len(events)
    errors = sum(1 for e in events if not e.get("ok", True))
    total = sum(e.get("latency_ms", 0) or 0 for e in events)
    by: dict = {}
    for e in events:
        d = by.setdefault(e.get("provider", "?"), {"calls": 0, "errors": 0, "latency_ms": 0.0})
        d["calls"] += 1
        d["latency_ms"] += e.get("latency_ms", 0) or 0
        if not e.get("ok", True):
            d["errors"] += 1
    return {
        "calls": calls,
        "errors": errors,
        "error_rate": round(errors / calls, 3) if calls else 0.0,
        "avg_latency_ms": round(total / calls, 1) if calls else 0.0,
        "by_provider": {
            name: {"calls": v["calls"], "errors": v["errors"],
                   "avg_latency_ms": round(v["latency_ms"] / v["calls"], 1) if v["calls"] else 0.0}
            for name, v in by.items()
        },
    }
