"""Loads and validates the data-driven provider registry (providers.json).

The registry is deliberately *data*, not code: the scout can append a verified
provider as a JSON row (safe, schema-checked) without an LLM ever editing executable
Python. `registry.py` is the only thing that reads that data into typed `Provider`
objects and decides which providers are actually usable right now (auth present).
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
REGISTRY_PATH = HERE / "providers.json"

VALID_WIRE_FORMATS = {"openai", "gemini", "cohere"}
_ENV_TEMPLATE = re.compile(r"\$\{([A-Z0-9_]+)\}")


@dataclass
class ModelInfo:
    id: str
    strengths: list[str] = field(default_factory=list)


@dataclass
class Provider:
    id: str
    name: str
    wire_format: str
    base_url: str
    auth_scheme: str                 # bearer | sdk | none
    auth_env: list[str]
    default_model: Optional[str]
    models: list[ModelInfo]
    rate_limits: dict
    priority: int = 0
    models_path: Optional[str] = None
    docs: str = ""
    source: str = ""
    status: str = "active"
    local: bool = False

    # ── availability ────────────────────────────────────────────────────────
    def missing_env(self) -> list[str]:
        """Auth env vars that are required but not set."""
        return [v for v in self.auth_env if not os.environ.get(v)]

    @property
    def is_configured(self) -> bool:
        """True when the credentials this provider needs are present. Local providers
        (no auth) are considered configured; reachability is checked lazily at call time."""
        if self.status != "active":
            return False
        return not self.missing_env()

    def resolved_base_url(self) -> str:
        """Substitute ${ENV_VAR} in base_url (e.g. Cloudflare's account id)."""
        def repl(m: re.Match) -> str:
            return os.environ.get(m.group(1), m.group(0))
        return _ENV_TEMPLATE.sub(repl, self.base_url)

    def api_key(self) -> Optional[str]:
        """The primary credential (first auth env var), or None for keyless/local."""
        if not self.auth_env:
            return None
        return os.environ.get(self.auth_env[0])

    def rpm(self) -> Optional[int]:
        return self.rate_limits.get("rpm")

    def rpd(self) -> Optional[int]:
        return self.rate_limits.get("rpd")

    def strengths(self) -> set[str]:
        """Union of every strength tag across this provider's known models."""
        tags: set[str] = set()
        for m in self.models:
            tags.update(m.strengths)
        return tags


# ── loading / validation ─────────────────────────────────────────────────────
def _provider_from_row(row: dict) -> Provider:
    auth = row.get("auth") or {}
    return Provider(
        id=row["id"],
        name=row.get("name", row["id"]),
        wire_format=row["wire_format"],
        base_url=row["base_url"],
        auth_scheme=auth.get("scheme", "bearer"),
        auth_env=list(auth.get("env", [])),
        default_model=row.get("default_model"),
        models=[ModelInfo(id=m["id"], strengths=list(m.get("strengths", [])))
                for m in row.get("models", [])],
        rate_limits=row.get("rate_limits") or {},
        priority=int(row.get("priority", 0)),
        models_path=row.get("models_path"),
        docs=row.get("docs", ""),
        source=row.get("source", ""),
        status=row.get("status", "active"),
        local=bool(row.get("local", False)),
    )


def validate_row(row: dict) -> list[str]:
    """Return a list of problems with a provider row; empty means valid.

    Used by the scout before it appends anything, so a malformed auto-discovered
    provider can never corrupt the file."""
    problems: list[str] = []
    for key in ("id", "name", "wire_format", "base_url"):
        if not row.get(key):
            problems.append(f"missing required field '{key}'")
    wf = row.get("wire_format")
    if wf and wf not in VALID_WIRE_FORMATS:
        problems.append(f"wire_format '{wf}' not in {sorted(VALID_WIRE_FORMATS)}")
    auth = row.get("auth") or {}
    if auth.get("scheme") not in {"bearer", "sdk", "none", None}:
        problems.append(f"bad auth.scheme '{auth.get('scheme')}'")
    if not isinstance(row.get("models", []), list):
        problems.append("'models' must be a list")
    return problems


def load_raw() -> dict:
    with REGISTRY_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def load_providers() -> list[Provider]:
    """All providers in the registry, active or not, sorted by descending priority."""
    data = load_raw()
    rows = data.get("providers", [])
    seen: set[str] = set()
    providers: list[Provider] = []
    for row in rows:
        problems = validate_row(row)
        if problems:
            # Skip invalid rows rather than crash the whole gateway.
            continue
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        providers.append(_provider_from_row(row))
    providers.sort(key=lambda p: p.priority, reverse=True)
    return providers


def configured_providers() -> list[Provider]:
    """Providers whose credentials are present — the ones the gateway can actually use."""
    return [p for p in load_providers() if p.is_configured]


def get_provider(provider_id: str) -> Optional[Provider]:
    for p in load_providers():
        if p.id == provider_id:
            return p
    return None


# ── scout write path (auto-apply additions / flag removals) ──────────────────
def _backup() -> None:
    shutil.copy2(REGISTRY_PATH, REGISTRY_PATH.with_suffix(".json.bak"))


def add_provider(row: dict) -> None:
    """Append a validated provider row (used by the scout to auto-apply additions).

    Raises ValueError if the row is invalid or the id already exists — the scout is
    only ever allowed to *add* new, well-formed rows this way."""
    problems = validate_row(row)
    if problems:
        raise ValueError(f"invalid provider row: {problems}")
    data = load_raw()
    if any(p["id"] == row["id"] for p in data.get("providers", [])):
        raise ValueError(f"provider '{row['id']}' already exists")
    _backup()
    data.setdefault("providers", []).append(row)
    _write(data)


def flag_provider(provider_id: str, reason: str) -> bool:
    """Mark a provider as flagged (status='flagged') without deleting it.

    Removals are never automatic — a human decides whether to drop a flagged row."""
    data = load_raw()
    changed = False
    for p in data.get("providers", []):
        if p["id"] == provider_id and p.get("status") != "flagged":
            p["status"] = "flagged"
            p["flag_reason"] = reason
            changed = True
    if changed:
        _backup()
        _write(data)
    return changed


def _write(data: dict) -> None:
    tmp = REGISTRY_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(REGISTRY_PATH)
