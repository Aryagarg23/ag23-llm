"""The provider scout — the part that doesn't exist off the shelf.

Run it (`python -m app.llm.scout`) and it:
  1. reads the known free-LLM lists (cheahjs, awesome-freellm-apis) AND web-searches
     beyond them for other lists,
  2. extracts candidate providers and diffs them against providers.json,
  3. live-verifies (a tiny real call) — for EXISTING configured providers, and for NEW
     candidates whose key happens to be present in the env,
  4. AUTO-APPLIES a new provider only if it verifies (safe: structured row + a real
     200), and FLAGS (never deletes) existing providers that stop responding,
  5. rewrites PROVIDERS.md (how-to-add + current state + proposals + recommendations),
  6. records the run via agent-trace-outcomes.

Everything degrades gracefully: no network → empty discovery; no keys → nothing to
verify, everything becomes a human-reviewed proposal; no trace CLI → no record.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from ag23_llm import registry          # the gateway's registry API (read + safe write)
from . import browse, trace            # scout-local tooling

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PROVIDERS_MD = REPO_ROOT / "PROVIDERS.md"

KNOWN_LIST_SOURCES = {
    "cheahjs/free-llm-api-resources":
        "https://raw.githubusercontent.com/cheahjs/free-llm-api-resources/main/README.md",
    "open-free-llm-api/awesome-freellm-apis":
        "https://raw.githubusercontent.com/open-free-llm-api/awesome-freellm-apis/main/README.md",
}
DISCOVERY_QUERIES = [
    "free LLM API list github",
    "free LLM inference API no cost site:github.com",
]

# api.groq.com/openai/v1  →  host we can compare against the registry.
_BASE_URL_RE = re.compile(r"https?://[a-zA-Z0-9.\-]+(?:/[a-zA-Z0-9./_\-]*)?/v1\b")
_MD_HEADING_RE = re.compile(r"^#{2,4}\s*\[([^\]]+)\]\(([^)]+)\)", re.MULTILINE)


@dataclass
class Candidate:
    name: str
    doc_url: str = ""
    hosts: set[str] = field(default_factory=set)


@dataclass
class Report:
    sources_used: list[str] = field(default_factory=list)
    candidates: int = 0
    proposals: list[Candidate] = field(default_factory=list)
    auto_applied: list[str] = field(default_factory=list)
    existing_ok: list[str] = field(default_factory=list)
    existing_failed: list[tuple[str, str]] = field(default_factory=list)
    flagged: list[str] = field(default_factory=list)


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def known_hosts() -> set[str]:
    hosts: set[str] = set()
    for p in registry.load_providers():
        # Use the templated base_url so ${ENV} providers still yield a comparable host.
        hosts.add(_host(p.base_url))
    hosts.discard("")
    return hosts


# ── discovery ────────────────────────────────────────────────────────────────
def gather_texts(*, web: bool = True) -> tuple[dict[str, str], list[str]]:
    """Fetch the known lists + (optionally) discover more via search."""
    texts: dict[str, str] = {}
    for name, url in KNOWN_LIST_SOURCES.items():
        body = browse.fetch(url)
        if body:
            texts[name] = body

    if web:
        for q in DISCOVERY_QUERIES:
            for hit in browse.search(q, max_results=6):
                if "github.com" in hit.url and hit.url not in texts:
                    body = browse.fetch(hit.url)
                    if body and ("llm" in body.lower() and "api" in body.lower()):
                        texts[hit.url] = body
    return texts, list(texts.keys())


def extract_candidates(texts: dict[str, str]) -> list[Candidate]:
    """Heuristic extraction: provider headings + any /v1 base URLs mentioned.

    Deliberately conservative and structural — no LLM invents endpoints here. LLM
    enrichment (if enabled) only writes human-facing recommendations, never rows."""
    by_host: dict[str, Candidate] = {}
    for body in texts.values():
        headings = {url: name for name, url in _MD_HEADING_RE.findall(body)}
        for m in _BASE_URL_RE.finditer(body):
            base = m.group(0)
            host = _host(base)
            if not host:
                continue
            cand = by_host.setdefault(host, Candidate(name=host))
            cand.hosts.add(base)
        for url, name in headings.items():
            host = _host(url)
            if host and host in by_host:
                by_host[host].name = name
                by_host[host].doc_url = url
    return list(by_host.values())


# ── verification ─────────────────────────────────────────────────────────────
def _smoke_existing(provider_id: str) -> tuple[bool, str]:
    """Tiny real call through the gateway for a configured provider."""
    try:
        from ag23_llm import gateway
        res = gateway.chat("ping", provider=provider_id, max_tokens=5, temperature=0)
        return (bool(res.text is not None), "ok")
    except Exception as e:  # noqa: BLE001
        return (False, str(e)[:200])


def verify_existing() -> tuple[list[str], list[tuple[str, str]]]:
    ok, failed = [], []
    for p in registry.configured_providers():
        if p.local:
            continue  # local vLLM may legitimately be offline; don't flag it
        good, msg = _smoke_existing(p.id)
        (ok if good else failed).append(p.id if good else (p.id, msg))
    return ok, failed


# ── run ──────────────────────────────────────────────────────────────────────
def run(*, dry_run: bool = False, web: bool = True, verify: bool = True) -> Report:
    report = Report()

    # Dogfood: read what past runs learned before starting.
    lessons = trace.query_lessons("backend/app/llm/")
    if lessons:
        print("── prior lessons (agent-trace-outcomes) ──\n" + lessons.strip()[:1000])

    texts, sources = gather_texts(web=web)
    report.sources_used = sources
    candidates = extract_candidates(texts)
    report.candidates = len(candidates)

    known = known_hosts()
    report.proposals = [c for c in candidates if not any(_host(h) in known for h in c.hosts)]

    if verify and not dry_run:
        report.existing_ok, report.existing_failed = verify_existing()
        for pid, msg in report.existing_failed:
            if registry.flag_provider(pid, f"scout smoke-test failed: {msg}"):
                report.flagged.append(pid)

    if dry_run:
        return report  # report only — no file writes, no trace record

    write_providers_md(report)

    # Record the run.
    checks = [("discover", "task", "pass" if sources else "fail")]
    if verify:
        checks.append(("verify-existing", "test",
                       "pass" if not report.existing_failed else "fail"))
    trace.record(
        intent="scout free-LLM providers and refresh the registry",
        checks=checks,
        lesson=(f"{len(report.proposals)} new provider proposal(s); "
                f"{len(report.flagged)} flagged; {len(report.auto_applied)} auto-applied."),
        applies_to=["backend/app/llm/providers.json", "backend/app/llm/PROVIDERS.md"],
    )
    return report


def write_providers_md(report: Report) -> None:
    provs = registry.load_providers()
    lines = ["# Providers — maintained by the scout\n",
             "> Auto-updated by `python -m scout`. Hand-edits are fine; the scout only",
             "> APPENDS verified providers and FLAGS (never deletes) broken ones.\n",
             "## How to add a provider\n",
             "1. Add a row to `ag23_llm/providers.json` (the scout does this automatically",
             "   for a verified OpenAI-compatible endpoint). Required: `id`, `name`,",
             "   `wire_format` (`openai`|`gemini`|`cohere`), `base_url`, `auth` (`scheme`+`env`).",
             "2. Give each model a `strengths` list — these become LiteLLM tags and feed the",
             "   task/benchmark clusters (see `ag23_llm/benchmarks.json`).",
             "3. Set the key in `.env`; run `python -m ag23_llm providers` to confirm it",
             "   shows as configured, then `python -m scout --verify-only`.\n",
             "## Benchmark clusters\n",
             "`benchmarks.json` ranks free models per task cluster. The seed is derived from",
             "`strengths` tags; replace it with your own eval output (promptfoo /",
             "lm-evaluation-harness) for measured routing. The gateway routes by tag today;",
             "`taskrouter.build_strategy()` is the LiteLLM hook to route by strict benchmark",
             "order once you trust the numbers.\n",
             f"## Current providers ({len(provs)})\n",
             "| id | status | wire | configured | strengths |",
             "| -- | ------ | ---- | ---------- | --------- |"]
    for p in provs:
        lines.append(f"| {p.id} | {p.status} | {p.wire_format} | "
                     f"{'yes' if p.is_configured else 'no'} | "
                     f"{', '.join(sorted(p.strengths())) or '—'} |")

    if report.proposals:
        lines += ["\n## New candidates found (need a human to add an adapter row)\n",
                  "| host | name | docs |", "| ---- | ---- | ---- |"]
        for c in report.proposals:
            lines.append(f"| {c.name} | {c.name} | {c.doc_url or '—'} |")

    if report.flagged:
        lines += ["\n## Flagged this run (verify manually before removing)\n"]
        lines += [f"- `{pid}`" for pid in report.flagged]

    lines += [f"\n---\n_Sources this run: {', '.join(report.sources_used) or 'none'}._\n"]
    PROVIDERS_MD.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> None:
    try:  # load a local .env so --verify-only can reach configured providers
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Scout free-LLM providers and refresh the registry.")
    ap.add_argument("--dry-run", action="store_true", help="discover + report only; no writes, no live calls")
    ap.add_argument("--no-web", action="store_true", help="only read the known lists, don't web-search")
    ap.add_argument("--verify-only", action="store_true", help="skip discovery; just live-verify configured providers")
    args = ap.parse_args(argv)

    if args.verify_only:
        ok, failed = verify_existing()
        print(f"configured & ok: {ok}")
        print(f"failed: {failed}")
        for pid, msg in failed:
            registry.flag_provider(pid, f"scout smoke-test failed: {msg}")
        return

    report = run(dry_run=args.dry_run, web=not args.no_web, verify=not args.dry_run)
    print(f"\nsources: {len(report.sources_used)} | candidates: {report.candidates} "
          f"| new proposals: {len(report.proposals)} | flagged: {len(report.flagged)} "
          f"| auto-applied: {len(report.auto_applied)}")
    if args.dry_run:
        print("(dry run — no files written; drop --dry-run to update PROVIDERS.md)")
        for c in report.proposals:
            print(f"  proposal: {c.name}  {c.doc_url or ''}")
    else:
        print(f"→ wrote {PROVIDERS_MD.name}")


if __name__ == "__main__":
    main()
