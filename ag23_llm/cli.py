"""Command line for the gateway: `python -m ag23_llm <cmd>`.

    python -m ag23_llm providers                 # what's wired + configured
    python -m ag23_llm route "fix this python"   # show cluster/tags a query maps to
    python -m ag23_llm chat "hello" --task general
    python -m ag23_llm config                     # dump the generated LiteLLM config.yaml

Provider discovery is a separate tool (it browses the web + mutates the registry — not
part of the importable gateway): `python -m scout --dry-run`.
"""
from __future__ import annotations

import argparse
import sys

from . import registry


def _cmd_providers(_args) -> int:
    provs = registry.load_providers()
    configured = [p for p in provs if p.is_configured]
    print(f"{len(provs)} providers, {len(configured)} configured\n")
    for p in provs:
        mark = "*" if p.is_configured else "-"
        miss = "" if p.is_configured else f"  (needs {', '.join(p.missing_env()) or 'reachable endpoint'})"
        print(f" {mark} {p.id:<20} {p.status:<8} {p.wire_format:<7}{miss}")
    if not configured:
        print("\nNo providers configured. Add a key in backend/.env — see .env.example.")
    return 0


def _cmd_route(args) -> int:
    from . import taskrouter
    q = args.query
    cluster = taskrouter.canonical_cluster(q) or taskrouter.classify(q)
    print(f"query:   {q}")
    print(f"cluster: {cluster or '(none — would route across all)'}")
    if cluster:
        print(f"tags:    {taskrouter.tags_for_task(cluster)}")
        ranked = taskrouter.ranked_ids_for(cluster)
        print("benchmark order (seed):")
        for i, mid in enumerate(ranked, 1):
            print(f"  {i}. {mid}")
    return 0


def _cmd_chat(args) -> int:
    from .types import LLMError
    from . import gateway
    try:
        res = gateway.chat(args.prompt, task=args.task, provider=args.provider,
                           model=args.model, temperature=args.temperature,
                           max_tokens=args.max_tokens)
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(res.text)
    print(f"\n-- {res.provider} | {res.model} | {res.usage.total_tokens} tok", file=sys.stderr)
    return 0


def _cmd_config(_args) -> int:
    from . import config_gen
    try:
        print(config_gen.to_yaml())
    except ImportError:
        print("PyYAML not installed; showing model_list as repr:\n", file=sys.stderr)
        import json
        print(json.dumps(config_gen.build_model_list(use_env_ref=True), indent=2))
    return 0


def main(argv=None) -> None:
    # Windows consoles default to cp1252; LLM output and status glyphs are UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    try:  # load a local .env of provider keys if python-dotenv is present
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "scout":
        # The scout is a separate tool, not part of the gateway. Nudge, don't import.
        print("The scout is a separate tool. Run it from the repo root:\n"
              "  python -m scout --dry-run", file=sys.stderr)
        sys.exit(2)

    ap = argparse.ArgumentParser(prog="ag23_llm", description="Free-LLM gateway CLI.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("providers", help="list providers and whether they're configured").set_defaults(fn=_cmd_providers)

    p_route = sub.add_parser("route", help="show the cluster/tags a query maps to")
    p_route.add_argument("query")
    p_route.set_defaults(fn=_cmd_route)

    p_chat = sub.add_parser("chat", help="run a chat completion")
    p_chat.add_argument("prompt")
    p_chat.add_argument("--task", help="capability hint: coding|reasoning|math|... or free text")
    p_chat.add_argument("--provider", help="force a provider id from providers.json")
    p_chat.add_argument("--model", help="force a model id (with --provider) or full LiteLLM string")
    p_chat.add_argument("--temperature", type=float, default=0.7)
    p_chat.add_argument("--max-tokens", type=int, default=None, dest="max_tokens")
    p_chat.set_defaults(fn=_cmd_chat)

    sub.add_parser("config", help="print the generated LiteLLM config.yaml").set_defaults(fn=_cmd_config)

    args = ap.parse_args(argv)
    if not hasattr(args, "fn"):
        ap.error("unknown command")
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
