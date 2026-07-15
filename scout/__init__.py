"""ag23-llm provider scout — a maintenance TOOL, not part of the importable gateway.

It browses free-LLM lists + the web, verifies providers, and updates the gateway's
registry (ag23_llm/providers.json) through the core's write API. Run it deliberately:
    python -m scout --dry-run
It is never imported by `ag23_llm` itself.
"""
