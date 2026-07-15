# ag23-llm

A task-routed gateway over many **free** LLM providers, plus a **scout** that keeps the
provider list fresh. Built by combining existing pieces rather than reinventing them.

This is a repo, not a pip library — clone it and run it. It has two clearly separated parts:

```
ag23_llm/   the gateway   — importable core: chat(), registry, config generation, task routing
scout/      the tool       — browses the web, discovers/verifies providers, rewrites the registry
```

The gateway is pure: `import ag23_llm` loads none of the scout. The scout is an application
that *imports* the gateway and operates on its data — it's kept out of the core on purpose.

| Concern | Owned by | Why |
| --- | --- | --- |
| Transport, fallback, retries, rate-limit cooldowns | [LiteLLM](https://github.com/BerriAI/litellm) | mature, 100+ providers |
| Which model for a task (clusters) | [Semantic Router](https://github.com/aurelio-labs/semantic-router) + `benchmarks.json` | keyless local embeddings |
| **Discovering new free providers** | `scout/` (this repo) | nothing off the shelf does it |
| Recording scout runs | [agent-trace-outcomes](https://github.com/Aryagarg23/agent-trace-outcomes) | optional, best-effort |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # add a key for any provider you have (Groq is the easiest free tier)
```

Every provider key is optional — unconfigured providers are skipped.

### Install it into another project (from GitHub, not PyPI)

Pip-installable straight from the repo — no PyPI. Add to a `requirements.txt`:

```
ag23-llm @ git+https://github.com/Aryagarg23/ag23-llm.git
```

or install directly:

```bash
pip install "git+https://github.com/Aryagarg23/ag23-llm.git"
pip install "ag23-llm[routing] @ git+https://github.com/Aryagarg23/ag23-llm.git"  # + embeddings
pip install -e .   # editable, from a local clone (live edits)
```

This installs the **gateway** (`import ag23_llm`, plus an `ag23-llm` CLI). The **scout** is
not part of the install — it's a repo-only tool (clone the repo and run `python -m scout`).

## Use the gateway

```bash
python -m ag23_llm providers                  # what's wired + configured
python -m ag23_llm route "solve this integral" # show the cluster/tags/benchmark order
python -m ag23_llm chat "hello" --task general
python -m ag23_llm config                      # dump the generated LiteLLM config.yaml
```

From code:

```python
from ag23_llm import chat
res = chat("summarize this", task="long-context")
print(res.text, res.provider, res.model)
```

`task=` is the seam for benchmark-cluster routing: it maps to a cluster (via Semantic
Router, or keywords as a fallback), and `benchmarks.json` ranks the free models in that
cluster. Today the gateway routes by tag + LiteLLM's rate-limit-aware strategy;
`ag23_llm/taskrouter.py:build_strategy()` is the LiteLLM `CustomRoutingStrategyBase` hook
to route by strict benchmark order once you trust measured evals.

## Run the scout

```bash
python -m scout --dry-run            # discover providers, report only, no writes
python -m scout                      # discover + verify + update ag23_llm/providers.json + PROVIDERS.md
python -m scout --verify-only        # just live-verify the configured providers
```

The scout reads the known free-LLM lists (cheahjs, awesome-freellm-apis), searches the web
beyond them, verifies candidates, **auto-applies** verified OpenAI-compatible providers, and
**flags** (never deletes) ones that stop responding. It records each run through
`agent-trace-outcomes` if that CLI is available (best-effort).

## How data flows

```
chat("fix this python", task="coding")
        │
        ▼
 taskrouter ──(Semantic Router / keywords)──▶ cluster "coding" ──▶ tags
        │                                         │
        │                              benchmarks.json ranks free models in-cluster
        ▼                                         │
 gateway → litellm.Router (group "free", tag-filtered) ──▶ provider call + fallback
        ▲
 providers.json ── config_gen ──┘        scout ──▶ updates providers.json + PROVIDERS.md
```

## Roadmap

- Replace the `benchmarks.json` seed (derived from `strengths` tags) with **measured** evals
  (promptfoo / lm-evaluation-harness), then enable `build_strategy()` for benchmark-ordered routing.
- Let the scout also refresh `benchmarks.json` from public leaderboards (LiveBench / Artificial Analysis).

## License

MIT — see [LICENSE](LICENSE).
