# Mem0 for Hermes — memory extraction on your auxiliary model (Codex * OpenRouter included)

[![License: Apache 2.0 OR MIT](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](#license)

**Mem0 memory for Hermes Agent, with every LLM call Mem0 makes routed through
Hermes's auxiliary model path instead of the OpenAI API** — so fact extraction
runs on whatever `hermes model` points at, including OAuth-only backends like
Codex that have no API key for Mem0 to use in the first place.

Mem0's OSS engine is good at the memory part — semantic search, dedup, an
add/update/delete decision pass — but every LLM call it makes goes through its
own `openai.OpenAI` client to `api.openai.com`, keyed off `OPENAI_API_KEY`. On
Claude, OpenRouter, Bedrock, or a local endpoint, that means memory generation is
the one component still billing a second provider. On **Codex it can't run at
all**: `openai-codex` authenticates with an OAuth token from Hermes's auth store
and credential pool, speaks the Responses API rather than Chat Completions, and
never produces an `OPENAI_API_KEY` for Mem0 to fall back on.

Routing Mem0 through `agent.auxiliary_client.call_llm` fixes both cases at once,
because that is the same path Hermes's own auxiliary tasks (compression, vision,
titles) already use — including its Codex Responses shim.

The plugin keeps the Mem0 engine and replaces just that client:

```
Mem0 Memory.add(infer=True)
  └─ self.llm.generate_response(...)          ← Mem0's extraction / update pass
       └─ HermesRoutedLLM                     ← this plugin (_hermes_llm.py)
            └─ agent.auxiliary_client.call_llm(task="mem0_hermes_extraction", …)
                 └─ your configured provider + model  (hermes model)
                    Codex OAuth · Anthropic · Bedrock · OpenRouter · local
                    auth, retries, fallbacks, usage accounting: all Hermes's
```

Because it's the standard auxiliary path, routed memory calls inherit Hermes's
credential pools and OAuth refresh, its Anthropic/Bedrock/Codex request shaping,
its transient-retry and provider-fallback chain, its usage/cost accounting, and
its `Auxiliary <task>: using <provider> (<model>)` log line.

## What still leaves Hermes

**Embeddings.** Hermes has no embedding path of its own, so the `embedder` block
is handed to Mem0 verbatim — it can't ride the auxiliary route. The default is
therefore **local**: a keyless default matters most for exactly the users this
plugin is for, since a Codex/OAuth setup has no `OPENAI_API_KEY` to fall back on.

| `embedder.provider` | Needs | Notes |
| --- | --- | --- |
| **`fastembed`** | nothing | **Default.** ONNX in-process; `BAAI/bge-small-en-v1.5`, 384 dims, 67 MB of weights on first use. Nothing leaves the machine |
| `ollama` | a running Ollama | `nomic-embed-text`, 768 dims |
| `openai` | `OPENAI_API_KEY` | `text-embedding-3-small`; embeddings only, never generation |
| `huggingface`, `azure_openai`, `gemini`, `lmstudio`, `together`, `aws_bedrock` | varies | Passed through to Mem0 |

`hermes memory setup` installs whichever embedder you pick, at the moment you
pick it — `fastembed` pulls `fastembed>=0.3.1` (matching mem0ai's own `extras`
constraint), `ollama` pulls `ollama`, `huggingface` pulls
`sentence-transformers`. Hosted embedders need no package. The install goes
through Hermes's own gated installer, so it honors
`security.allow_lazy_installs` and the durable-target redirect on sealed images.
Installing fastembed also switches Mem0's BM25 keyword search on, which it
otherwise skips.

Vector widths are read back from fastembed's own model registry after install
and written into the config, so a wrong dimension can never silently create a
mis-sized collection.

fastembed's weights are cached in `%LOCALAPPDATA%\fastembed` (Windows) or
`~/.cache/fastembed` — not in fastembed's default temp directory, where a disk
cleanup would delete them and trigger a silent re-download, and deliberately not
under `$HERMES_HOME`, which `hermes backup` archives. Set
`FASTEMBED_CACHE_PATH` yourself to override.

Everything else — the vector store, the history DB — is local by default under
`$HERMES_HOME/mem0_hermes/`.

## Install

```bash
hermes plugins install hermes-pro/mem0 --no-enable
hermes memory setup                      # pick mem0_hermes, choose an embedder
```

`hermes memory setup` installs `mem0ai` plus the embedder you select (the
default, `fastembed`, needs no API key), writes `memory.provider: mem0_hermes`
to `config.yaml`, and saves your answers to `$HERMES_HOME/mem0_hermes.json`.
Start a new session to activate. Accepting the defaults gets you working memory
with **no credentials of any kind** — extraction borrows your Hermes provider's
auth, embeddings run locally.

**Why `--no-enable`.** `hermes plugins install` ends with
`Enable 'mem0_hermes' now? [y/N]`, which manages the `plugins.enabled`
allow-list. Memory providers don't use that allow-list — they're all
discovered, and exactly one is activated by `memory.provider` — so the answer
has no effect either way; `--no-enable` just skips the prompt and keeps the
allow-list free of an entry that does nothing. `--enable` is equally harmless.
Nothing else is prompted: this plugin declares no `requires_env`, because it
needs no credentials of its own.

The repo is named `mem0` but installs as **`mem0_hermes`** — the destination
comes from `name` in `plugin.yaml`, not the repo. That distinction matters:
Hermes resolves `memory.provider` against bundled providers first, so a
provider directory named `mem0` would be shadowed by the bundled Mem0 plugin
and never load.

Updating and removing:

```bash
hermes plugins update mem0_hermes        # git pull in place
hermes plugins remove mem0_hermes
```

### Development install

`hermes plugins install` clones a snapshot from GitHub, so it can't see
uncommitted work. To point Hermes at a checkout instead:

```bash
git clone https://github.com/hermes-pro/mem0.git && cd mem0
python scripts/dev_link.py               # link $HERMES_HOME/plugins/mem0_hermes → here
python scripts/dev_link.py --activate    # also set memory.provider
python scripts/dev_link.py --dry-run     # show the plan, change nothing
python scripts/dev_link.py --uninstall   # remove the link again
```

It uses a symlink, or a Windows junction when symlinks need privileges, falling
back to a copy — so edits land in the next session with no reinstall. The
checkout directory name doesn't matter; the destination is always
`mem0_hermes`.

## Configure

`$HERMES_HOME/mem0_hermes.json` — every key is optional:

```jsonc
{
  "user_id": "hermes-user",        // "hermes-user" = defer to the gateway's id
  "agent_id": "hermes",
  "rerank": false,
  "telemetry": false,              // Mem0's PostHog telemetry, off by default

  "llm": {                         // the Hermes-routed generation settings
    "task": "mem0_hermes_extraction",
    "provider": "",                // "" = whatever `hermes model` is set to
    "model": "",                   // "" = your main chat model
    "base_url": "",
    "api_key": "",
    "temperature": 0.1,
    "max_tokens": null,            // null = provider default
    "timeout": 120,
    "json_mode": "prompt"          // prompt | response_format | off
  },

  "embedder": {                    // passed to Mem0 as-is; local by default
    "provider": "fastembed",
    "config": { "model": "BAAI/bge-small-en-v1.5", "embedding_dims": 384 }
  },
  "vector_store": {                // passed to Mem0 as-is
    "provider": "qdrant",
    "config": { "path": "<HERMES_HOME>/mem0_hermes/qdrant",
                "collection_name": "mem0_hermes" }
  },
  "history_db_path": "<HERMES_HOME>/mem0_hermes/history.db",
  "custom_instructions": null      // extra guidance for fact extraction
}
```

Config resolution, lowest precedence first: built-in defaults → an existing
bundled-`mem0` OSS setup in `mem0.json` (embedder + vector store + ids, so you
keep reading the same memories) → `mem0_hermes.json` → environment
(`MEM0_HERMES_LLM_MODEL`, `MEM0_HERMES_LLM_PROVIDER`, `MEM0_HERMES_USER_ID`,
`MEM0_HERMES_AGENT_ID`, `MEM0_HERMES_JSON_MODE`, …; the bundled `MEM0_USER_ID` /
`MEM0_AGENT_ID` are honored too).

### Choosing the extraction model

Leaving `llm.provider` and `llm.model` empty is the point of the plugin: the
auxiliary task is unconfigured, so `auxiliary_client` resolves it to your main
provider and model. To spend a cheaper model on extraction instead, either set
`llm.model` above, or pin the auxiliary task in `config.yaml`:

```yaml
auxiliary:
  mem0_hermes_extraction:
    provider: openrouter
    model: anthropic/claude-haiku-4.5
    timeout: 120
```

### `json_mode`

Mem0 asks for JSON and parses it back. Providers differ in how they'll agree:

- `prompt` (default) — send no `response_format`; rely on Mem0's "return only
  valid JSON" instructions plus local cleanup that strips code fences,
  `<think>` blocks and surrounding prose. Works on every provider.
- `response_format` — forward `{"type": "json_object"}` in the request body.
  Only for OpenAI-compatible endpoints that support JSON mode.
- `off` — return the model's text untouched (debugging).

Leave it on `prompt` for Codex. Hermes's Codex adapter translates the call into a
Responses API request, rebuilding the payload and forwarding only `timeout`,
`extra_body.reasoning` and `tools` — a `response_format` is silently dropped
rather than honored, so `prompt` mode is what actually keeps extraction parseable
there. Same reasoning for Anthropic and Bedrock.

### Reranking (advanced)

A `reranker` block is passed to Mem0 verbatim. Mem0 builds LLM rerankers with
the same factory this plugin registers into, so a reranker can be Hermes-routed
too:

```jsonc
"rerank": true,
"reranker": {
  "provider": "llm_reranker",
  "config": { "llm": { "provider": "hermes_routed", "config": { "max_tokens": 256 } } }
}
```

## Verify it's working

1. `hermes memory status` lists `mem0_hermes` as the active provider, and the
   session's system prompt carries a `# Mem0 Memory (Hermes-routed)` block
   naming the route.
2. After a turn, the log shows Hermes's own auxiliary line for the extraction
   call: `Auxiliary mem0_hermes_extraction: using <provider> (<model>)`, plus
   `mem0_hermes: memory extraction routed through …` at startup.
3. Ask the agent to recall something from a previous session; `mem0_search`
   should return it.

Tools exposed to the model (same names as the bundled plugin, so prompting and
habits carry over): `mem0_search`, `mem0_add`, `mem0_update`, `mem0_delete`.
`mem0_add` stores verbatim (`infer=False`) and spends no LLM call; turn-level
extraction is what runs on your model, in a background thread after each turn.

## Migrating from the bundled `mem0` plugin

Both can sit on disk; only the provider named in `memory.provider` is active.
If you were running bundled `mem0` in **OSS** mode, this plugin inherits its
`embedder`, `vector_store` and ids from `mem0.json`, so existing memories keep
working — switch `memory.provider` and you're done. Coming from **platform**
mode (`MEM0_API_KEY`), memories live on Mem0's servers and do not transfer;
this plugin is OSS-only by design, since server-side extraction is exactly what
it replaces.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `backend not initialized: embedder provider 'openai' needs OPENAI_API_KEY` | You switched off the local default. Set the key, or `hermes memory setup` → embedder `fastembed`. |
| `embedder 'fastembed' needs fastembed>=0.3.1` | Its package went missing — usually a rebuilt venv after `hermes update`. The next session reinstalls it automatically; if installs are gated off (`security.allow_lazy_installs: false`), `pip install fastembed` or re-run `hermes memory setup`. |
| `fastembed does not offer model '…'` | Typo in `embedder.config.model`. The message lists valid names; setup checks this against fastembed's registry. |
| `Hermes-routed memory LLM call failed …` | The routed provider rejected the call. The message includes the route; check `hermes model`. |
| `… returned an empty response` | The model produced nothing (often a reasoning model that spent its budget). Raise `llm.max_tokens` or pin a different `llm.model`. |
| `circuit breaker tripped after 5 consecutive failures` | Five failures in a row pause memory calls for 120s. Check the routed model and the vector store. |
| Memory tools missing entirely | `memory.provider` not set to `mem0_hermes`, or the plugin isn't in `$HERMES_HOME/plugins/`. |
| Vector dimension errors after switching embedder | Handled automatically for local Qdrant (the collection is recreated); other stores need a manual reset. |

## Layout

The plugin **is** the repo root, so `hermes plugins install hermes-pro/mem0`
finds `plugin.yaml` where it looks for it (and `hermes plugins update` can pull
in place, since the clone's `.git` comes along).

```
plugin.yaml            # manifest: name: mem0_hermes, kind: exclusive, mem0ai dep
__init__.py            # MemoryProvider: lifecycle, tools, prefetch, breaker
_hermes_llm.py         # HermesRoutedLLM → agent.auxiliary_client.call_llm
_backend.py            # builds Mem0 Memory with the routed LLM injected
_config.py             # config resolution, wizard schema, save_config
scripts/dev_link.py    # link a checkout into $HERMES_HOME/plugins (dev)
tests/                 # unittest suite (no pytest needed)
```

Subdirectories are inert to Hermes: the memory loader only pre-registers
top-level `*.py`, and the general plugin scanner never descends into a directory
that already has a `plugin.yaml`. So `tests/` and `scripts/` ship along
harmlessly and are never imported at load time.

`_hermes_llm.py` deliberately imports no Mem0 symbols at module level: Hermes's
plugin loader executes each submodule before `mem0ai` is guaranteed to be
installed, and a failed module-level import would leave a half-initialized
module in `sys.modules`. Mem0 only duck-types its LLM object, so a plain class
with `.config` and `.generate_response()` is enough.

## Tests

```powershell
& "$env:LOCALAPPDATA\hermes\hermes-agent\venv\Scripts\python.exe" -m unittest discover -s tests -t tests -v
```

The suite finds hermes-agent at `./hermes-agent` (a gitignored sibling checkout)
or wherever `HERMES_AGENT_DIR` points.

Use the Hermes venv's interpreter: with `mem0ai` and `qdrant-client` importable,
the suite also builds a real Mem0 `Memory` over a temporary Qdrant collection
and asserts extraction actually flows through the routed adapter, and loads the
plugin through Hermes's own `plugins.memory` loader. Those tests skip on a bare
interpreter; the rest still run.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, the test command, and the
handful of non-obvious constraints (chiefly: `_hermes_llm.py` must not import
Mem0 at module level). Opening a pull request accepts the
[Contributor License Agreement](CLA.md) — a license grant, not a copyright
assignment; you keep ownership of your work. Issues and reviews need no CLA.

## License

Copyright (c) 2026 Hermes Pro.

Dual-licensed under **[Apache License 2.0](LICENSE-APACHE)** or the
**[MIT License](LICENSE-MIT)**, at your option:

```
SPDX-License-Identifier: Apache-2.0 OR MIT
```

Take whichever fits your project — you don't need to satisfy both. Apache-2.0
adds an explicit patent grant and a NOTICE/attribution requirement; MIT is
shorter and permissive. Contributions come in under the same dual license.

Hermes Agent ([MIT](https://github.com/NousResearch/hermes-agent)) and Mem0
([Apache-2.0](https://github.com/mem0ai/mem0)) are dependencies, not vendored
here; their own licenses govern their code.
