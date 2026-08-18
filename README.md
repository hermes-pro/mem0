# Mem0 for Hermes — memory extraction on your auxiliary model (Codex * OpenRouter included)

[![License: Apache 2.0 OR MIT](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](#license)

**Mem0 memory for Hermes Agent, with every LLM call Mem0 makes routed through
Hermes's auxiliary model path instead of the OpenAI API** — so fact extraction
runs on whatever `hermes model` points at, including OAuth-only backends like
Codex that have no API key for Mem0 to use in the first place.

## Install

```bash
hermes plugins install hermes-pro/mem0 --no-enable
hermes memory setup                      # pick mem0_hermes, choose an embedder
```

`hermes memory setup` installs `mem0ai` plus the embedder you select (the
default, `fastembed`, runs locally), your answers saved to `$HERMES_HOME/mem0_hermes.json`.  
  
Start a new session to activate.

## Updating and uninstalling:

```bash
hermes plugins update mem0_hermes        # git pull in place
hermes plugins remove mem0_hermes
```

## Notes

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
  "concurrency": {                 // see "Multiple Hermes processes" below
    "lease_local_store": true,     // release the embedded-Qdrant lock between calls
    "lease_idle_release": false,   // opt in: hold it briefly instead of per call
    "lease_idle_seconds": 2.0,
    "lock_retries": 5,
    "lock_retry_backoff": 0.25,
    "sqlite_wal": true,            // defer to Hermes's WAL policy for history.db
    "sqlite_busy_timeout_ms": 15000,
    "background_init": true        // build the backend off the startup path
  },
  "history_db_path": "<HERMES_HOME>/mem0_hermes/history.db",
  "custom_instructions": null      // extra guidance for fact extraction
}
```

Config resolution, lowest precedence first: built-in defaults → an existing
bundled-`mem0` OSS setup in `mem0.json` (embedder + vector store + ids, so you
keep reading the same memories) → `mem0_hermes.json` → environment.

Every `llm` key has an environment override, so the whole extraction route can
be set from a shell profile or a unit file without writing the config file:

| Variable | Sets |
| --- | --- |
| `MEM0_HERMES_USER_ID` | `user_id` (the bundled `MEM0_USER_ID` is honored too) |
| `MEM0_HERMES_AGENT_ID` | `agent_id` (likewise `MEM0_AGENT_ID`) |
| `MEM0_HERMES_LLM_PROVIDER` | `llm.provider` |
| `MEM0_HERMES_LLM_MODEL` | `llm.model` |
| `MEM0_HERMES_LLM_BASE_URL` | `llm.base_url` |
| `MEM0_HERMES_LLM_API_KEY` | `llm.api_key` |
| `MEM0_HERMES_LLM_TASK` | `llm.task` |
| `MEM0_HERMES_JSON_MODE` | `llm.json_mode` |
| `MEM0_HERMES_LLM_TEMPERATURE` | `llm.temperature` |
| `MEM0_HERMES_LLM_MAX_TOKENS` | `llm.max_tokens` |
| `MEM0_HERMES_LLM_TIMEOUT` | `llm.timeout` |

The three numeric ones are parsed as numbers; an unparseable value is logged and
ignored rather than silently taking effect as the default.

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

## What blocks, and for how long

Hermes calls `initialize()` inline on the session-startup path and dispatches
memory tool calls inline on the agent loop, so both are worth knowing the cost
of. Measured on this machine with the fastembed default:

| Path | Cost | Notes |
| --- | --- | --- |
| `initialize()` | **2 ms** | The backend builds on a background thread. Building it inline costs ~1 511 ms — mostly *importing* `qdrant_client` (1 252 ms, almost all of it pydantic model construction) and `fastembed` (693 ms). Loading the embedding model itself is only ~199 ms, and the tokenizer inside it ~15 ms |
| `system_prompt_block()` | 0 ms | Never touches the backend |
| First turn's `prefetch()` | up to the build (~1.5 s), capped at 3 s | Waits for the backend so the first turn still gets recall |
| Later `prefetch()` | ~22 ms, off the hot path | Started at `on_turn_start`, consumed when the turn needs it |
| `mem0_add` / `mem0_search` tool call | ~15–25 ms | Runs inline on the agent loop. Over half of it is reopening the leased store, which grows with the store — see [`lease_idle_release`](#lease_idle_release--paying-the-reopen-once-per-turn) |
| `sync_turn` (extraction) | 0 ms on the loop | Queued to a single background worker; the LLM call happens there. Turns that arrive mid-extraction wait their turn rather than being dropped, up to a backlog of 8 |
| `recall_status()` | 0 ms | Reports what the last prefetch injected, so the agent can show `🧠 Mem0 — recalled N memories` without relying on the model to mention it |

The build isn't free, it's just moved somewhere it overlaps with the user typing
their first message. Worst case — a message submitted the instant the session
starts — it's paid once at the first prefetch, bounded by that 3 s cap, and
`mem0_search` remains available afterwards as the backstop.

`mem0_add` stays synchronous deliberately. At ~20 ms it's noise next to a model
round-trip, and making it fire-and-forget would mean a `mem0_search` immediately
after couldn't see what was just stored. `concurrency.background_init: false`
restores the old inline build if you'd rather pay it at startup.

One case can exceed these numbers: if another process holds the embedded Qdrant
store, a tool call waits out the lease retry (up to ~6 s by default) before
reporting. See below.

## Multiple Hermes processes

`vector_store.provider: qdrant` with a `path` is **not** a Qdrant server — it's
qdrant-client's embedded `QdrantLocal`, which takes an exclusive OS lock on
`<path>/.lock` for the lifetime of the client object. Anyone else gets:

```
RuntimeError: Storage folder <path> is already accessed by another instance of
Qdrant client. If you require concurrent access, use Qdrant server instead.
```

The lock is deliberate: QdrantLocal loads collection state into the owning
process, so two live owners would each write from their own stale snapshot. But
Hermes routinely has more than one candidate owner — CLI, gateway, Desktop, cron,
or two provider objects in one process.

This plugin handles that in three ways:

1. **One owner per store, per process.** Providers share a single backend keyed
   by storage path, so the plugin can't fight itself. Recall and writes stay
   correctly scoped because `user_id`/`agent_id` are per call.
2. **Leasing (default on).** The lock is taken and released around each *storage
   call* rather than held for the process's lifetime, so cooperating Hermes
   processes interleave. The granularity is deliberate: it's what a Qdrant server
   gives you — individual requests are serialized, not your whole
   read-think-write cycle — and crucially the lock is *not* held during Mem0's
   extraction LLM call, which would otherwise block every other process for
   seconds each turn.
3. **Bounded retry** (`lock_retries`, `lock_retry_backoff`) at both construction
   and each call, so a brief overlap waits instead of failing.

Cost of leasing is one store reopen per call: measured ~10 ms at 100 memories,
~45 ms at 1 000, ~180 ms at 5 000. That cost is per storage call and grows with
the store, so on a large store it dominates every memory tool call.

### `lease_idle_release` — paying the reopen once per turn

Off by default. When on, the store stays open for `lease_idle_seconds` (2 s)
after the last storage call instead of closing immediately, so a turn's
prefetch, tool calls and extraction write share one open:

| Store size | Default | `lease_idle_release: true` |
| --- | ---: | ---: |
| ~100 memories | 24.7 ms/add | **9.0 ms/add** |
| ~1 000 memories | 76.1 ms/add | **9.0 ms/add** |

The per-call cost stops scaling with the store, because there is no reopen.
What you give up:

- Another Hermes process may wait up to the idle window (plus its retry backoff)
  to get in, rather than slipping between calls.
- The store stays open into the *start* of Mem0's extraction LLM call — until
  the window lapses — instead of being released before it.

So it's the right switch for one busy process with a large store, and the wrong
one if several Hermes processes are genuinely interleaving. `lease_local_store:
false` remains the extreme of the same trade: fastest, but the process owns the
directory for its lifetime.

**What this cannot fix:** a process that holds the directory and never leases —
the bundled `mem0` plugin, or this plugin with leasing off. Leasing only works
when every participant leases. For genuinely concurrent access, run a Qdrant
server and point the config at it:

```jsonc
"vector_store": {
  "provider": "qdrant",
  "config": { "url": "http://localhost:6333", "collection_name": "mem0_hermes" }
}
```

A `url` (or `host`) disables leasing and sharing automatically — the server owns
concurrency.

**Never delete `.lock` to "fix" this.** The lock lives on the open file handle,
not on the file; deleting it while a process holds the store permits concurrent
writers and real corruption. Close the other process instead.

### The history database

`history.db` (Mem0's record of memory changes) is the other file every process
opens. Mem0 connects with `sqlite3.connect(path, check_same_thread=False)` and
nothing more, which leaves two defaults in place:

- **A 5 s busy timeout.** Mem0's writes are short — `BEGIN`, one INSERT,
  `COMMIT` — so contention is normally absorbed; measured here at 1 200 writes
  across 3 processes with zero failures and a 549 ms worst case. But 5 s is a
  cliff: past it the write raises and takes the memory operation with it. Raised
  to 15 s (`concurrency.sqlite_busy_timeout_ms`), which costs nothing
  uncontended.
- **Journal mode**, which decides whether a writer blocks readers.

Journal mode is delegated to Hermes's own `hermes_state.apply_wal_with_fallback`
rather than set here, so this database follows the same policy as Hermes's
`state.db`: WAL where it helps, DELETE on filesystems that reject WAL (NFS, SMB,
some FUSE), and **no WAL on SQLite builds carrying the WAL-reset corruption bug**
(3.7.0–3.51.2; fixed in 3.51.3, backported to 3.50.7 and 3.44.6). On such a build
you'll see one warning per process and `journal_mode=delete` — enabling WAL
ourselves would hand multi-process users a corruption risk Hermes deliberately
refuses elsewhere. `hermes update` can repair the embedded runtime;
`concurrency.sqlite_wal: false` opts out of touching journal mode at all.

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

Tools exposed to the model (the bundled plugin's four names are unchanged, so
prompting and habits carry over): `mem0_search`, `mem0_get_all`, `mem0_add`,
`mem0_update`, `mem0_delete`. `mem0_add` stores verbatim (`infer=False`) and
spends no LLM call; turn-level extraction is what runs on your model, in a
background worker after each turn.

`mem0_get_all` lists memories without a query — what you want when the user asks
"what do you know about me?" or wants to prune. It takes `limit` (default 20,
max 100) and `offset`, and reports `has_more`. Paging is sliced client-side:
Mem0's `get_all` accepts only `top_k` and its Qdrant store discards the
`next_page_offset` that `scroll` returns, so each page over-fetches `offset`
extra rows and a write landing between two pages can shift the window. Fine for
reviewing a store; don't build a paginated UI on it.

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
| `the local Qdrant store at … is held by another Qdrant client` | Another process owns the embedded store and isn't releasing it — often the bundled `mem0` plugin, or a Hermes instance with `lease_local_store: false`. Close it, or move to a Qdrant server. See [Multiple Hermes processes](#multiple-hermes-processes). Don't delete `.lock`. |
| `sqlite3.OperationalError: database is locked` | A history write waited past `concurrency.sqlite_busy_timeout_ms` (15 s). Something is holding a long write transaction on `history.db` — look for a stuck process rather than raising the timeout further. |
| `linked SQLite … is vulnerable to the WAL-reset corruption bug` | Informational, once per process: WAL was declined and `journal_mode=delete` used instead. `hermes update` repairs the embedded runtime. |
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
