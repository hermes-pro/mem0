# AGENTS.md

Guidance for coding agents working in this repository. Human-facing docs:
[`README.md`](README.md) (behavior, configuration, performance) and
[`CONTRIBUTING.md`](CONTRIBUTING.md) (legal, setup, PR expectations).

## What this repo is

`mem0_hermes` — a Hermes Agent **memory provider plugin**. It runs the Mem0 OSS
engine (vector store, dedup, semantic search) but replaces Mem0's built-in
`openai.OpenAI` client with `agent.auxiliary_client.call_llm`, so fact
extraction and the add/update/delete decision pass run on whatever `hermes
model` points at — including OAuth-only backends like Codex that never produce
an `OPENAI_API_KEY`. Embeddings are the one thing that still leaves Hermes
(`embedder` is handed to Mem0 verbatim; default `fastembed`, local, keyless).

**The plugin is the repo root.** There is no `src/`, no package directory, no
build step. `plugin.yaml` sits at the root so `hermes plugins install
hermes-pro/mem0` finds it.

## Layout

```
plugin.yaml            # manifest: name: mem0_hermes, kind: exclusive, mem0ai dep
__init__.py            # Mem0HermesMemoryProvider: lifecycle, tools, prefetch, breaker
_hermes_llm.py         # HermesRoutedLLM -> agent.auxiliary_client.call_llm
_backend.py            # builds Mem0 Memory with the routed LLM; Qdrant leasing, history.db
_config.py             # config resolution, env overrides, wizard schema, save_config
scripts/dev_link.py    # link a checkout into $HERMES_HOME/plugins (dev only)
tests/                 # stdlib unittest suite; tests/_bootstrap.py does the import wiring
.github/workflows/     # CI: the suite on Linux/macOS/Windows x Python 3.10-3.13
```

`tests/` and `scripts/` are inert to Hermes: the memory loader pre-registers
only top-level `*.py`, and the plugin scanner never descends into a directory
that already holds a `plugin.yaml`.

## Setup

The suite imports `agent.*` and `plugins.*` from a hermes-agent checkout:

```bash
git clone https://github.com/NousResearch/hermes-agent.git   # gitignored sibling
python scripts/dev_link.py --activate                        # link into $HERMES_HOME
```

`HERMES_AGENT_DIR` overrides the `./hermes-agent` default. `dev_link.py` links
rather than copies, so edits apply on the next session with no reinstall.

## Test commands

Run with the **Hermes venv's interpreter** — that is where `mem0ai` and
`qdrant-client` live, and their presence is what enables the integration tests:

```bash
"$HOME/.local/share/hermes/hermes-agent/venv/bin/python" -m unittest discover -s tests -t tests -v
```

```powershell
& "$env:LOCALAPPDATA\hermes\hermes-agent\venv\Scripts\python.exe" -m unittest discover -s tests -t tests -v
```

Single module or case, from the repo root:

```bash
python -m unittest discover -s tests -t tests -v -k UpdateSignatureTests
```

A bare interpreter still runs the pure-logic tests; the Mem0/Qdrant integration
classes and the `plugins.memory` loader tests skip themselves. CI installs
nothing, so **CI never covers the Mem0 integration path** — run it locally in a
Hermes venv before claiming a backend change works.

There is no linter, formatter, or type checker configured. Do not add one, and
do not reformat untouched code.

## Hard constraints

Breaking any of these produces a failure that does not show up in the file you
edited.

- **`_hermes_llm.py` must not import Mem0 at module level.** Hermes's loader
  execs each top-level `*.py` before `mem0ai` is guaranteed installed and
  registers the module in `sys.modules` first, so a failed module-level import
  leaves a half-initialized module behind. Mem0 duck-types its LLM object; keep
  Mem0 imports inside functions.
- **`name: mem0_hermes` in `plugin.yaml` is load-bearing.** It decides the
  installed directory name. Hermes resolves `memory.provider` against bundled
  providers first, so a directory named `mem0` would be shadowed by the bundled
  plugin and never load.
- **Do not move the plugin into a subdirectory.**
- **No new unconditional runtime dependencies.** `mem0ai` is the only declared
  one. Embedder packages install per selection via
  `ensure_embedder_dependencies`, because `hermes memory setup` installs
  `plugin.yaml` dependencies *before* the user picks an embedder. Anything else
  must be optional and lazily imported.
- **The test suite must never install anything.** `tests/_bootstrap.py` sets
  `MEM0_HERMES_NO_INSTALL=1`, which makes `ensure_embedder_dependencies` refuse
  to run pip. Tests that exercise the install path patch
  `tools.lazy_deps.install_specs` and lift the guard for their duration — see
  `EmbedderDependencyTests`. Never point a test at an embedder whose package
  isn't already present.
- **Embedded Qdrant allows one live owner per directory.** Everything touching
  `memory.vector_store` goes through the lease in `_backend.py`
  (`_install_lease` wraps the store's public methods) so the OS lock is held per
  call. Never open `QdrantClient(path=...)` outside `_open_local_client` (it
  owns the retry and must not leak the refused `.lock` handle), and never hold a
  lease across an LLM call.
- **Never set `journal_mode=WAL` directly.** `history.db` goes through
  `hermes_state.apply_wal_with_fallback`, which knows about WAL-hostile
  filesystems and SQLite builds carrying the WAL-reset corruption bug.
- **Never route embeddings through `call_llm`.** Hermes has no embedding path;
  the `embedder` block is Mem0's to handle.
- **Never write system credential file paths into any tracked file** — the
  password and shadow file paths under the system config directory. Hermes's
  plugin scanner rates them CRITICAL under its traversal category, one critical
  finding makes the verdict `dangerous`, and for a community plugin that verdict
  cannot be overridden by `--force`: the install is simply refused. This has
  happened once, from a test fixture. `ScannerCriticalPatternTests` in
  `tests/test_plugin_loading.py` guards it in CI, and it scans Markdown too.

## Conventions

- New source files carry `# SPDX-License-Identifier: Apache-2.0 OR MIT` as the
  first line.
- Comments explain *why*, usually with a pointer into hermes-agent or Mem0.
  Match the surrounding density — it is deliberately high, and the constants at
  the top of `__init__.py` and the block comments in `_backend.py` are the
  reference for tone.
- Stdlib `unittest` only. No pytest, no fixtures package, no dev dependencies:
  the suite runs inside whatever interpreter Hermes was installed with.
- Behavior that only manifests through Mem0 belongs in `tests/test_backend.py`;
  provider lifecycle in `tests/test_provider.py`; config resolution and the
  wizard in `tests/test_config.py`; the routed adapter and JSON coercion in
  `tests/test_hermes_llm.py`; loader/scanner contracts in
  `tests/test_plugin_loading.py`.
- Platform-conditional code needs a test: `HERMES_HOME` resolution, the
  fastembed cache directory, console encoding, and `dev_link.py`'s junction
  fallback all branch on the platform, and that is where this plugin's bugs have
  actually been.
- Update `README.md` when configuration keys, tool names, or install steps
  change. Config keys also appear in `_WIZARD_KEY_MAP` and `config_schema()` in
  `_config.py` — a new key that is only added to `default_config` will not be
  offered by `hermes memory setup`.

## Before finishing a change

1. Run the full suite in a Hermes venv and read the skip lines — a change that
   silently moved everything into "skipped" proved nothing.
2. If the change touches the config surface, confirm `load_config` precedence
   still holds: defaults -> inherited `mem0.json` -> `mem0_hermes.json` -> env.
3. One logical change per commit, and say what breaks if it is wrong.
