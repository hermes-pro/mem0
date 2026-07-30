# Contributing to mem0-hermes

Thanks for helping out. This page covers the legal bit, then the practical bit.

**Contact:** _maintainers: replace this line with the address that should receive
CLA copies and private security reports before publishing this repo — it is
referenced from [`CLA.md`](CLA.md)._

## Licensing and the CLA

This project is dual-licensed **Apache-2.0 OR MIT** (see [`LICENSE`](LICENSE)).
Contributions come in under the same terms — inbound equals outbound.

**Opening a pull request constitutes acceptance of the
[Contributor License Agreement](CLA.md)**, for that pull request and any later
ones. There is no bot and nothing to sign. The CLA is a license grant, not a
copyright assignment: you keep ownership of your work, and it lets us keep
distributing the project under both licenses.

Two things to check before you open a PR:

- **Employed contributors.** If you write code on work time, with employer
  equipment, or within the scope of your job, your employer may own the
  copyright. Make sure you have permission, or have your employer accept the
  [Corporate CLA](CLA.md#corporate-contributor-license-agreement).
- **Third-party code.** Don't paste in code you didn't write without saying so.
  If a change includes third-party work, call it out in the PR description with
  its source and license, per [CLA §6](CLA.md#6-third-party-work). Vendored code
  under a license incompatible with Apache-2.0 OR MIT can't be merged.

New source files should carry the SPDX header used by the existing ones:

```python
# SPDX-License-Identifier: Apache-2.0 OR MIT
```

Issues, reviews, and bug reports need no CLA. **Security problems: report them
privately to the contact address above — please don't open a public issue.**

## Development setup

You need a working Hermes install and a hermes-agent checkout (for the
`agent.*` and `plugins.*` imports the tests exercise):

```bash
git clone https://github.com/hermes-pro/mem0.git && cd mem0
git clone https://github.com/NousResearch/hermes-agent.git      # gitignored here
python scripts/dev_link.py --activate                          # link into $HERMES_HOME
```

`dev_link.py` links rather than copies, so edits apply on the next session with
no reinstall. Point `HERMES_AGENT_DIR` elsewhere if your checkout lives outside
this directory.

Run the suite with the **Hermes venv's interpreter** — that's where `mem0ai` and
`qdrant-client` live, and their presence enables the integration tests:

```powershell
& "$env:LOCALAPPDATA\hermes\hermes-agent\venv\Scripts\python.exe" -m unittest discover -s tests -t tests -v
```

```bash
"$HOME/.local/share/hermes/hermes-agent/venv/bin/python" -m unittest discover -s tests -t tests -v
```

The suite is stdlib `unittest` — no pytest, no fixtures to install. It must stay
that way: it runs inside whatever interpreter Hermes was installed with, and we
don't get to add dev dependencies there.

## Things to know before changing code

A few constraints aren't obvious from reading a single file:

- **`_hermes_llm.py` must not import Mem0 at module level.** Hermes's plugin
  loader execs each top-level `*.py` in the plugin directory *before* `mem0ai`
  is guaranteed installed, and registers the module in `sys.modules` first — so
  a failed module-level import leaves a half-initialized module behind. Mem0
  only duck-types its LLM object, so keep Mem0 imports inside functions.
- **The plugin lives at the repo root**, so `hermes plugins install
  hermes-pro/mem0` finds `plugin.yaml`. Don't move it into a subdirectory.
- **`name: mem0_hermes` in `plugin.yaml` is load-bearing.** It decides the
  installed directory name, and Hermes resolves `memory.provider` against
  bundled providers first — a directory named `mem0` would be shadowed by the
  bundled Mem0 plugin and never load.
- **No new unconditional runtime dependencies.** `mem0ai` is the only declared
  one. Embedder packages (`fastembed`, `ollama`, `sentence-transformers`) are
  installed per selection via `ensure_embedder_dependencies`, because
  `hermes memory setup` installs `plugin.yaml` dependencies *before* the user
  picks an embedder — declaring them there would download every embedder's stack
  for everyone. Anything else must be optional and lazily imported.
- **The test suite must never install anything.** `tests/_bootstrap.py` sets
  `MEM0_HERMES_NO_INSTALL=1`, which makes `ensure_embedder_dependencies` refuse
  to run pip. Tests that need to exercise the install path patch
  `tools.lazy_deps.install_specs` and lift that guard for their duration — see
  `EmbedderDependencyTests`. Never point a test at an embedder whose package
  isn't already present.
- **Don't route embeddings through `call_llm`.** Hermes has no embedding path;
  the `embedder` block is Mem0's to handle. If that changes upstream, that's a
  feature, not a refactor.

## Pull requests

- One logical change per PR. Say what breaks if it's wrong.
- Add or update tests. Behavior that only manifests through Mem0 belongs in
  `tests/test_backend.py`; provider lifecycle in `tests/test_provider.py`.
- Run the full suite before pushing, and paste the result in the PR.
- Match the surrounding code — comment density included. Comments here explain
  *why* something is the way it is, usually with a pointer into hermes-agent or
  Mem0. Keep that.
- Update `README.md` when you change configuration keys or install steps.
- Leave unrelated formatting alone.
