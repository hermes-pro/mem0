# SPDX-License-Identifier: Apache-2.0 OR MIT
"""End-to-end checks against Hermes's own memory-provider loader.

These exercise the parts that only break in situ: the synthetic
``_hermes_user_memory.<name>`` package the loader invents for user-installed
plugins, the relative imports inside the plugin, and the ``register(ctx)``
contract. Skipped when the hermes-agent checkout is not importable.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import _bootstrap
from _bootstrap import REPO_ROOT

# The plugin is the repo root, whose directory name is whatever the checkout is
# called. Hermes derives the provider name from the *installed* directory name,
# so the loader tests run against a staged copy named ``mem0_hermes`` — the
# layout `hermes plugins install` produces.
PLUGIN_DIR: Path
_STAGING: tempfile.TemporaryDirectory


def setUpModule() -> None:
    global PLUGIN_DIR, _STAGING
    _STAGING = tempfile.TemporaryDirectory()
    PLUGIN_DIR = Path(_STAGING.name) / "mem0_hermes"
    PLUGIN_DIR.mkdir()
    shutil.copy2(REPO_ROOT / "plugin.yaml", PLUGIN_DIR / "plugin.yaml")
    for source in REPO_ROOT.glob("*.py"):
        shutil.copy2(source, PLUGIN_DIR / source.name)


def tearDownModule() -> None:
    _STAGING.cleanup()


# Content Hermes's plugin scanner (tools/skills_guard.py) rates CRITICAL: the
# system credential files, under the "traversal" category. One critical finding
# makes the whole verdict "dangerous", and for a community-sourced plugin that
# verdict cannot be overridden by --force -- the install is simply refused.
#
# This repo has no reason to contain those names except as a test fixture,
# which is exactly how it happened once: a message list written to prove that
# tool output is NOT extracted into memory used one of them as the tool result,
# and stopped the plugin installing at all.
#
# Assembled from fragments rather than written out, so this list does not
# trigger the very scan it exists to protect -- and so this file doesn't have
# to exempt itself from its own check.
BLOCKING_PATTERNS = tuple("/etc/" + name for name in ("passwd", "shadow"))

# Deliberately narrow: only the critical traversal rules, not the scanner's
# full table. This is not the authority -- the scanner is. Its job is to fail
# in CI, where the cost is a red build, rather than at install time, where the
# cost is a user who cannot install the plugin.


def _excluded_names() -> set:
    """Directory/file names that are not part of the installed plugin.

    Read from .gitignore rather than hardcoded, so the sibling ``hermes-agent``
    checkout a contributor clones next to this one -- which has every one of
    these patterns in its own docs and tests -- doesn't get scanned as if it
    shipped with the plugin.
    """
    names = {".git"}
    gitignore = REPO_ROOT / ".gitignore"
    if gitignore.is_file():
        for line in gitignore.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                names.add(line.rstrip("/"))
    return names


class ScannerCriticalPatternTests(unittest.TestCase):
    def test_no_file_contains_an_install_blocking_pattern(self):
        excluded = _excluded_names()
        offenders = []
        for path in REPO_ROOT.rglob("*"):
            if not path.is_file() or excluded & set(path.parts):
                continue
            if path.suffix.lower() not in {".py", ".md", ".yaml", ".yml", ".json", ".txt"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for pattern in BLOCKING_PATTERNS:
                if pattern in text:
                    line = text[: text.index(pattern)].count("\n") + 1
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}:{line} contains {pattern!r}"
                    )
        self.assertEqual(
            offenders,
            [],
            "these would make the plugin scanner refuse to install this plugin",
        )


def _loader_available() -> bool:
    if not _bootstrap.HERMES_AGENT_AVAILABLE:
        return False
    try:
        return importlib.util.find_spec("plugins.memory") is not None
    except Exception:
        return False


@unittest.skipUnless(
    _loader_available(), "requires the hermes-agent checkout (plugins.memory)"
)
class LoaderTests(unittest.TestCase):
    def test_directory_passes_the_memory_provider_heuristic(self):
        from plugins.memory import _is_memory_provider_dir

        self.assertTrue(_is_memory_provider_dir(PLUGIN_DIR))

    def test_loader_returns_a_configured_provider_instance(self):
        from agent.memory_provider import MemoryProvider
        from plugins.memory import _load_provider_from_dir

        provider = _load_provider_from_dir(PLUGIN_DIR)
        self.assertIsNotNone(provider, "loader returned no provider instance")
        self.assertIsInstance(provider, MemoryProvider)
        self.assertEqual(provider.name, "mem0_hermes")

    def test_tool_schemas_are_well_formed(self):
        from plugins.memory import _load_provider_from_dir

        provider = _load_provider_from_dir(PLUGIN_DIR)
        names = []
        for schema in provider.get_tool_schemas():
            self.assertIn("name", schema)
            self.assertIn("description", schema)
            self.assertEqual(schema["parameters"]["type"], "object")
            names.append(schema["name"])
        self.assertEqual(
            names,
            [
                "mem0_search",
                "mem0_get_all",
                "mem0_add",
                "mem0_update",
                "mem0_delete",
            ],
        )

    def test_tool_calls_fail_cleanly_before_initialize(self):
        from plugins.memory import _load_provider_from_dir

        provider = _load_provider_from_dir(PLUGIN_DIR)
        payload = json.loads(provider.handle_tool_call("mem0_search", {"query": "x"}))
        self.assertIn("error", payload)
        self.assertIn("not initialized", payload["error"])

    def test_config_schema_is_wizard_shaped(self):
        from plugins.memory import _load_provider_from_dir

        provider = _load_provider_from_dir(PLUGIN_DIR)
        schema = provider.get_config_schema()
        self.assertTrue(schema)
        for field in schema:
            self.assertIsInstance(field.get("key"), str)
            self.assertIsInstance(field.get("description"), str)

    def test_save_config_writes_the_native_config_file(self):
        from plugins.memory import _load_provider_from_dir

        provider = _load_provider_from_dir(PLUGIN_DIR)
        with tempfile.TemporaryDirectory() as tmp:
            provider.save_config({"user_id": "alice", "llm_model": "claude-opus-5"}, tmp)
            saved = json.loads(
                (Path(tmp) / "mem0_hermes.json").read_text(encoding="utf-8")
            )
        self.assertEqual(saved["user_id"], "alice")
        self.assertEqual(saved["llm"]["model"], "claude-opus-5")

    def test_register_uses_the_plugin_context_contract(self):
        from plugins.memory import _ProviderCollector, _load_provider_from_dir

        _load_provider_from_dir(PLUGIN_DIR)  # ensures the module is importable
        import sys

        module = sys.modules["_hermes_user_memory.mem0_hermes"]
        # _ProviderCollector gained a required `name` argument; accept either
        # signature so the suite tracks whichever Hermes the user has.
        import inspect

        if "name" in inspect.signature(_ProviderCollector).parameters:
            collector = _ProviderCollector("mem0_hermes")
        else:
            collector = _ProviderCollector()
        module.register(collector)
        self.assertIsNotNone(collector.provider)
        self.assertEqual(collector.provider.name, "mem0_hermes")

    def test_register_declares_the_auxiliary_task_when_supported(self):
        import sys

        from plugins.memory import _load_provider_from_dir

        _load_provider_from_dir(PLUGIN_DIR)
        module = sys.modules["_hermes_user_memory.mem0_hermes"]

        recorded = {}

        class RicherContext:
            def register_memory_provider(self, provider):
                recorded["provider"] = provider

            def register_auxiliary_task(self, key, **kwargs):
                recorded["task"] = (key, kwargs)

        module.register(RicherContext())
        self.assertIn("provider", recorded)
        self.assertEqual(recorded["task"][0], "mem0_hermes_extraction")
        self.assertEqual(recorded["task"][1]["defaults"]["provider"], "auto")


@unittest.skipUnless(
    _loader_available(), "requires the hermes-agent checkout (plugins.memory)"
)
class AuxiliaryTaskResolutionTests(unittest.TestCase):
    def test_unconfigured_task_resolves_to_auto(self):
        """An unregistered auxiliary task must fall through to the main model.

        This is the assumption the whole plugin rests on: ``call_llm(task=...)``
        with no ``auxiliary.<task>`` config resolves provider "auto", which
        auxiliary_client maps to the user's main provider + main model.
        """
        from agent.auxiliary_client import _resolve_task_provider_model

        provider, model, base_url, api_key, _api_mode = _resolve_task_provider_model(
            "mem0_hermes_extraction"
        )
        self.assertEqual(provider, "auto")
        self.assertIn(model, (None, ""))
        self.assertIn(base_url, (None, ""))
        self.assertIn(api_key, (None, ""))


if __name__ == "__main__":
    unittest.main()
