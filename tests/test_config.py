# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Tests for mem0_hermes configuration resolution."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _bootstrap  # noqa: F401 - sys.path bootstrap

from mem0_hermes import _config


class TempHomeTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self._saved_env = {
            key: os.environ.pop(key)
            for key in list(os.environ)
            if key.startswith(("MEM0_HERMES_", "MEM0_USER_ID", "MEM0_AGENT_ID"))
        }
        self.addCleanup(self._restore_env)
        self.addCleanup(self._tmp.cleanup)

    def _restore_env(self):
        for key in list(os.environ):
            if key.startswith(("MEM0_HERMES_", "MEM0_USER_ID", "MEM0_AGENT_ID")):
                del os.environ[key]
        os.environ.update(self._saved_env)

    def write_config(self, data, name=_config.CONFIG_FILENAME):
        (self.home / name).write_text(json.dumps(data), encoding="utf-8")


class DefaultsTests(TempHomeTestCase):
    def test_defaults_route_to_main_model(self):
        config = _config.load_config(self.home)
        llm = config["llm"]
        self.assertEqual(llm["task"], _config.DEFAULT_AUX_TASK)
        self.assertEqual(llm["provider"], "")
        self.assertEqual(llm["model"], "")
        self.assertEqual(llm["json_mode"], "prompt")

    def test_state_paths_live_under_hermes_home(self):
        config = _config.load_config(self.home)
        self.assertTrue(
            config["vector_store"]["config"]["path"].startswith(str(self.home))
        )
        self.assertTrue(config["history_db_path"].startswith(str(self.home)))

    def test_telemetry_off_by_default(self):
        self.assertFalse(_config.load_config(self.home)["telemetry"])

    def test_default_embedder_is_local_and_keyless(self):
        # The whole plugin exists for people without an OpenAI key; a default
        # embedder that needs one would block them at step one.
        config = _config.load_config(self.home)
        embedder = config["embedder"]
        self.assertEqual(embedder["provider"], "fastembed")
        self.assertEqual(embedder["config"]["model"], _config.DEFAULT_FASTEMBED_MODEL)
        self.assertEqual(
            embedder["config"]["embedding_dims"],
            _config.KNOWN_DIMS[_config.DEFAULT_FASTEMBED_MODEL],
        )
        self.assertEqual(_config.embedder_key_missing(config), "")

    def test_fastembed_is_first_in_the_picker(self):
        self.assertEqual(_config.EMBEDDER_CHOICES[0], "fastembed")


class PlatformDefaultHomeTests(unittest.TestCase):
    """The fallback used when ``hermes_constants`` is not importable.

    Must agree with ``hermes_constants._get_platform_default_hermes_home()``.
    A POSIX-only fallback here sends the vector store to ``~/.hermes`` on
    Windows while Hermes itself uses ``%LOCALAPPDATA%\\hermes`` — the plugin
    then reads an empty collection and writes memories nothing else can see.
    """

    def test_windows_uses_local_appdata(self):
        with mock.patch.object(_config.sys, "platform", "win32"):
            with mock.patch.dict(
                os.environ, {"LOCALAPPDATA": r"C:\Users\tester\AppData\Local"}
            ):
                self.assertEqual(
                    _config._platform_default_home(),
                    Path(r"C:\Users\tester\AppData\Local") / "hermes",
                )

    def test_windows_without_localappdata_falls_back_under_home(self):
        with mock.patch.object(_config.sys, "platform", "win32"):
            with mock.patch.dict(os.environ, {"LOCALAPPDATA": ""}):
                with mock.patch.object(Path, "home", staticmethod(lambda: Path("/u"))):
                    self.assertEqual(
                        _config._platform_default_home(),
                        Path("/u") / "AppData" / "Local" / "hermes",
                    )

    def test_posix_uses_dot_hermes(self):
        with mock.patch.object(_config.sys, "platform", "linux"):
            with mock.patch.object(Path, "home", staticmethod(lambda: Path("/home/t"))):
                self.assertEqual(
                    _config._platform_default_home(), Path("/home/t") / ".hermes"
                )

    def test_env_var_still_wins_over_platform_default(self):
        with mock.patch.dict(os.environ, {"HERMES_HOME": r"D:\custom"}):
            with mock.patch.dict("sys.modules", {"hermes_constants": None}):
                self.assertEqual(_config.hermes_home(), Path(r"D:\custom"))


class OverrideTests(TempHomeTestCase):
    def test_file_overrides_defaults_and_preserves_siblings(self):
        self.write_config({"llm": {"model": "claude-opus-5"}, "agent_id": "scribe"})
        config = _config.load_config(self.home)
        self.assertEqual(config["llm"]["model"], "claude-opus-5")
        self.assertEqual(config["agent_id"], "scribe")
        # Untouched keys still come from the defaults.
        self.assertEqual(config["llm"]["task"], _config.DEFAULT_AUX_TASK)
        self.assertEqual(config["llm"]["temperature"], 0.1)

    def test_env_beats_file(self):
        self.write_config({"llm": {"model": "from-file"}})
        os.environ["MEM0_HERMES_LLM_MODEL"] = "from-env"
        self.assertEqual(_config.load_config(self.home)["llm"]["model"], "from-env")

    def test_env_covers_every_llm_key(self):
        os.environ.update(
            {
                "MEM0_HERMES_LLM_PROVIDER": "anthropic",
                "MEM0_HERMES_LLM_MODEL": "claude-opus-5",
                "MEM0_HERMES_LLM_BASE_URL": "https://example.invalid",
                "MEM0_HERMES_LLM_API_KEY": "sk-env",
                "MEM0_HERMES_LLM_TASK": "custom_task",
                "MEM0_HERMES_JSON_MODE": "response_format",
                "MEM0_HERMES_LLM_TEMPERATURE": "0.7",
                "MEM0_HERMES_LLM_MAX_TOKENS": "512",
                "MEM0_HERMES_LLM_TIMEOUT": "45",
            }
        )
        llm = _config.load_config(self.home)["llm"]
        self.assertEqual(llm["provider"], "anthropic")
        self.assertEqual(llm["model"], "claude-opus-5")
        self.assertEqual(llm["base_url"], "https://example.invalid")
        self.assertEqual(llm["api_key"], "sk-env")
        self.assertEqual(llm["task"], "custom_task")
        self.assertEqual(llm["json_mode"], "response_format")
        self.assertEqual(llm["temperature"], 0.7)
        self.assertEqual(llm["max_tokens"], 512)
        self.assertEqual(llm["timeout"], 45.0)

    def test_numeric_env_overrides_are_typed_not_strings(self):
        os.environ["MEM0_HERMES_LLM_MAX_TOKENS"] = "512"
        llm = _config.load_config(self.home)["llm"]
        self.assertIsInstance(llm["max_tokens"], int)

    def test_malformed_numeric_env_falls_back_to_the_default(self):
        os.environ["MEM0_HERMES_LLM_TEMPERATURE"] = "warm"
        with self.assertLogs("mem0_hermes._config", level="WARNING") as logs:
            llm = _config.load_config(self.home)["llm"]
        self.assertEqual(llm["temperature"], 0.1)
        self.assertIn("MEM0_HERMES_LLM_TEMPERATURE", "".join(logs.output))

    def test_inherits_embedder_and_vector_store_from_bundled_mem0(self):
        self.write_config(
            {
                "user_id": "alice",
                "oss": {
                    "llm": {"provider": "openai", "config": {"model": "gpt-5-mini"}},
                    "embedder": {
                        "provider": "ollama",
                        "config": {"model": "nomic-embed-text"},
                    },
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {"path": "/tmp/existing-qdrant"},
                    },
                },
            },
            name="mem0.json",
        )
        config = _config.load_config(self.home)
        self.assertEqual(config["embedder"]["provider"], "ollama")
        self.assertEqual(config["vector_store"]["config"]["path"], "/tmp/existing-qdrant")
        self.assertEqual(config["user_id"], "alice")
        # The bundled OSS llm block must NOT leak in — replacing it is the point.
        self.assertEqual(config["llm"]["provider"], "")
        self.assertEqual(config["llm"]["model"], "")

    def test_embedding_dims_backfilled_for_known_model(self):
        self.write_config(
            {"embedder": {"provider": "ollama", "config": {"model": "nomic-embed-text"}}}
        )
        config = _config.load_config(self.home)
        self.assertEqual(config["embedder"]["config"]["embedding_dims"], 768)

    def test_explicit_dims_are_respected(self):
        self.write_config(
            {
                "embedder": {
                    "provider": "openai",
                    "config": {"model": "text-embedding-3-small", "embedding_dims": 512},
                }
            }
        )
        config = _config.load_config(self.home)
        self.assertEqual(config["embedder"]["config"]["embedding_dims"], 512)


class UserIdTests(TempHomeTestCase):
    def test_configured_id_wins(self):
        config = {"user_id": "alice"}
        self.assertEqual(_config.resolved_user_id(config, "telegram-123"), "alice")

    def test_placeholder_defers_to_gateway_id(self):
        config = {"user_id": _config.DEFAULT_USER_ID}
        self.assertEqual(_config.resolved_user_id(config, "telegram-123"), "telegram-123")

    def test_falls_back_to_default(self):
        self.assertEqual(_config.resolved_user_id({}, ""), _config.DEFAULT_USER_ID)


class WizardTests(TempHomeTestCase):
    def test_flat_answers_become_nested_config(self):
        values = {
            "user_id": "alice",
            "llm_model": "claude-opus-5",
            "json_mode": "response_format",
            "embedder_provider": "ollama",
            "embedder_model": "nomic-embed-text",
            "embedder_url": "http://localhost:11434",
            "vector_path": "/data/qdrant",
        }
        out = _config.wizard_values_to_config(values)
        self.assertEqual(out["user_id"], "alice")
        self.assertEqual(out["llm"]["model"], "claude-opus-5")
        self.assertEqual(out["llm"]["json_mode"], "response_format")
        self.assertEqual(out["embedder"]["provider"], "ollama")
        self.assertEqual(
            out["embedder"]["config"]["ollama_base_url"], "http://localhost:11434"
        )
        self.assertEqual(out["embedder"]["config"]["embedding_dims"], 768)
        self.assertEqual(out["vector_store"]["config"]["path"], "/data/qdrant")

    def test_numeric_answers_are_coerced(self):
        out = _config.wizard_values_to_config({"temperature": "0.2", "max_tokens": "800"})
        self.assertEqual(out["llm"]["temperature"], 0.2)
        self.assertEqual(out["llm"]["max_tokens"], 800)

    def test_switching_embedder_drops_stale_dims(self):
        existing = {
            "embedder": {
                "provider": "openai",
                "config": {"model": "text-embedding-3-small", "embedding_dims": 1536},
            }
        }
        out = _config.wizard_values_to_config(
            {"embedder_provider": "fastembed", "embedder_model": "thenlper/gte-large"},
            existing,
        )
        self.assertEqual(out["embedder"]["config"]["embedding_dims"], 1024)

    def test_save_config_round_trips_and_merges(self):
        _config.save_config({"user_id": "alice"}, self.home)
        _config.save_config({"llm_model": "claude-opus-5"}, self.home)
        saved = json.loads((self.home / _config.CONFIG_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(saved["user_id"], "alice")
        self.assertEqual(saved["llm"]["model"], "claude-opus-5")

    def test_schema_covers_wizard_keys(self):
        keys = {field["key"] for field in _config.config_schema(_config.default_config(self.home))}
        self.assertTrue({"user_id", "llm_model", "embedder_provider"} <= keys)


class EmbedderDependencyTests(TempHomeTestCase):
    """Per-embedder pip requirements and the install-on-selection hook."""

    def setUp(self):
        super().setUp()
        self._installed = []
        self._result = type("R", (), {"ok": True, "reason": "", "stderr": ""})()

    def _fake_installer(self, specs, **kwargs):
        self._installed.append(list(specs))
        return self._result

    def _patch_installer(self):
        """Route ensure_embedder_dependencies at a recorder, not real pip."""
        import sys
        import types

        module = sys.modules.get("tools.lazy_deps")
        created = module is None
        if created:
            tools = sys.modules.get("tools")
            if tools is None:
                tools = types.ModuleType("tools")
                tools.__path__ = []
                sys.modules["tools"] = tools
            module = types.ModuleType("tools.lazy_deps")
            sys.modules["tools.lazy_deps"] = module
        previous = getattr(module, "install_specs", None)
        module.install_specs = self._fake_installer
        # The suite sets MEM0_HERMES_NO_INSTALL to stop real installs; lift it
        # only while the installer is a recorder.
        saved_guard = os.environ.pop(_config.NO_INSTALL_ENV, None)

        def restore():
            if created:
                sys.modules.pop("tools.lazy_deps", None)
            elif previous is not None:
                module.install_specs = previous
            else:
                delattr(module, "install_specs")
            if saved_guard is not None:
                os.environ[_config.NO_INSTALL_ENV] = saved_guard

        self.addCleanup(restore)

    def test_requirements_listed_only_for_the_selected_provider(self):
        for provider, expected in (
            ("fastembed", "fastembed"),
            ("ollama", "ollama"),
            ("huggingface", "sentence-transformers"),
            ("openai", None),
            ("gemini", None),
        ):
            config = {"embedder": {"provider": provider, "config": {}}}
            specs = _config.embedder_pip_requirements(config)
            if expected is None:
                self.assertEqual(specs, (), provider)
            else:
                # Already-importable packages drop out, so assert on membership.
                self.assertTrue(
                    all(spec.startswith(expected) for spec in specs), (provider, specs)
                )

    def test_importable_packages_are_not_reinstalled(self):
        config = {"embedder": {"provider": "fastembed", "config": {}}}
        if not _config._importable("fastembed"):
            self.skipTest("fastembed is not installed in this interpreter")
        self.assertEqual(_config.embedder_pip_requirements(config), ())
        self.assertEqual(_config.ensure_embedder_dependencies(config), (True, ""))

    def test_install_guard_blocks_runtime_installs(self):
        os.environ[_config.NO_INSTALL_ENV] = "1"
        config = {"embedder": {"provider": "ollama", "config": {}}}
        if not _config.embedder_pip_requirements(config):
            self.skipTest("the ollama package is already installed")
        ok, message = _config.ensure_embedder_dependencies(config)
        self.assertFalse(ok)
        self.assertIn(_config.NO_INSTALL_ENV, message)

    def test_ensure_installs_missing_specs(self):
        self._patch_installer()
        config = {"embedder": {"provider": "ollama", "config": {}}}
        missing = _config.embedder_pip_requirements(config)
        if not missing:
            self.skipTest("the ollama package is already installed")
        ok, message = _config.ensure_embedder_dependencies(config)
        self.assertTrue(ok, message)
        self.assertEqual(self._installed, [list(missing)])

    def test_ensure_reports_a_blocked_install(self):
        self._patch_installer()
        self._result = type("R", (), {"ok": False, "reason": "installs disabled", "stderr": ""})()
        config = {"embedder": {"provider": "ollama", "config": {}}}
        if not _config.embedder_pip_requirements(config):
            self.skipTest("the ollama package is already installed")
        ok, message = _config.ensure_embedder_dependencies(config)
        self.assertFalse(ok)
        self.assertIn("installs disabled", message)

    def test_save_config_installs_for_the_selected_embedder(self):
        self._patch_installer()
        logged = []
        self.write_config({"embedder": {"provider": "ollama", "config": {}}})
        missing = _config.embedder_pip_requirements(_config.load_config(self.home))
        if not missing:
            self.skipTest("the ollama package is already installed")
        _config.save_config({"agent_id": "x"}, self.home, log=logged.append)
        self.assertEqual(self._installed, [list(missing)])
        self.assertTrue(any("Installing ollama" in line for line in logged), logged)

    def test_save_config_installs_nothing_for_hosted_embedders(self):
        self._patch_installer()
        self.write_config({"embedder": {"provider": "openai", "config": {}}})
        _config.save_config({"agent_id": "x"}, self.home, log=lambda _m: None)
        self.assertEqual(self._installed, [])

    def test_save_config_writes_the_choice_before_installing(self):
        """A failed install must not lose the user's selection."""
        self._patch_installer()
        self._result = type("R", (), {"ok": False, "reason": "no network", "stderr": ""})()
        _config.save_config(
            {"embedder_provider": "ollama", "embedder_model": "nomic-embed-text"},
            self.home,
            log=lambda _m: None,
        )
        saved = json.loads(
            (self.home / _config.CONFIG_FILENAME).read_text(encoding="utf-8")
        )
        self.assertEqual(saved["embedder"]["provider"], "ollama")

    def test_install_can_be_skipped_entirely(self):
        self._patch_installer()
        self.write_config({"embedder": {"provider": "ollama", "config": {}}})
        _config.save_config({"agent_id": "x"}, self.home, install=False)
        self.assertEqual(self._installed, [])


class FastembedDimensionTests(TempHomeTestCase):
    """Vector widths come from fastembed's registry, not from our table."""

    def setUp(self):
        super().setUp()
        if not _config._importable("fastembed"):
            self.skipTest("fastembed is not installed in this interpreter")

    def test_known_dims_match_the_real_registry(self):
        for model, dims in _config.KNOWN_DIMS.items():
            actual = _config.resolve_fastembed_dims(model)
            if actual is None:
                continue  # not a fastembed model (OpenAI/Ollama entry)
            self.assertEqual(actual, dims, model)

    def test_default_model_width_is_authoritative(self):
        self.assertEqual(
            _config.resolve_fastembed_dims(_config.DEFAULT_FASTEMBED_MODEL),
            _config.KNOWN_DIMS[_config.DEFAULT_FASTEMBED_MODEL],
        )

    def test_sync_is_a_noop_when_dims_already_correct(self):
        config = _config.load_config(self.home)
        self.assertEqual(_config.sync_fastembed_dims(config), (False, ""))

    def test_sync_corrects_a_wrong_width(self):
        config = _config.load_config(self.home)
        config["embedder"]["config"]["embedding_dims"] = 1536
        changed, note = _config.sync_fastembed_dims(config)
        self.assertTrue(changed)
        self.assertIn("1536", note)
        self.assertEqual(
            config["embedder"]["config"]["embedding_dims"],
            _config.KNOWN_DIMS[_config.DEFAULT_FASTEMBED_MODEL],
        )

    def test_sync_reports_an_unsupported_model(self):
        config = _config.load_config(self.home)
        config["embedder"]["config"]["model"] = "not/a-real-model"
        changed, note = _config.sync_fastembed_dims(config)
        self.assertFalse(changed)
        self.assertIn("does not offer model", note)
        self.assertIn("Supported models include", note)

    def test_sync_ignores_other_providers(self):
        config = {"embedder": {"provider": "openai", "config": {"model": "x"}}}
        self.assertEqual(_config.sync_fastembed_dims(config), (False, ""))


class EmbedderCredentialTests(TempHomeTestCase):
    def test_missing_openai_key_reported(self):
        saved = os.environ.pop("OPENAI_API_KEY", None)
        try:
            config = {"embedder": {"provider": "openai", "config": {}}}
            self.assertEqual(_config.embedder_key_missing(config), "OPENAI_API_KEY")
        finally:
            if saved is not None:
                os.environ["OPENAI_API_KEY"] = saved

    def test_local_embedders_need_no_key(self):
        for provider in ("ollama", "fastembed", "huggingface", "lmstudio"):
            config = {"embedder": {"provider": provider, "config": {}}}
            self.assertEqual(_config.embedder_key_missing(config), "", provider)

    def test_inline_key_satisfies_check(self):
        config = {"embedder": {"provider": "openai", "config": {"api_key": "sk-x"}}}
        self.assertEqual(_config.embedder_key_missing(config), "")


class BackupPathTests(TempHomeTestCase):
    def test_paths_inside_hermes_home_are_omitted(self):
        config = _config.load_config(self.home)
        self.assertEqual(_config.external_state_paths(config, self.home), [])

    def test_relocated_paths_are_reported(self):
        config = _config.load_config(self.home)
        config["history_db_path"] = "/elsewhere/history.db"
        config["vector_store"]["config"]["path"] = "/elsewhere/qdrant"
        paths = _config.external_state_paths(config, self.home)
        self.assertEqual(len(paths), 2, paths)


if __name__ == "__main__":
    unittest.main()
