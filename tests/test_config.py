# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Tests for mem0_hermes configuration resolution."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

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
