# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Tests for the Mem0 backend wiring.

The translation tests are pure logic. ``Mem0IntegrationTests`` builds a real
Mem0 ``Memory`` against a temporary local Qdrant collection and asserts that
extraction actually runs through the Hermes-routed adapter — it is skipped when
``mem0ai``/``qdrant-client`` are not importable (i.e. outside the Hermes venv).
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from _bootstrap import FakeResponse, RecordingCallLlm, install_call_llm

from mem0_hermes import _backend, _config


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


class MemoryConfigTranslationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_llm_block_is_validated_as_openai_but_carries_our_config(self):
        config = _config.load_config(self.home)
        built = _backend.build_memory_config(config)
        # "openai" only satisfies Mem0's provider allowlist at validation time;
        # build_memory() reassigns it to the registered routed provider.
        self.assertEqual(built["llm"]["provider"], "openai")
        self.assertEqual(built["llm"]["config"]["task"], _config.DEFAULT_AUX_TASK)
        self.assertEqual(built["llm"]["config"]["json_mode"], "prompt")
        # Crucially, no OpenAI credentials are injected anywhere.
        self.assertEqual(built["llm"]["config"].get("api_key", ""), "")

    def test_embedding_dims_propagate_to_vector_store(self):
        config = _config.load_config(self.home)
        built = _backend.build_memory_config(config)
        self.assertEqual(built["vector_store"]["config"]["embedding_model_dims"], 1536)

    def test_paths_are_expanded_and_parents_created(self):
        config = _config.load_config(self.home)
        config["vector_store"]["config"]["path"] = str(self.home / "nested" / "qdrant")
        config["history_db_path"] = str(self.home / "nested2" / "history.db")
        built = _backend.build_memory_config(config)
        self.assertTrue((self.home / "nested").is_dir())
        self.assertTrue((self.home / "nested2").is_dir())
        self.assertTrue(built["history_db_path"].endswith("history.db"))

    def test_custom_instructions_forwarded_only_when_set(self):
        config = _config.load_config(self.home)
        self.assertNotIn("custom_instructions", _backend.build_memory_config(config))
        config["custom_instructions"] = "Only remember work facts."
        self.assertEqual(
            _backend.build_memory_config(config)["custom_instructions"],
            "Only remember work facts.",
        )

    def test_reranker_block_passed_through_only_when_set(self):
        config = _config.load_config(self.home)
        self.assertNotIn("reranker", _backend.build_memory_config(config))
        config["reranker"] = {
            "provider": "llm_reranker",
            "config": {"llm": {"provider": "hermes_routed", "config": {}}},
        }
        built = _backend.build_memory_config(config)
        self.assertEqual(built["reranker"]["provider"], "llm_reranker")
        self.assertIsNot(built["reranker"], config["reranker"])  # deep-copied

    def test_telemetry_disabled_by_default(self):
        saved = os.environ.pop("MEM0_TELEMETRY", None)
        try:
            _backend._prepare_environment({"telemetry": False})
            self.assertEqual(os.environ.get("MEM0_TELEMETRY"), "false")
        finally:
            os.environ.pop("MEM0_TELEMETRY", None)
            if saved is not None:
                os.environ["MEM0_TELEMETRY"] = saved

    def test_telemetry_opt_in_leaves_env_alone(self):
        saved = os.environ.pop("MEM0_TELEMETRY", None)
        try:
            _backend._prepare_environment({"telemetry": True})
            self.assertIsNone(os.environ.get("MEM0_TELEMETRY"))
        finally:
            if saved is not None:
                os.environ["MEM0_TELEMETRY"] = saved


class FakeEmbedder:
    """Deterministic stand-in so no embedding API is contacted."""

    def __init__(self, dims=1536):
        self.dims = dims
        self.calls = 0

    def _vector(self, text):
        self.calls += 1
        vector = [0.0] * self.dims
        for index, char in enumerate(str(text)[: self.dims]):
            vector[index] = (ord(char) % 17) / 17.0
        vector[0] = vector[0] or 0.5
        return vector

    def embed(self, text, memory_action=None):
        return self._vector(text)

    def embed_batch(self, texts, memory_action="add"):
        return [self._vector(text) for text in texts]


@unittest.skipUnless(
    _have("mem0") and _have("qdrant_client"),
    "requires mem0ai + qdrant-client (run with the Hermes venv interpreter)",
)
class Mem0IntegrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.config = _config.load_config(self.home)
        # A placeholder key keeps Mem0's OpenAI *embedder* constructible; it is
        # replaced with FakeEmbedder before any call is made.
        self.config["embedder"]["config"]["api_key"] = "test-key-not-used"
        self.backend = None

    def tearDown(self):
        if self.backend is not None:
            self.backend.close()

    def _backend_with_fake_llm(self, response):
        fake = RecordingCallLlm(response=response)
        install_call_llm(fake)
        self.backend = _backend.HermesRoutedMem0Backend(self.config)
        self.backend.memory.embedding_model = FakeEmbedder()
        return fake

    def test_memory_uses_the_routed_llm(self):
        from mem0_hermes._hermes_llm import HermesRoutedLLM

        self._backend_with_fake_llm(FakeResponse('{"memory": []}'))
        self.assertIsInstance(self.backend.memory.llm, HermesRoutedLLM)
        # No OpenAI client anywhere on the LLM object.
        self.assertFalse(hasattr(self.backend.memory.llm, "client"))
        self.assertIn("main Hermes model", self.backend.routing)

    def test_add_with_inference_extracts_through_hermes(self):
        fake = self._backend_with_fake_llm(
            FakeResponse('{"memory": [{"text": "Prefers dark roast coffee"}]}')
        )
        self.backend.add(
            [
                {"role": "user", "content": "I only drink dark roast coffee"},
                {"role": "assistant", "content": "Noted."},
            ],
            user_id="tester",
            agent_id="hermes",
            infer=True,
        )
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.last["task"], _config.DEFAULT_AUX_TASK)
        # Extraction prompt reached the routed model, and the fact landed in the
        # vector store.
        results = self.backend.search("coffee", filters={"user_id": "tester"}, top_k=5)
        self.assertTrue(
            any("dark roast" in (r.get("memory") or "") for r in results), results
        )

    def test_verbatim_add_makes_no_llm_call(self):
        fake = self._backend_with_fake_llm(FakeResponse('{"memory": []}'))
        self.backend.add(
            [{"role": "user", "content": "My cat is named Ada"}],
            user_id="tester",
            agent_id="hermes",
            infer=False,
        )
        self.assertEqual(fake.calls, [])
        results = self.backend.search("cat", filters={"user_id": "tester"}, top_k=5)
        self.assertTrue(any("Ada" in (r.get("memory") or "") for r in results), results)

    def test_update_and_delete_round_trip(self):
        self._backend_with_fake_llm(FakeResponse('{"memory": []}'))
        self.backend.add(
            [{"role": "user", "content": "Lives in Berlin"}],
            user_id="tester",
            agent_id="hermes",
            infer=False,
        )
        found = self.backend.search("Berlin", filters={"user_id": "tester"}, top_k=5)
        memory_id = found[0]["id"]
        self.backend.update(memory_id, "Lives in Hamburg")
        updated = self.backend.search("Hamburg", filters={"user_id": "tester"}, top_k=5)
        self.assertTrue(any("Hamburg" in (r.get("memory") or "") for r in updated))
        self.backend.delete(memory_id)
        remaining = self.backend.search("Hamburg", filters={"user_id": "tester"}, top_k=5)
        self.assertFalse(any(r.get("id") == memory_id for r in remaining))


if __name__ == "__main__":
    unittest.main()
