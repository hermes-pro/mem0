# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Tests for the Mem0 backend wiring.

The translation tests are pure logic. ``Mem0IntegrationTests`` builds a real
Mem0 ``Memory`` against a temporary local Qdrant collection and asserts that
extraction actually runs through the Hermes-routed adapter — it is skipped when
``mem0ai``/``qdrant-client`` are not importable (i.e. outside the Hermes venv).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import _bootstrap
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
        self.assertEqual(
            built["vector_store"]["config"]["embedding_model_dims"],
            _config.KNOWN_DIMS[_config.DEFAULT_FASTEMBED_MODEL],
        )

    def test_default_embedder_is_local_fastembed(self):
        config = _config.load_config(self.home)
        built = _backend.build_memory_config(config)
        self.assertEqual(built["embedder"]["provider"], "fastembed")
        self.assertEqual(
            built["embedder"]["config"]["model"], _config.DEFAULT_FASTEMBED_MODEL
        )

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

    def test_fastembed_cache_is_durable_and_outside_hermes_home(self):
        cache = _backend._fastembed_cache_dir()
        self.assertEqual(cache.name, "fastembed")
        # Temp dirs get swept; a swept cache means a silent re-download.
        self.assertNotIn("temp", str(cache).lower())
        home = _config.hermes_home().resolve()
        self.assertFalse(
            home == cache or home in cache.parents,
            f"{cache} is inside HERMES_HOME and would bloat `hermes backup`",
        )

    def test_cache_path_set_only_for_fastembed(self):
        saved = os.environ.pop("FASTEMBED_CACHE_PATH", None)
        try:
            _backend._prepare_environment(
                {"embedder": {"provider": "openai", "config": {}}}
            )
            self.assertIsNone(os.environ.get("FASTEMBED_CACHE_PATH"))
            _backend._prepare_environment(
                {"embedder": {"provider": "fastembed", "config": {}}}
            )
            self.assertEqual(
                os.environ.get("FASTEMBED_CACHE_PATH"),
                str(_backend._fastembed_cache_dir()),
            )
        finally:
            os.environ.pop("FASTEMBED_CACHE_PATH", None)
            if saved is not None:
                os.environ["FASTEMBED_CACHE_PATH"] = saved

    def test_existing_cache_path_is_respected(self):
        saved = os.environ.get("FASTEMBED_CACHE_PATH")
        os.environ["FASTEMBED_CACHE_PATH"] = "/operators/choice"
        try:
            _backend._prepare_environment(
                {"embedder": {"provider": "fastembed", "config": {}}}
            )
            self.assertEqual(os.environ["FASTEMBED_CACHE_PATH"], "/operators/choice")
        finally:
            if saved is None:
                os.environ.pop("FASTEMBED_CACHE_PATH", None)
            else:
                os.environ["FASTEMBED_CACHE_PATH"] = saved


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
        # Deliberately NOT the fastembed default: building it would pip-install
        # fastembed and download ONNX weights into whatever interpreter is
        # running the suite. Mem0's OpenAI embedder needs only a placeholder key
        # to construct, and FakeEmbedder replaces it before any call is made.
        self.config["embedder"] = {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
                "embedding_dims": 1536,
                "api_key": "test-key-not-used",
            },
        }
        self.backend = None

    def tearDown(self):
        if self.backend is not None:
            self.backend.close()

    def _backend_with_fake_llm(self, response):
        fake = RecordingCallLlm(response=response)
        install_call_llm(fake)
        self.backend = _backend.HermesRoutedMem0Backend(self.config)
        # Match the collection's width, or upserts fail on dimension mismatch.
        dims = self.config["embedder"]["config"]["embedding_dims"]
        self.backend.memory.embedding_model = FakeEmbedder(dims)
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


@unittest.skipUnless(
    _have("mem0") and _have("qdrant_client") and _have("fastembed"),
    "requires mem0ai + qdrant-client + fastembed",
)
class FastembedDefaultTests(unittest.TestCase):
    """The shipped default, exercised for real: local embeddings, no API key.

    Uses real fastembed embeddings against a real local Qdrant collection — only
    the routed LLM is faked. Loads ONNX weights, so it is the slowest test here;
    it skips when fastembed isn't installed rather than installing it.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.config = _config.load_config(self.home)
        self.backend = None

    def tearDown(self):
        if self.backend is not None:
            self.backend.close()

    def test_default_config_stores_and_recalls_without_any_api_key(self):
        saved = os.environ.pop("OPENAI_API_KEY", None)
        try:
            self.assertEqual(_config.embedder_key_missing(self.config), "")
            install_call_llm(RecordingCallLlm(response=FakeResponse('{"memory": []}')))
            self.backend = _backend.HermesRoutedMem0Backend(self.config)
            self.assertEqual(
                type(self.backend.memory.embedding_model).__name__, "FastEmbedEmbedding"
            )
            self.backend.add(
                [{"role": "user", "content": "Deploys on Fridays are forbidden"}],
                user_id="tester",
                agent_id="hermes",
                infer=False,
            )
            results = self.backend.search(
                "when can we deploy?", filters={"user_id": "tester"}, top_k=5
            )
        finally:
            if saved is not None:
                os.environ["OPENAI_API_KEY"] = saved
        self.assertTrue(
            any("Fridays" in (r.get("memory") or "") for r in results), results
        )

    def test_configured_width_matches_what_fastembed_produces(self):
        # A mismatch here is what silently breaks every write.
        install_call_llm(RecordingCallLlm(response=FakeResponse('{"memory": []}')))
        self.backend = _backend.HermesRoutedMem0Backend(self.config)
        vector = self.backend.memory.embedding_model.embed("probe", "add")
        self.assertEqual(
            len(vector), self.config["embedder"]["config"]["embedding_dims"]
        )


@unittest.skipUnless(
    _have("mem0") and _have("qdrant_client") and _have("fastembed"),
    "requires mem0ai + qdrant-client + fastembed",
)
class LocalStoreSharingTests(unittest.TestCase):
    """Embedded Qdrant allows one live owner per directory; prove we cope.

    The failure being defended against:
        RuntimeError: Storage folder <path> is already accessed by another
        instance of Qdrant client.
    """

    def setUp(self):
        # ignore_cleanup_errors: a *refused* QdrantLocal lock leaves its .lock
        # handle open inside qdrant-client until the exception is collected, and
        # Windows won't delete a file with an open handle. Tests here deliberately
        # provoke refusals.
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.home = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.config = _config.load_config(self.home)
        install_call_llm(RecordingCallLlm(response=FakeResponse('{"memory": []}')))
        self.backends = []
        self.addCleanup(self._close_all)

    def _close_all(self):
        for backend in self.backends:
            try:
                _backend.release_backend(backend)
            except Exception:
                backend.close()

    def _new_backend(self, config=None, shared=False):
        config = config if config is not None else self.config
        backend = (
            _backend.acquire_backend(config)
            if shared
            else _backend.HermesRoutedMem0Backend(config)
        )
        self.backends.append(backend)
        return backend

    def _store_path(self):
        return _backend.local_store_path(self.config)

    def _foreign_client(self):
        """A second, independent Qdrant client on the same directory."""
        from qdrant_client import QdrantClient

        return QdrantClient(path=self._store_path())

    def test_lease_is_on_by_default_for_a_local_store(self):
        backend = self._new_backend()
        self.assertIsNotNone(backend._lease, "expected a lease for a local path store")

    def test_lock_is_released_between_operations(self):
        backend = self._new_backend()
        backend.add(
            [{"role": "user", "content": "Prefers dark roast"}],
            user_id="u", agent_id="hermes", infer=False,
        )
        # The whole point: another process can now take the directory.
        foreign = self._foreign_client()
        try:
            self.assertTrue(foreign.collection_exists("mem0_hermes"))
        finally:
            foreign.close()
        # And we can still use it afterwards.
        results = backend.search("coffee", filters={"user_id": "u"}, top_k=5)
        self.assertTrue(any("dark roast" in (r.get("memory") or "") for r in results))

    def test_each_operation_takes_a_fresh_lease(self):
        backend = self._new_backend()
        before = backend.leases
        backend.search("anything", filters={"user_id": "u"}, top_k=3)
        self.assertGreater(backend.leases, before)

    def test_two_backends_on_one_directory_both_work(self):
        # Without leasing, constructing the second one raises.
        first = self._new_backend()
        second = self._new_backend()
        first.add(
            [{"role": "user", "content": "Fact from the first backend"}],
            user_id="u", agent_id="hermes", infer=False,
        )
        second.add(
            [{"role": "user", "content": "Fact from the second backend"}],
            user_id="u", agent_id="hermes", infer=False,
        )
        found = " ".join(
            r.get("memory") or ""
            for r in first.search("fact", filters={"user_id": "u"}, top_k=10)
        )
        self.assertIn("first backend", found)
        self.assertIn("second backend", found)

    def test_a_brief_foreign_holder_is_waited_out(self):
        backend = self._new_backend()
        foreign = self._foreign_client()
        released = threading.Event()

        def release_soon():
            time.sleep(0.4)  # shorter than the retry budget
            foreign.close()
            released.set()

        thread = threading.Thread(target=release_soon, daemon=True)
        thread.start()
        try:
            # Retries with backoff rather than failing on the first conflict.
            results = backend.search("anything", filters={"user_id": "u"}, top_k=3)
            self.assertEqual(results, [])
            self.assertTrue(released.is_set(), "expected to wait for the holder")
        finally:
            thread.join(timeout=5)
            foreign.close()

    def test_a_permanent_foreign_holder_reports_how_to_fix_it(self):
        config = copy.deepcopy(self.config)
        config["concurrency"] = {"lock_retries": 1, "lock_retry_backoff": 0.05}
        backend = self._new_backend(config)
        foreign = self._foreign_client()
        try:
            with self.assertRaises(RuntimeError) as ctx:
                backend.search("anything", filters={"user_id": "u"}, top_k=3)
            message = str(ctx.exception)
            self.assertIn("held by another Qdrant client", message)
            self.assertIn("Qdrant server", message)
            self.assertIn("Do not delete the .lock file", message)
        finally:
            foreign.close()

    def test_lease_can_be_disabled(self):
        config = copy.deepcopy(self.config)
        config["concurrency"] = {"lease_local_store": False}
        backend = self._new_backend(config)
        self.assertIsNone(backend._lease)
        # Documents the trade-off: the directory stays locked for the process.
        with self.assertRaises(RuntimeError) as ctx:
            self._foreign_client()
        self.assertTrue(_backend.is_lock_conflict(ctx.exception))

    def test_process_shares_one_owner_per_directory(self):
        first = self._new_backend(shared=True)
        second = self._new_backend(shared=True)
        self.assertIs(first, second)
        self.assertEqual(first._refcount, 2)

    def test_shared_owner_survives_one_release(self):
        first = self._new_backend(shared=True)
        second = _backend.acquire_backend(self.config)
        _backend.release_backend(second)
        # Still usable: the remaining holder's store was not closed underneath it.
        self.assertEqual(first.search("x", filters={"user_id": "u"}, top_k=1), [])

    def test_a_server_url_is_never_leased_or_shared(self):
        config = copy.deepcopy(self.config)
        config["vector_store"] = {
            "provider": "qdrant",
            "config": {"url": "http://localhost:6333", "collection_name": "mem0_hermes"},
        }
        self.assertEqual(_backend.local_store_path(config), "")
        self.assertEqual(_backend._share_key(config), "")


_WORKER = '''
import importlib.util, json, os, sys, time, types
repo, home, tag = sys.argv[1], sys.argv[2], sys.argv[3]
hold_seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
sys.path.insert(0, os.path.join(repo, "hermes-agent"))
sys.path.insert(0, repo)
spec = importlib.util.spec_from_file_location(
    "mem0_hermes", os.path.join(repo, "__init__.py"), submodule_search_locations=[repo]
)
module = importlib.util.module_from_spec(spec)
sys.modules["mem0_hermes"] = module
spec.loader.exec_module(module)

# No real model call: stand in for the routed LLM.
aux = types.ModuleType("agent.auxiliary_client")
class _Response:
    def __init__(self):
        message = types.SimpleNamespace(content='{"memory": []}', tool_calls=None)
        self.choices = [types.SimpleNamespace(message=message)]
        self.model = "fake"
aux.call_llm = lambda **kwargs: _Response()
sys.modules["agent.auxiliary_client"] = aux

from mem0_hermes import _backend, _config
backend = _backend.HermesRoutedMem0Backend(_config.load_config(home))
try:
    if hold_seconds:
        # Hold the store open, as a non-leasing owner would.
        print("READY", flush=True)
        time.sleep(hold_seconds)
        print(json.dumps({"tag": tag, "held": hold_seconds}))
        raise SystemExit(0)
    for index in range(4):
        backend.add(
            [{"role": "user", "content": f"{tag} fact {index}"}],
            user_id="u", agent_id="hermes", infer=False,
        )
        hits = backend.search("fact", filters={"user_id": "u"}, top_k=50)
        time.sleep(0.05)
    print(json.dumps({"tag": tag, "leases": backend.leases, "visible": len(hits)}))
finally:
    backend.close()
'''


@unittest.skipUnless(
    _have("mem0") and _have("qdrant_client") and _have("fastembed"),
    "requires mem0ai + qdrant-client + fastembed",
)
class CrossProcessStoreTests(unittest.TestCase):
    """Two OS processes on one embedded store — the reported failure.

    Slowest test in the suite: it spawns two interpreters that each load Mem0
    and fastembed. It earns that by covering what in-process tests cannot —
    a genuinely separate process holding the same directory.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.home = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.worker = self.home / "worker.py"
        self.worker.write_text(_WORKER, encoding="utf-8")

    def _run_pair(self):
        import subprocess

        repo = str(_bootstrap.REPO_ROOT)
        env = dict(os.environ, MEM0_HERMES_NO_INSTALL="1")
        procs = [
            subprocess.Popen(
                [sys.executable, str(self.worker), repo, str(self.home), tag],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
            )
            for tag in ("A", "B")
        ]
        return [proc.communicate(timeout=180) + (proc.returncode,) for proc in procs]

    def test_two_processes_share_the_store_when_leasing(self):
        results = self._run_pair()
        for stdout, stderr, code in results:
            self.assertEqual(code, 0, f"worker failed:\n{stderr}")
            self.assertNotIn("already accessed by another instance", stderr)
        payloads = [json.loads(stdout.strip().splitlines()[-1]) for stdout, _e, _c in results]
        for payload in payloads:
            self.assertGreater(payload["leases"], 0, payload)
        # At least one process must observe the other's writes, which is only
        # possible because each lease reopens the store instead of holding a
        # snapshot for its lifetime.
        self.assertTrue(any(p["visible"] > 4 for p in payloads), payloads)

    def test_a_non_leasing_process_locks_everyone_out_with_a_clear_message(self):
        """The case leasing cannot fix, reported so the user can act.

        A process that holds the directory and never leases — the bundled
        ``mem0`` plugin, or this plugin with ``lease_local_store: false`` —
        keeps the lock for its lifetime. Nothing this plugin does can share it,
        so the requirement is that the error names the cause and the fix.
        """
        import subprocess

        (self.home / "mem0_hermes.json").write_text(
            json.dumps({"concurrency": {"lease_local_store": False}}), encoding="utf-8"
        )
        env = dict(os.environ, MEM0_HERMES_NO_INSTALL="1")
        holder = subprocess.Popen(
            [sys.executable, str(self.worker), str(_bootstrap.REPO_ROOT),
             str(self.home), "HOLDER", "20"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        )
        try:
            # Wait until the other process actually owns the directory.
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                line = holder.stdout.readline()
                if line.strip() == "READY":
                    break
                if holder.poll() is not None:
                    self.fail(f"holder exited early: {holder.stderr.read()}")
            else:
                self.fail("holder never became ready")

            config = _config.load_config(self.home)
            config["concurrency"] = {"lock_retries": 1, "lock_retry_backoff": 0.05}
            with self.assertRaises(RuntimeError) as ctx:
                _backend.HermesRoutedMem0Backend(config)
            message = str(ctx.exception)
            self.assertIn("held by another Qdrant client", message)
            self.assertIn("Qdrant server", message)
            self.assertIn("Do not delete the .lock file", message)
        finally:
            holder.kill()
            holder.communicate(timeout=30)


@unittest.skipUnless(
    _have("mem0") and _have("qdrant_client") and _have("fastembed"),
    "requires mem0ai + qdrant-client + fastembed",
)
class HistoryDatabaseTests(unittest.TestCase):
    """Mem0's history.db is opened by every Hermes process too."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.home = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.config = _config.load_config(self.home)
        install_call_llm(RecordingCallLlm(response=FakeResponse('{"memory": []}')))
        self.backend = None

    def tearDown(self):
        if self.backend is not None:
            self.backend.close()

    def _build(self, concurrency=None):
        config = copy.deepcopy(self.config)
        if concurrency is not None:
            config["concurrency"] = concurrency
        self.backend = _backend.HermesRoutedMem0Backend(config)
        return self.backend

    def _pragma(self, name):
        return self.backend.memory.db.connection.execute(f"PRAGMA {name}").fetchone()[0]

    def test_busy_timeout_is_raised_above_sqlites_default(self):
        self._build()
        # SQLite's own default is 5000 ms, which is a hard cliff for a
        # contended write; past it the memory operation fails outright.
        self.assertEqual(self._pragma("busy_timeout"), 15000)

    def test_busy_timeout_is_configurable(self):
        self._build({"sqlite_busy_timeout_ms": 250})
        self.assertEqual(self._pragma("busy_timeout"), 250)

    def test_junk_timeout_falls_back_to_the_default(self):
        self._build({"sqlite_busy_timeout_ms": "soon"})
        self.assertEqual(
            self._pragma("busy_timeout"), _backend.DEFAULT_SQLITE_BUSY_TIMEOUT_MS
        )

    def test_journal_mode_follows_hermes_policy(self):
        """Never enable WAL where Hermes itself refuses to."""
        try:
            from hermes_state import is_sqlite_wal_reset_vulnerable
        except ImportError:
            self.skipTest("hermes_state unavailable")
        self._build()
        mode = str(self._pragma("journal_mode")).lower()
        if is_sqlite_wal_reset_vulnerable():
            # This SQLite build can corrupt multi-process WAL databases.
            self.assertNotEqual(mode, "wal", "WAL enabled on a vulnerable build")
        else:
            self.assertEqual(mode, "wal")

    def test_wal_can_be_declined_entirely(self):
        self._build({"sqlite_wal": False, "sqlite_busy_timeout_ms": 1000})
        self.assertEqual(self._pragma("busy_timeout"), 1000)

    def test_timeout_decides_whether_a_contended_write_survives(self):
        """The knob has to actually change behaviour, not just report a value."""
        import sqlite3

        # A short ceiling gives up while another connection holds the write lock.
        self._build({"sqlite_busy_timeout_ms": 100})
        db_path = self.config["history_db_path"]
        # check_same_thread=False: the releasing thread below has to be able to
        # roll this back, or the "holder releases" half of the test silently
        # never releases and both halves fail for the wrong reason.
        blocker = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
        try:
            blocker.execute("BEGIN EXCLUSIVE")
            with self.assertRaises(sqlite3.OperationalError) as ctx:
                self.backend.memory.db.add_history("m1", None, "fact", "ADD")
            self.assertIn("locked", str(ctx.exception).lower())

            # A ceiling longer than the hold rides it out instead.
            self.backend.memory.db.connection.execute("PRAGMA busy_timeout=4000")
            released = threading.Event()

            def release():
                time.sleep(0.5)
                blocker.rollback()
                released.set()

            thread = threading.Thread(target=release, daemon=True)
            thread.start()
            try:
                self.backend.memory.db.add_history("m2", None, "fact", "ADD")
                self.assertTrue(released.is_set())
            finally:
                thread.join(timeout=10)
        finally:
            blocker.close()


class LockDiagnosticsTests(unittest.TestCase):
    def test_recognizes_qdrants_lock_error(self):
        self.assertTrue(
            _backend.is_lock_conflict(
                RuntimeError(
                    "Storage folder /x is already accessed by another instance of "
                    "Qdrant client. If you require concurrent access, use Qdrant "
                    "server instead."
                )
            )
        )

    def test_ignores_unrelated_errors(self):
        self.assertFalse(_backend.is_lock_conflict(RuntimeError("disk full")))

    def test_concurrency_settings_defaults_and_overrides(self):
        self.assertEqual(
            _backend._concurrency_settings({}),
            {"lease": True, "retries": 5, "backoff": 0.25},
        )
        self.assertEqual(
            _backend._concurrency_settings(
                {"concurrency": {"lease_local_store": "false", "lock_retries": "2",
                                 "lock_retry_backoff": "0.1"}}
            ),
            {"lease": False, "retries": 2, "backoff": 0.1},
        )
        # Junk values fall back rather than crashing a session.
        self.assertEqual(
            _backend._concurrency_settings(
                {"concurrency": {"lock_retries": "many", "lock_retry_backoff": None}}
            ),
            {"lease": True, "retries": 5, "backoff": 0.25},
        )


if __name__ == "__main__":
    unittest.main()
