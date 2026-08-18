# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Tests for the MemoryProvider implementation (lifecycle, tools, breaker)."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import _bootstrap  # noqa: F401 - sys.path bootstrap

from mem0_hermes import Mem0HermesMemoryProvider, _backend


class FakeBackend:
    """Records calls; raises on demand to drive the circuit breaker."""

    routing = "main Hermes model (auxiliary.mem0_hermes_extraction)"

    # The provider acquires backends through _backend.acquire_backend, which
    # shares one owner per storage path and refcounts it; a double has to carry
    # that bookkeeping.
    _share_key = ""
    _refcount = 0

    def __init__(self, config=None, memory=None, results=None, error=None):
        self.config = config
        self.results = results if results is not None else []
        self.error = error
        self.adds = []
        self.updates = []
        self.deletes = []
        self.searches = []
        self.closed = False

    def _maybe_raise(self):
        if self.error is not None:
            raise self.error

    def search(self, query, *, filters, top_k=10, rerank=False):
        self.searches.append((query, filters, top_k, rerank))
        self._maybe_raise()
        return self.results

    def add(self, messages, *, user_id, agent_id, infer=False, metadata=None):
        self._maybe_raise()
        self.adds.append(
            {
                "messages": messages,
                "user_id": user_id,
                "agent_id": agent_id,
                "infer": infer,
                "metadata": metadata,
            }
        )
        return {}

    def update(self, memory_id, text):
        self._maybe_raise()
        self.updates.append((memory_id, text))
        return {"result": "Memory updated.", "memory_id": memory_id}

    def delete(self, memory_id):
        self._maybe_raise()
        self.deletes.append(memory_id)
        return {"result": "Memory deleted.", "memory_id": memory_id}

    def close(self):
        self.closed = True


class BlockingBackend(FakeBackend):
    """Parks inside add() until released, so a write can be held in flight."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.entered = threading.Event()
        self.release = threading.Event()

    def add(self, messages, **kwargs):
        self.entered.set()
        self.release.wait(timeout=10)
        return super().add(messages, **kwargs)


class ProviderTestCase(unittest.TestCase):
    """Provider tests against a temporary HERMES_HOME and a fake backend."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self._saved_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(self.home)
        self.addCleanup(self._restore_home)
        # A local embedder needs no API key, so the credential gate is open.
        # Name the small default model explicitly: if a test ever reaches a real
        # backend build, a missing model would send Mem0's FastEmbedEmbedding to
        # its own default (thenlper/gte-large) and download 1.3 GB of weights.
        (self.home / "mem0_hermes.json").write_text(
            json.dumps(
                {
                    "user_id": "tester",
                    "embedder": {
                        "provider": "fastembed",
                        "config": {
                            "model": "BAAI/bge-small-en-v1.5",
                            "embedding_dims": 384,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        self._saved_backend_cls = _backend.HermesRoutedMem0Backend
        self.addCleanup(self._restore_backend_cls)

    def _restore_home(self):
        if self._saved_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self._saved_home

    def _restore_backend_cls(self):
        _backend.HermesRoutedMem0Backend = self._saved_backend_cls

    def wait_for_sync(self, provider, timeout=10.0):
        """Block until the extraction worker has drained everything queued."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if provider.sync_idle():
                return
            time.sleep(0.01)
        self.fail("queued turns were not extracted in time")

    def make_provider(self, backend=None, **init_kwargs):
        backend = backend if backend is not None else FakeBackend()
        _backend.HermesRoutedMem0Backend = lambda config: backend
        provider = Mem0HermesMemoryProvider()
        provider.initialize("session-1", hermes_home=str(self.home), **init_kwargs)
        self.addCleanup(provider.shutdown)
        return provider, backend


class AvailabilityTests(ProviderTestCase):
    def test_available_without_any_api_key(self):
        provider = Mem0HermesMemoryProvider()
        self.assertTrue(provider.is_available())

    def test_available_even_when_embedder_key_is_missing(self):
        # Silently skipping the provider would leave the user with no memory
        # tools and no explanation; initialize() reports the problem instead.
        (self.home / "mem0_hermes.json").write_text(
            json.dumps({"embedder": {"provider": "openai", "config": {}}}),
            encoding="utf-8",
        )
        saved = os.environ.pop("OPENAI_API_KEY", None)
        try:
            provider = Mem0HermesMemoryProvider()
            self.assertTrue(provider.is_available())
        finally:
            if saved is not None:
                os.environ["OPENAI_API_KEY"] = saved

    def test_missing_embedder_key_reported_through_tool_errors(self):
        (self.home / "mem0_hermes.json").write_text(
            json.dumps({"embedder": {"provider": "openai", "config": {}}}),
            encoding="utf-8",
        )
        saved = os.environ.pop("OPENAI_API_KEY", None)
        try:
            provider = Mem0HermesMemoryProvider()
            provider.initialize("session-1", hermes_home=str(self.home))
            payload = json.loads(provider.handle_tool_call("mem0_search", {"query": "x"}))
        finally:
            if saved is not None:
                os.environ["OPENAI_API_KEY"] = saved
        self.assertIn("OPENAI_API_KEY", payload["error"])
        self.assertIn("hermes memory setup", payload["error"])


class IdentityTests(ProviderTestCase):
    def test_configured_user_id_wins_over_gateway_id(self):
        provider, _backend_obj = self.make_provider(user_id="telegram-42")
        self.assertEqual(provider._user_id, "tester")

    def test_gateway_id_used_when_config_has_placeholder(self):
        (self.home / "mem0_hermes.json").write_text(
            json.dumps({"user_id": "hermes-user"}), encoding="utf-8"
        )
        provider, _backend_obj = self.make_provider(user_id="telegram-42")
        self.assertEqual(provider._user_id, "telegram-42")

    def test_channel_tags_writes(self):
        provider, backend = self.make_provider(platform="telegram")
        provider.handle_tool_call("mem0_add", {"content": "likes tea"})
        self.assertEqual(backend.adds[0]["metadata"], {"channel": "telegram"})

    def test_system_prompt_names_the_routing(self):
        provider, _backend_obj = self.make_provider()
        block = provider.system_prompt_block()
        self.assertIn("mem0_search", block)
        self.assertIn("main Hermes model", block)


class ToolTests(ProviderTestCase):
    def test_search_returns_normalized_results(self):
        backend = FakeBackend(
            results=[{"id": "1", "memory": "likes tea", "score": 0.9}]
        )
        provider, _ = self.make_provider(backend)
        payload = json.loads(provider.handle_tool_call("mem0_search", {"query": "tea"}))
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["results"][0]["memory"], "likes tea")
        self.assertEqual(backend.searches[0][1], {"user_id": "tester"})

    def test_search_top_k_is_clamped(self):
        provider, backend = self.make_provider()
        provider.handle_tool_call("mem0_search", {"query": "x", "top_k": 900})
        self.assertEqual(backend.searches[0][2], 50)
        provider.handle_tool_call("mem0_search", {"query": "x", "top_k": "junk"})
        self.assertEqual(backend.searches[1][2], 10)

    def test_search_without_results_says_so(self):
        provider, _backend_obj = self.make_provider()
        payload = json.loads(provider.handle_tool_call("mem0_search", {"query": "x"}))
        self.assertIn("No relevant memories", payload["result"])

    def test_add_is_verbatim(self):
        provider, backend = self.make_provider()
        provider.handle_tool_call("mem0_add", {"content": "likes tea"})
        # infer=False: an explicit save must not spend an LLM call.
        self.assertFalse(backend.adds[0]["infer"])
        self.assertEqual(backend.adds[0]["user_id"], "tester")

    def test_update_and_delete_pass_through(self):
        provider, backend = self.make_provider()
        provider.handle_tool_call("mem0_update", {"memory_id": "m1", "text": "new"})
        provider.handle_tool_call("mem0_delete", {"memory_id": "m1"})
        self.assertEqual(backend.updates, [("m1", "new")])
        self.assertEqual(backend.deletes, ["m1"])

    def test_missing_arguments_are_rejected(self):
        provider, _backend_obj = self.make_provider()
        for tool, args in (
            ("mem0_search", {}),
            ("mem0_add", {}),
            ("mem0_update", {"memory_id": "m1"}),
            ("mem0_delete", {}),
        ):
            payload = json.loads(provider.handle_tool_call(tool, args))
            self.assertIn("error", payload, tool)

    def test_unknown_tool(self):
        provider, _backend_obj = self.make_provider()
        payload = json.loads(provider.handle_tool_call("mem0_nope", {}))
        self.assertIn("Unknown tool", payload["error"])

    def test_not_found_errors_do_not_trip_the_breaker(self):
        backend = FakeBackend(error=ValueError("Memory with id m1 not found"))
        provider, _ = self.make_provider(backend)
        for _ in range(6):
            payload = json.loads(
                provider.handle_tool_call("mem0_update", {"memory_id": "m1", "text": "t"})
            )
            self.assertIn("not found", payload["error"].lower())
        self.assertFalse(provider._is_breaker_open())


class BreakerTests(ProviderTestCase):
    def test_breaker_opens_after_repeated_backend_failures(self):
        backend = FakeBackend(error=RuntimeError("upstream 503"))
        provider, _ = self.make_provider(backend)
        for _ in range(5):
            payload = json.loads(provider.handle_tool_call("mem0_add", {"content": "x"}))
            self.assertIn("Failed to store", payload["error"])
        self.assertTrue(provider._is_breaker_open())
        payload = json.loads(provider.handle_tool_call("mem0_search", {"query": "x"}))
        self.assertIn("temporarily unavailable", payload["error"])

    def test_success_resets_the_failure_count(self):
        backend = FakeBackend(error=RuntimeError("blip"))
        provider, _ = self.make_provider(backend)
        for _ in range(4):
            provider.handle_tool_call("mem0_add", {"content": "x"})
        backend.error = None
        provider.handle_tool_call("mem0_add", {"content": "x"})
        self.assertEqual(provider._consecutive_failures, 0)
        self.assertFalse(provider._is_breaker_open())


class TurnLifecycleTests(ProviderTestCase):
    def test_prefetch_injects_recalled_memories(self):
        backend = FakeBackend(results=[{"memory": "likes tea"}, {"memory": "lives in Berlin"}])
        provider, _ = self.make_provider(backend)
        provider.on_turn_start(1, "what tea do I like?")
        block = provider.prefetch("what tea do I like?")
        self.assertIn("## Mem0 Memory", block)
        self.assertIn("- likes tea", block)
        self.assertIn("- lives in Berlin", block)

    def test_prefetch_result_is_consumed_once(self):
        backend = FakeBackend(results=[{"memory": "likes tea"}])
        provider, _ = self.make_provider(backend)
        provider.on_turn_start(1, "tea?")
        self.assertIn("likes tea", provider.prefetch("tea?"))
        # Second call for the same query re-runs the search rather than
        # replaying a stale cached block.
        before = len(backend.searches)
        provider.prefetch("tea?")
        self.assertGreater(len(backend.searches), before)

    def test_prefetch_budget_is_spent_once_when_the_backend_is_slow(self):
        # The worker gives up waiting for a backend that is still building. It
        # must still publish that "nothing to inject" answer, or the next
        # prefetch() sees a finished-but-not-done worker, starts a second one,
        # and blocks the hot path for the whole budget a second time.
        import mem0_hermes as plugin

        provider = Mem0HermesMemoryProvider()
        self.addCleanup(provider.shutdown)
        provider._init_started = True  # initialize() ran; the build is in flight

        with mock.patch.object(plugin, "_PREFETCH_WAIT_SECS", 0.2):
            provider.on_turn_start(1, "tea?")
            provider._prefetch_thread.join(timeout=5)
            self.assertTrue(provider._prefetch_done)

            started = time.monotonic()
            self.assertEqual(provider.prefetch("tea?"), "")
            elapsed = time.monotonic() - started

        self.assertLess(
            elapsed, 0.2, "prefetch restarted the worker and waited a second time"
        )

    def test_prefetch_retries_on_a_later_turn(self):
        # Publishing the give-up must not latch: once the backend is up, the
        # next turn prefetches normally.
        backend = FakeBackend(results=[{"memory": "likes tea"}])
        provider, _ = self.make_provider(backend)
        provider.on_turn_start(1, "tea?")
        self.assertIn("likes tea", provider.prefetch("tea?"))
        provider.on_turn_start(2, "coffee?")
        self.assertIn("likes tea", provider.prefetch("coffee?"))

    def test_prefetch_failure_is_silent_but_counted(self):
        backend = FakeBackend(error=RuntimeError("vector store down"))
        provider, _ = self.make_provider(backend)
        provider.on_turn_start(1, "tea?")
        self.assertEqual(provider.prefetch("tea?"), "")
        self.assertGreaterEqual(provider._consecutive_failures, 1)

    def test_sync_turn_sends_both_roles_with_inference(self):
        provider, backend = self.make_provider()
        provider.sync_turn("I drink dark roast", "Noted.")
        self.wait_for_sync(provider)
        self.assertEqual(len(backend.adds), 1)
        call = backend.adds[0]
        self.assertTrue(call["infer"])  # extraction runs on the Hermes model
        self.assertEqual([m["role"] for m in call["messages"]], ["user", "assistant"])

    def test_sync_turn_does_not_block_the_caller(self):
        # The previous implementation joined the running sync thread for up to
        # 5s on the caller's thread and then discarded the turn.
        backend = BlockingBackend()
        provider, _ = self.make_provider(backend)
        provider.sync_turn("first", "ok")
        self.assertTrue(backend.entered.wait(timeout=5))

        started = time.monotonic()
        provider.sync_turn("second", "ok")
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.5)

        backend.release.set()
        self.wait_for_sync(provider)
        # ...and the turn that arrived mid-extraction was written, not dropped.
        self.assertEqual(len(backend.adds), 2)
        self.assertEqual(
            [call["messages"][0]["content"] for call in backend.adds],
            ["first", "second"],
        )

    def test_queue_drops_the_oldest_turn_when_extraction_falls_behind(self):
        import mem0_hermes as plugin

        backend = BlockingBackend()
        provider, _ = self.make_provider(backend)
        provider.sync_turn("in flight", "ok")
        self.assertTrue(backend.entered.wait(timeout=5))

        overflow = 2
        for index in range(plugin._SYNC_QUEUE_MAX + overflow):
            provider.sync_turn(f"turn {index}", "ok")
        backend.release.set()
        self.wait_for_sync(provider)

        written = [call["messages"][0]["content"] for call in backend.adds]
        self.assertEqual(provider._sync_dropped, overflow)
        self.assertEqual(written[0], "in flight")
        # The oldest queued turns go; the newest are kept and stay in order.
        self.assertNotIn("turn 0", written)
        self.assertNotIn("turn 1", written)
        self.assertEqual(written[1:], [
            f"turn {index}"
            for index in range(overflow, plugin._SYNC_QUEUE_MAX + overflow)
        ])

    def test_sync_turn_ignores_the_full_conversation_history(self):
        # MemoryProvider passes the entire conversation as of this turn --
        # earlier turns, assistant tool calls, tool results. Extracting from it
        # would re-mine the whole history every turn and read tool output as
        # facts about the user; only the turn's own content is written.
        provider, backend = self.make_provider()
        provider.sync_turn(
            "I drink dark roast",
            "Noted.",
            messages=[
                {"role": "user", "content": "an old turn"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
                {"role": "tool", "content": "/etc/passwd contents"},
                {"role": "user", "content": "I drink dark roast"},
                {"role": "assistant", "content": "Noted."},
            ],
        )
        self.wait_for_sync(provider)
        self.assertEqual(len(backend.adds), 1)
        self.assertEqual(
            backend.adds[0]["messages"],
            [
                {"role": "user", "content": "I drink dark roast"},
                {"role": "assistant", "content": "Noted."},
            ],
        )

    def test_sync_turn_skips_empty_turns(self):
        provider, backend = self.make_provider()
        provider.sync_turn("", "")
        self.assertIsNone(provider._sync_thread)
        self.assertEqual(backend.adds, [])

    def test_machine_driven_contexts_do_not_write(self):
        # A cron prompt extracted as a "user preference" would poison recall.
        for kwargs in ({"agent_context": "cron"}, {"platform": "cron"},
                       {"agent_context": "flush"}):
            provider, backend = self.make_provider(**kwargs)
            provider.sync_turn("scheduled job output", "done")
            self.assertIsNone(provider._sync_thread, kwargs)
            self.assertEqual(backend.adds, [], kwargs)

    def test_machine_driven_contexts_still_recall(self):
        backend = FakeBackend(results=[{"memory": "likes tea"}])
        provider, _ = self.make_provider(backend, agent_context="cron")
        provider.on_turn_start(1, "tea?")
        self.assertIn("likes tea", provider.prefetch("tea?"))

    def test_shutdown_closes_the_backend(self):
        provider, backend = self.make_provider()
        provider.shutdown()
        self.assertTrue(backend.closed)

    def test_reinitialize_does_not_leak_a_backend_reference(self):
        # acquire_backend refcounts one shared owner per storage path, so a
        # second initialize() hands back the same object with the count bumped.
        # If the first reference is never dropped the count never reaches zero,
        # shutdown() closes nothing, and the embedded Qdrant directory stays
        # locked against every other Hermes process until the interpreter exits.
        provider, backend = self.make_provider()
        key = _backend._share_key(provider._config)
        self.assertIs(_backend._OWNERS.get(key), backend)

        provider.initialize("session-2", hermes_home=str(self.home))
        provider._backend_thread.join(timeout=10)
        self.assertIs(provider._backend, backend)
        self.assertEqual(backend._refcount, 1)

        provider.shutdown()
        self.assertTrue(backend.closed)
        self.assertNotIn(key, _backend._OWNERS)


BUILD_SECONDS = 0.6


class BackgroundInitTests(ProviderTestCase):
    """initialize() runs on Hermes's session-startup path and must not stall it.

    Building the backend loads the embedder — ~1.5 s for fastembed's ONNX
    weights — so doing it inline delays every session start by that much.
    """

    def _slow_provider(self, backend=None, **init_kwargs):
        backend = backend if backend is not None else FakeBackend()

        def slow_build(config):
            # The delay has to be inside the factory: a pre-built double would
            # make the build instant and every assertion here vacuous.
            time.sleep(BUILD_SECONDS)
            return backend

        _backend.HermesRoutedMem0Backend = slow_build
        provider = Mem0HermesMemoryProvider()
        self.addCleanup(provider.shutdown)
        started = time.monotonic()
        provider.initialize("session-1", hermes_home=str(self.home), **init_kwargs)
        return provider, backend, time.monotonic() - started

    def test_initialize_returns_before_the_backend_is_built(self):
        provider, _backend_obj, elapsed = self._slow_provider()
        self.assertLess(
            elapsed, BUILD_SECONDS / 2,
            f"initialize() blocked for {elapsed:.2f}s on the backend build",
        )
        self.assertFalse(provider._backend_ready.is_set())

    def test_the_backend_arrives_shortly_after(self):
        provider, backend, _elapsed = self._slow_provider()
        self.assertIsNotNone(provider._await_backend(10))
        self.assertIs(provider._backend, backend)

    def test_a_tool_call_waits_for_the_backend_instead_of_failing(self):
        provider, _backend_obj, _elapsed = self._slow_provider()
        payload = json.loads(provider.handle_tool_call("mem0_add", {"content": "x"}))
        self.assertNotIn("error", payload)
        self.assertEqual(payload["result"], "Fact stored.")

    def test_a_tool_call_that_times_out_says_so(self):
        provider, _backend_obj, _elapsed = self._slow_provider()
        import mem0_hermes

        original = mem0_hermes._BACKEND_WAIT_SECS
        mem0_hermes._BACKEND_WAIT_SECS = 0.0
        try:
            payload = json.loads(provider.handle_tool_call("mem0_search", {"query": "x"}))
        finally:
            mem0_hermes._BACKEND_WAIT_SECS = original
        self.assertIn("still starting up", payload["error"])

    def test_first_turn_recall_still_works_while_building(self):
        # The prefetch worker waits for the backend, so a turn that starts
        # during the build isn't silently left without memory.
        provider, _backend_obj, _elapsed = self._slow_provider(
            FakeBackend(results=[{"memory": "likes tea"}])
        )
        provider.on_turn_start(1, "tea?")
        self.assertIn("likes tea", provider.prefetch("tea?"))

    def test_sync_turn_waits_for_the_backend(self):
        provider, backend, _elapsed = self._slow_provider()
        provider.sync_turn("I drink dark roast", "Noted.")
        self.wait_for_sync(provider)
        self.assertEqual(len(backend.adds), 1)

    def test_system_prompt_never_blocks(self):
        provider, _backend_obj, _elapsed = self._slow_provider()
        started = time.monotonic()
        block = provider.system_prompt_block()
        self.assertLess(time.monotonic() - started, BUILD_SECONDS / 2)
        self.assertIn("mem0_search", block)

    def test_background_init_can_be_disabled(self):
        (self.home / "mem0_hermes.json").write_text(
            json.dumps({"concurrency": {"background_init": False}}), encoding="utf-8"
        )
        provider, _backend_obj, elapsed = self._slow_provider()
        # Not an exact bound: time.sleep can return a few ms early (Windows
        # timer granularity). The claim is "it blocked for the build", not a
        # precise duration.
        self.assertGreaterEqual(elapsed, BUILD_SECONDS * 0.8)
        self.assertTrue(provider._backend_ready.is_set())

    def test_shutdown_during_a_build_does_not_hang(self):
        provider, backend, _elapsed = self._slow_provider()
        started = time.monotonic()
        provider.shutdown()
        self.assertLess(time.monotonic() - started, 10)
        self.assertTrue(backend.closed)


class BackupTests(ProviderTestCase):
    def test_default_paths_are_not_reported(self):
        # They live under HERMES_HOME, which `hermes backup` already walks.
        self.assertEqual(Mem0HermesMemoryProvider().backup_paths(), [])

    def test_relocated_state_is_reported(self):
        outside = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(outside, ignore_errors=True))
        (self.home / "mem0_hermes.json").write_text(
            json.dumps(
                {
                    "history_db_path": str(outside / "history.db"),
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {"path": str(outside / "qdrant")},
                    },
                }
            ),
            encoding="utf-8",
        )
        paths = Mem0HermesMemoryProvider().backup_paths()
        self.assertEqual(len(paths), 2, paths)
        self.assertTrue(any(p.endswith("history.db") for p in paths))


if __name__ == "__main__":
    unittest.main()
