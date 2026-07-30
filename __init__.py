# SPDX-License-Identifier: Apache-2.0 OR MIT
"""mem0_hermes — Mem0 memory with fact extraction on your Hermes model.

Drop-in alternative to the bundled ``mem0`` memory plugin. It uses the same
Mem0 OSS engine (vector store, dedup, semantic search) and exposes the same
four tools, but every LLM call Mem0 makes — fact extraction on each turn, the
add/update/delete decision pass — is routed through
``agent.auxiliary_client.call_llm`` instead of Mem0's built-in OpenAI client.
No ``OPENAI_API_KEY`` is needed for memory generation; extraction runs on
whatever ``hermes model`` is pointed at.

Embeddings are the one thing that still leaves Hermes: the agent has no
embedding path of its own, so ``embedder`` in the config is handed to Mem0
verbatim. Choose ``fastembed`` (local, no key) or ``ollama`` if you want zero
third-party API calls; see the README.

Configuration lives in ``$HERMES_HOME/mem0_hermes.json`` (see :mod:`._config`).
Activate with ``memory.provider: mem0_hermes`` in ``config.yaml``, or by running
``hermes memory setup``.

Lifecycle behavior (prefetch-with-short-wait, background turn sync, and the
consecutive-failure circuit breaker) follows the bundled ``plugins/memory/mem0``
provider so the two behave identically from the agent's point of view.
"""

from __future__ import annotations

import atexit
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# Circuit breaker: pause calls after this many consecutive failures.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120
# How long the hot path waits for an in-flight prefetch before giving up and
# leaving mem0_search as the backstop.
_PREFETCH_WAIT_SECS = 3

_CLIENT_ERROR_TYPES = ("MemoryNotFoundError", "ValidationError", "ValueError")


def _is_client_error(exc: Exception) -> bool:
    """True for user-caused errors (bad id, not found) — don't trip the breaker."""
    if type(exc).__name__ in _CLIENT_ERROR_TYPES:
        return True
    text = str(exc).lower()
    return "not found" in text or "404" in text or "valid uuid" in text


def _tool_error(message: str) -> str:
    try:
        from tools.registry import tool_error

        return tool_error(message)
    except Exception:
        return json.dumps({"error": message})


SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search the user's memories by meaning; returns facts ranked by "
        "relevance. Use this before answering any question that may depend on "
        "what you know about the user (preferences, facts, history, people, "
        "projects, past decisions). For multi-part or multi-hop questions, "
        "call it several times — vary the wording and run follow-up searches "
        "on what earlier results reveal; one search is rarely enough."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
        },
        "required": ["query"],
    },
}

ADD_SCHEMA = {
    "name": "mem0_add",
    "description": (
        "Store a durable fact about the user, verbatim (no LLM extraction). "
        "Call this the moment the user states a lasting preference, correction, "
        "decision, or personal detail worth recalling on future turns — don't "
        "wait to be asked to remember. Skip transient chit-chat and facts "
        "you've already stored."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to store."},
        },
        "required": ["content"],
    },
}

UPDATE_SCHEMA = {
    "name": "mem0_update",
    "description": (
        "Replace the text of an existing memory by its ID (take the ID from a "
        "mem0_search result). Use when a stored fact has changed or was wrong — "
        "correct it in place instead of adding a duplicate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory UUID to update."},
            "text": {"type": "string", "description": "New text content."},
        },
        "required": ["memory_id", "text"],
    },
}

DELETE_SCHEMA = {
    "name": "mem0_delete",
    "description": (
        "Delete a memory by its ID (take the ID from a mem0_search result). "
        "Use when a stored fact is obsolete or the user asks you to forget it; "
        "prefer mem0_update if the fact merely changed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory UUID to delete."},
        },
        "required": ["memory_id"],
    },
}


class Mem0HermesMemoryProvider(MemoryProvider):
    """Mem0 OSS memory whose extraction LLM is the configured Hermes model."""

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {}
        self._backend = None
        self._init_error = ""
        self._user_id = ""
        self._agent_id = "hermes"
        self._channel = "cli"
        self._agent_context = "primary"
        self._rerank_default = False
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_query = ""
        self._prefetch_result = ""
        self._prefetch_done = False
        self._prefetch_lock = threading.Lock()
        self._sync_thread: Optional[threading.Thread] = None
        self._sync_lock = threading.Lock()
        self._breaker_lock = threading.Lock()
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        self._atexit_registered = False

    # -- Identity / availability --------------------------------------------

    @property
    def name(self) -> str:
        return "mem0_hermes"

    def is_available(self) -> bool:
        """True whenever the config resolves — no credentials of our own needed.

        The routed LLM borrows the main Hermes provider's auth, so the only
        credential this plugin can be missing is the embedder's. That is
        deliberately NOT treated as unavailable: ``agent_init`` skips
        unavailable providers *silently*, leaving the user with no memory tools
        and no explanation. Activating and failing loudly in ``initialize()``
        (see :meth:`_create_backend`) is the diagnosable behavior.
        """
        from . import _config as cfgmod

        try:
            config = cfgmod.load_config()
        except Exception as exc:
            logger.warning("mem0_hermes: config could not be read: %s", exc)
            return False
        if not (config.get("vector_store") or {}).get("provider"):
            logger.warning("mem0_hermes: no vector_store configured")
            return False
        missing = cfgmod.embedder_key_missing(config)
        if missing:
            logger.warning(
                "mem0_hermes: embedder provider %r needs %s — memory writes will "
                "fail until it is set (or switch to a local embedder with "
                "`hermes memory setup`)",
                (config.get("embedder") or {}).get("provider"), missing,
            )
        return True

    # -- Setup --------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        from . import _config as cfgmod

        try:
            return cfgmod.config_schema()
        except Exception:
            return cfgmod.config_schema(cfgmod.default_config())

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        from pathlib import Path

        from . import _config as cfgmod

        cfgmod.save_config(values, Path(hermes_home))

    def backup_paths(self) -> List[str]:
        from . import _config as cfgmod

        try:
            return cfgmod.external_state_paths(cfgmod.load_config())
        except Exception:
            return []

    # -- Lifecycle ----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        from pathlib import Path

        from . import _config as cfgmod

        home = kwargs.get("hermes_home")
        self._config = cfgmod.load_config(Path(home) if home else None)
        self._user_id = cfgmod.resolved_user_id(self._config, kwargs.get("user_id") or "")
        self._agent_id = str(self._config.get("agent_id") or cfgmod.DEFAULT_AGENT_ID)
        self._channel = kwargs.get("platform") or "cli"
        # "primary" is a conversation with the user; "cron"/"flush"/"subagent"
        # turns are machine-driven and must not be written into the user's
        # memory (a cron prompt extracted as a "user preference" poisons
        # recall). Recall itself stays enabled — reading is always useful.
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        rerank = self._config.get("rerank", False)
        self._rerank_default = (
            rerank.lower() in ("true", "1", "yes") if isinstance(rerank, str) else bool(rerank)
        )

        self._backend = self._create_backend()
        if self._backend is not None and not self._atexit_registered:
            atexit.register(self._shutdown_backend)
            self._atexit_registered = True

    def _create_backend(self):
        from . import _backend as backend_mod
        from . import _config as cfgmod

        missing = cfgmod.embedder_key_missing(self._config)
        if missing:
            embedder = (self._config.get("embedder") or {}).get("provider")
            self._init_error = (
                f"embedder provider '{embedder}' needs {missing}. Embeddings do "
                "not route through Hermes — set that key, or run "
                "`hermes memory setup` and choose a local embedder "
                "(fastembed / ollama)."
            )
            logger.error("mem0_hermes: %s", self._init_error)
            return None

        try:
            # Shared per storage path within this process: embedded Qdrant
            # allows one live owner per directory, and a gateway can build
            # several providers.
            return backend_mod.acquire_backend(self._config)
        except Exception as exc:
            self._init_error = str(exc)
            logger.error("mem0_hermes: backend failed to initialize: %s", exc)
            return None

    def _shutdown_backend(self) -> None:
        backend, self._backend = self._backend, None
        if backend is None:
            return
        try:
            from . import _backend as backend_mod

            backend_mod.release_backend(backend)
        except Exception:
            try:
                backend.close()
            except Exception:
                pass

    def shutdown(self) -> None:
        for thread in (self._prefetch_thread, self._sync_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=5.0)
        self._shutdown_backend()

    # -- Circuit breaker ----------------------------------------------------

    def _is_breaker_open(self) -> bool:
        with self._breaker_lock:
            if self._consecutive_failures < _BREAKER_THRESHOLD:
                return False
            if time.monotonic() >= self._breaker_open_until:
                self._consecutive_failures = 0
                return False
            return True

    def _record_success(self) -> None:
        with self._breaker_lock:
            self._consecutive_failures = 0

    def _record_failure(self) -> None:
        with self._breaker_lock:
            self._consecutive_failures += 1
            tripped = self._consecutive_failures == _BREAKER_THRESHOLD
            if self._consecutive_failures >= _BREAKER_THRESHOLD:
                self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
        if tripped:
            logger.warning(
                "mem0_hermes: circuit breaker tripped after %d consecutive "
                "failures; pausing memory calls for %ds. Check the routed model "
                "(%s) and the embedder/vector store.",
                _BREAKER_THRESHOLD, _BREAKER_COOLDOWN_SECS, self._routing_label(),
            )

    def _routing_label(self) -> str:
        if self._backend is not None:
            try:
                return self._backend.routing
            except Exception:
                pass
        llm = (self._config.get("llm") or {}) if self._config else {}
        return f"{llm.get('provider') or 'auto'}/{llm.get('model') or 'main model'}"

    # -- Recall -------------------------------------------------------------

    def _read_filters(self) -> Dict[str, Any]:
        # user_id only, by design: recall spans every gateway and agent for
        # this principal. Writes still tag agent_id and metadata.channel.
        return {"user_id": self._user_id}

    def _write_metadata(self) -> Dict[str, Any]:
        return {"channel": self._channel} if self._channel else {}

    def system_prompt_block(self) -> str:
        return (
            "# Mem0 Memory (Hermes-routed)\n"
            f"Active. User: {self._user_id}. Fact extraction runs on "
            f"{self._routing_label()}.\n"
            "You have persistent memory of this user from past conversations. "
            "Call mem0_search before answering anything that could depend on "
            "prior context (the user's preferences, facts, history, people, "
            "projects, or earlier decisions) — do not rely on the chat window "
            "alone, and do not assume you have no memory.\n"
            "For multi-part or multi-hop questions, run several searches with "
            "different wording and follow up on what the first results surface; "
            "one search is rarely enough.\n"
            "Tools: mem0_search to find memories, mem0_add to store facts, "
            "mem0_update and mem0_delete to manage them by ID."
        )

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._start_prefetch(message)

    def _consume_prefetch_result(self, query: str) -> Optional[str]:
        with self._prefetch_lock:
            if self._prefetch_query != query or not self._prefetch_done:
                return None
            result = self._prefetch_result
            self._prefetch_result = ""
            self._prefetch_done = False
            return result

    def _start_prefetch(self, query: str) -> None:
        if not query or self._backend is None or self._is_breaker_open():
            return
        backend = self._backend
        with self._prefetch_lock:
            if self._prefetch_query == query:
                if self._prefetch_done:
                    return
                if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
                    return
            self._prefetch_query = query
            self._prefetch_result = ""
            self._prefetch_done = False

        def _run() -> None:
            body = ""
            try:
                results = backend.search(
                    query, filters=self._read_filters(), top_k=10, rerank=False
                )
                lines = [r.get("memory", "") for r in (results or []) if r.get("memory")]
                if lines:
                    body = "## Mem0 Memory\n" + "\n".join(f"- {line}" for line in lines)
                self._record_success()
            except Exception as exc:
                self._record_failure()
                logger.debug("mem0_hermes: prefetch failed: %s", exc)
            with self._prefetch_lock:
                if self._prefetch_query == query:
                    self._prefetch_result = body
                    self._prefetch_done = True

        thread = threading.Thread(target=_run, daemon=True, name="mem0-hermes-prefetch")
        with self._prefetch_lock:
            self._prefetch_thread = thread
        thread.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        cached = self._consume_prefetch_result(query)
        if cached is not None:
            return cached
        self._start_prefetch(query)
        with self._prefetch_lock:
            thread = self._prefetch_thread if self._prefetch_query == query else None
        if thread is not None:
            thread.join(timeout=_PREFETCH_WAIT_SECS)
        cached = self._consume_prefetch_result(query)
        return cached if cached is not None else ""

    # -- Write --------------------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Hand the turn to Mem0 for extraction on the Hermes model (async)."""
        if self._backend is None or self._is_breaker_open():
            return
        if not (user_content or assistant_content):
            return
        if self._agent_context != "primary" or self._channel == "cron":
            logger.debug(
                "mem0_hermes: skipping write (agent_context=%s, channel=%s)",
                self._agent_context, self._channel,
            )
            return

        backend = self._backend

        def _sync() -> None:
            try:
                backend.add(
                    [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ],
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    infer=True,
                    metadata=self._write_metadata(),
                )
                self._record_success()
            except Exception as exc:
                self._record_failure()
                logger.warning("mem0_hermes: turn sync failed: %s", exc)

        with self._sync_lock:
            if self._sync_thread is not None and self._sync_thread.is_alive():
                self._sync_thread.join(timeout=5.0)
            # Still running: skip rather than risk double ingestion of a turn.
            if self._sync_thread is not None and self._sync_thread.is_alive():
                logger.debug("mem0_hermes: previous sync still running; skipping turn")
                return
            self._sync_thread = threading.Thread(
                target=_sync, daemon=True, name="mem0-hermes-sync"
            )
            self._sync_thread.start()

    # -- Tools --------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, ADD_SCHEMA, UPDATE_SCHEMA, DELETE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if self._backend is None:
            return _tool_error(
                "mem0_hermes backend not initialized: "
                f"{self._init_error or 'unknown error'}"
            )
        if self._is_breaker_open():
            return _tool_error(
                "Memory temporarily unavailable (multiple consecutive failures). "
                "Will retry automatically."
            )

        if tool_name == "mem0_search":
            query = str(args.get("query") or "")
            if not query:
                return _tool_error("Missing required parameter: query")
            try:
                top_k = max(1, min(int(args.get("top_k", 10)), 50))
            except (TypeError, ValueError):
                top_k = 10
            try:
                results = self._backend.search(
                    query,
                    filters=self._read_filters(),
                    top_k=top_k,
                    rerank=self._rerank_default,
                )
                self._record_success()
            except Exception as exc:
                if not _is_client_error(exc):
                    self._record_failure()
                return _tool_error(f"Search failed: {exc}")
            if not results:
                return json.dumps({"result": "No relevant memories found."})
            items = [
                {
                    "id": r.get("id"),
                    "memory": r.get("memory", ""),
                    "score": r.get("score", 0),
                }
                for r in results
            ]
            return json.dumps({"results": items, "count": len(items)})

        if tool_name == "mem0_add":
            content = str(args.get("content") or "")
            if not content:
                return _tool_error("Missing required parameter: content")
            try:
                self._backend.add(
                    [{"role": "user", "content": content}],
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    infer=False,  # verbatim — no LLM call for explicit saves
                    metadata=self._write_metadata(),
                )
                self._record_success()
            except Exception as exc:
                self._record_failure()
                return _tool_error(f"Failed to store: {exc}")
            return json.dumps({"result": "Fact stored."})

        if tool_name == "mem0_update":
            memory_id = str(args.get("memory_id") or "")
            text = str(args.get("text") or "")
            if not memory_id:
                return _tool_error("Missing required parameter: memory_id")
            if not text:
                return _tool_error("Missing required parameter: text")
            try:
                result = self._backend.update(memory_id, text)
                self._record_success()
            except Exception as exc:
                if _is_client_error(exc):
                    return _tool_error(f"Memory not found: {memory_id}")
                self._record_failure()
                return _tool_error(f"Update failed: {exc}")
            return json.dumps(result)

        if tool_name == "mem0_delete":
            memory_id = str(args.get("memory_id") or "")
            if not memory_id:
                return _tool_error("Missing required parameter: memory_id")
            try:
                result = self._backend.delete(memory_id)
                self._record_success()
            except Exception as exc:
                if _is_client_error(exc):
                    return _tool_error(f"Memory not found: {memory_id}")
                self._record_failure()
                return _tool_error(f"Delete failed: {exc}")
            return json.dumps(result)

        return _tool_error(f"Unknown tool: {tool_name}")


def register(ctx) -> None:
    """Register this plugin as the active memory provider."""
    # Declaring the auxiliary task makes it appear in
    # `hermes model → Configure auxiliary models`, so the extraction model can
    # be pinned separately from the chat model. The memory-plugin loader passes
    # a minimal context object that has no such method — hence the guard.
    register_task = getattr(ctx, "register_auxiliary_task", None)
    if callable(register_task):
        try:
            from ._config import DEFAULT_AUX_TASK

            register_task(
                key=DEFAULT_AUX_TASK,
                display_name="Mem0 memory extraction",
                description="mem0_hermes fact extraction / update decisions",
                defaults={"provider": "auto", "timeout": 120},
            )
        except Exception as exc:
            logger.debug("mem0_hermes: auxiliary task registration skipped: %s", exc)
    ctx.register_memory_provider(Mem0HermesMemoryProvider())
