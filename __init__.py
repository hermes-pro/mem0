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
provider so the two behave identically from the agent's point of view, with two
departures:

* The backend is built on a background thread, because ``initialize()`` runs
  inline on Hermes's session-startup path and building it costs ~1.5 s — mostly
  importing ``qdrant_client`` and ``fastembed``, not the embedding model, which
  is ~200 ms of that. Consumers wait for readiness where it matters instead:
  prefetch inside its existing budget, tool calls and turn sync in their own
  threads.
* Turn sync is queued to one long-lived worker rather than started per turn.
  Extraction is an LLM call, so a burst of short turns outruns it; the bundled
  provider discards the turns that arrive meanwhile, which loses memories from
  exactly the fast back-and-forth where they matter. Queued turns are extracted
  in order and only dropped once the backlog exceeds ``_SYNC_QUEUE_MAX``.
"""

from __future__ import annotations

import atexit
import json
import logging
import re
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# Circuit breaker: pause calls after this many consecutive failures.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120
# How long the hot path waits for an in-flight prefetch before giving up and
# leaving mem0_search as the backstop.
_PREFETCH_WAIT_SECS = 3
# How long a tool call or a turn sync waits for the backend to finish building.
# Generous because it only applies while the backend builds (~1.5 s, mostly
# Mem0/qdrant-client/fastembed imports) and the alternative is telling the model
# memory is unavailable.
_BACKEND_WAIT_SECS = 30

# Turns waiting to be extracted. Extraction is an LLM call, so a burst of short
# turns can outrun it; queueing lets them catch up instead of being thrown away.
# Bounded because the alternative to dropping under sustained overload is
# unbounded memory growth and an ever-staler backlog — and the oldest turn is
# the least useful one to keep.
_SYNC_QUEUE_MAX = 8
# How long shutdown waits for the worker to finish the write in flight.
_SYNC_SHUTDOWN_WAIT_SECS = 5.0

_CLIENT_ERROR_TYPES = ("MemoryNotFoundError", "ValidationError", "ValueError")
# Anchored on word boundaries so it matches a status code and not the "404" that
# happens to sit inside a byte count, a port, an id or a timestamp. Classifying
# an infrastructure failure as user error is not cosmetic: it stops the failure
# from counting toward the circuit breaker, so a vector store that is genuinely
# down keeps being retried on every turn instead of backing off.
_NOT_FOUND_RE = re.compile(r"\bnot found\b|\b404\b|\bvalid uuid\b")


def _is_client_error(exc: Exception) -> bool:
    """True for user-caused errors (bad id, not found) — don't trip the breaker."""
    if type(exc).__name__ in _CLIENT_ERROR_TYPES:
        return True
    return _NOT_FOUND_RE.search(str(exc).lower()) is not None


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
        self._init_started = False
        self._backend_ready = threading.Event()
        self._backend_thread: Optional[threading.Thread] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_query = ""
        self._prefetch_result = ""
        self._prefetch_count = 0
        self._prefetch_done = False
        self._last_recall_count = 0
        self._prefetch_lock = threading.Lock()
        self._sync_thread: Optional[threading.Thread] = None
        self._sync_cond = threading.Condition()
        self._sync_queue: Deque[Tuple[str, str]] = deque()
        self._sync_busy = False
        self._sync_stopping = False
        self._sync_dropped = 0
        self._breaker_lock = threading.Lock()
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        self._atexit_registered = False
        self._unavailable_reason = ""

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

        self._unavailable_reason = ""
        try:
            config = cfgmod.load_config()
        except Exception as exc:
            self._unavailable_reason = (
                f"{cfgmod.CONFIG_FILENAME} in {cfgmod.hermes_home()} could not "
                f"be read ({exc}). Fix the JSON, or delete it and run "
                "`hermes memory setup`."
            )
            logger.warning("mem0_hermes: config could not be read: %s", exc)
            return False
        if not (config.get("vector_store") or {}).get("provider"):
            self._unavailable_reason = (
                "no vector_store.provider is configured; run "
                "`hermes memory setup` to pick one."
            )
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

    def unavailable_reason(self) -> str:
        """Why :meth:`is_available` said no, for the caller's warning.

        An unavailable provider is never initialized, so nothing this plugin
        would log from ``initialize()`` is reachable — without this the user
        sees "provider unavailable" and has to guess between a malformed config
        file and an unconfigured vector store.
        """
        return self._unavailable_reason

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

        concurrency = self._config.get("concurrency")
        concurrency = concurrency if isinstance(concurrency, dict) else {}
        background = concurrency.get("background_init", True)
        if isinstance(background, str):
            background = background.strip().lower() not in ("false", "0", "no")

        self._backend_ready.clear()
        self._init_started = True
        with self._sync_cond:
            # A provider reused after shutdown() (the gateway rebuilds one on
            # provider error) must accept turns again.
            self._sync_stopping = False
        if background:
            # Building the backend costs ~1.5 s — dominated by importing
            # qdrant_client (~1.25 s, pydantic model construction) and fastembed
            # (~0.7 s), with the embedding model itself only ~200 ms — and Hermes
            # calls initialize() inline on the session startup path
            # (agent_init.py). Doing it here would stall every session start. Consumers wait for readiness where
            # it actually matters: prefetch inside its existing budget, tool calls
            # and turn sync in their own threads.
            self._backend_thread = threading.Thread(
                target=self._build_backend, daemon=True, name="mem0-hermes-init",
            )
            self._backend_thread.start()
        else:
            self._build_backend()

    def _build_backend(self) -> None:
        """Construct the backend and publish it to whoever is waiting."""
        backend = None
        try:
            backend = self._create_backend()
        finally:
            previous, self._backend = self._backend, backend
            self._backend_ready.set()
        # A second initialize() (a gateway starting another session, a rebuild
        # after a provider error) takes a second reference from
        # acquire_backend's registry — usually to the very same shared object,
        # since sharing is keyed on the storage path. Dropping the old
        # reference here is what keeps this provider holding exactly one:
        # without it the refcount never reaches zero, shutdown() closes
        # nothing, and the embedded Qdrant directory stays locked against every
        # other Hermes process until the interpreter exits. Released after the
        # swap, so nothing waiting on _backend_ready can observe a released
        # backend.
        if previous is not None:
            self._release(previous)
        if backend is not None and not self._atexit_registered:
            atexit.register(self._shutdown_backend)
            self._atexit_registered = True

    def _await_backend(self, timeout: float):
        """Return the backend once built, or ``None`` if it isn't ready in time.

        Never raises: callers treat ``None`` as "memory unavailable right now",
        which is the same path a failed build already takes.
        """
        if self._backend_ready.is_set():
            return self._backend
        if not self._init_started:
            # Nothing is building, so waiting would burn the full timeout for a
            # result that can never arrive.
            return None
        self._backend_ready.wait(timeout=max(0.0, timeout))
        return self._backend

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

    def _release(self, backend) -> None:
        """Drop one reference to ``backend``; the registry closes it at zero."""
        try:
            from . import _backend as backend_mod

            backend_mod.release_backend(backend)
        except Exception:
            try:
                backend.close()
            except Exception:
                pass

    def _shutdown_backend(self) -> None:
        backend, self._backend = self._backend, None
        if backend is None:
            return
        self._release(backend)

    def shutdown(self) -> None:
        # The builder first: closing a store that is still being constructed
        # would race the construction, and the workers below may be waiting on it.
        if self._backend_thread is not None and self._backend_thread.is_alive():
            self._backend_thread.join(timeout=_BACKEND_WAIT_SECS)
        # Before joining the workers, not after: a worker parked in
        # _await_backend would otherwise sit out its full timeout while
        # shutdown waits on it. Set, they see no backend and give up.
        self._backend_ready.set()
        self._stop_sync_worker()
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=5.0)
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

    def _consume_prefetch_result(self, query: str) -> Optional[Tuple[str, int]]:
        """The prefetched block and how many memories it holds, once only."""
        with self._prefetch_lock:
            if self._prefetch_query != query or not self._prefetch_done:
                return None
            result = (self._prefetch_result, self._prefetch_count)
            self._prefetch_result = ""
            self._prefetch_count = 0
            self._prefetch_done = False
            return result

    def _start_prefetch(self, query: str) -> None:
        # Deliberately does not require the backend to exist yet: the worker
        # waits for it, so a first turn that starts while the embedder is still
        # loading still gets recall (within prefetch's budget) instead of
        # silently returning nothing.
        if not query or self._is_breaker_open():
            return
        with self._prefetch_lock:
            if self._prefetch_query == query:
                if self._prefetch_done:
                    return
                if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
                    return
            self._prefetch_query = query
            self._prefetch_result = ""
            self._prefetch_count = 0
            self._prefetch_done = False

        def _run() -> None:
            body = ""
            count = 0
            try:
                backend = self._await_backend(_PREFETCH_WAIT_SECS)
                if backend is not None:
                    results = backend.search(
                        query, filters=self._read_filters(), top_k=10, rerank=False
                    )
                    lines = [
                        r.get("memory", "") for r in (results or []) if r.get("memory")
                    ]
                    if lines:
                        body = "## Mem0 Memory\n" + "\n".join(
                            f"- {line}" for line in lines
                        )
                        count = len(lines)
                    self._record_success()
            except Exception as exc:
                self._record_failure()
                logger.debug("mem0_hermes: prefetch failed: %s", exc)
            finally:
                # Unconditional, including the backend-not-ready path: an empty
                # body IS the answer ("nothing to inject, fall back to
                # mem0_search"). Leaving the flag unset instead makes the next
                # prefetch() see a finished-but-not-done worker and start a
                # second one, spending _PREFETCH_WAIT_SECS all over again on
                # the hot path.
                with self._prefetch_lock:
                    if self._prefetch_query == query:
                        self._prefetch_result = body
                        self._prefetch_count = count
                        self._prefetch_done = True

        thread = threading.Thread(target=_run, daemon=True, name="mem0-hermes-prefetch")
        with self._prefetch_lock:
            self._prefetch_thread = thread
        thread.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        cached = self._consume_prefetch_result(query)
        if cached is None:
            self._start_prefetch(query)
            with self._prefetch_lock:
                thread = self._prefetch_thread if self._prefetch_query == query else None
            if thread is not None:
                thread.join(timeout=_PREFETCH_WAIT_SECS)
            cached = self._consume_prefetch_result(query)
        body, count = cached if cached is not None else ("", 0)
        # Set on every path, including the empty ones: recall_status() must
        # describe THIS turn, and a count left over from an earlier turn would
        # show the user an indicator for memories that were not injected.
        self._last_recall_count = count
        return body

    def recall_status(self):
        """What the last :meth:`prefetch` injected, for the agent's indicator."""
        if not self._last_recall_count:
            return None
        try:
            from agent.memory_provider import RecallStatus
        except ImportError:
            # Older Hermes without the indicator: nothing to report to.
            return None
        return RecallStatus(provider_label="Mem0", count=self._last_recall_count)

    # -- Write --------------------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Hand the turn to Mem0 for extraction on the Hermes model (async).

        ``messages`` is accepted to satisfy the ``MemoryProvider`` contract and
        deliberately ignored. It is the whole OpenAI-style conversation as of
        this turn — every earlier turn, plus assistant tool calls and tool
        results — not the turn alone. Passing it to ``Memory.add`` would
        re-extract the entire history on every turn (quadratic model spend, and
        Mem0's dedup pass asked to re-decide every prior fact each time), and
        would mine tool output for "user facts": a file listing or an API
        response read as something the user said about themselves. The
        user/assistant pair is the turn's actual new content, so that is what
        gets extracted. The ABC explicitly allows ignoring it.
        """
        if self._is_breaker_open():
            return
        if not (user_content or assistant_content):
            return
        if self._agent_context != "primary" or self._channel == "cron":
            logger.debug(
                "mem0_hermes: skipping write (agent_context=%s, channel=%s)",
                self._agent_context, self._channel,
            )
            return
        if self._backend is None and self._backend_ready.is_set():
            return  # the build finished and failed; nothing to write to

        with self._sync_cond:
            if self._sync_stopping:
                return
            if len(self._sync_queue) >= _SYNC_QUEUE_MAX:
                # Oldest first: it is the least useful turn to keep, and the
                # backlog is already staler than whatever is arriving now.
                self._sync_queue.popleft()
                self._sync_dropped += 1
                logger.warning(
                    "mem0_hermes: extraction is %d turns behind; dropped the "
                    "oldest queued turn (%d dropped this session). The routed "
                    "model (%s) is slower than the conversation.",
                    _SYNC_QUEUE_MAX, self._sync_dropped, self._routing_label(),
                )
            self._sync_queue.append((user_content, assistant_content))
            self._ensure_sync_worker()
            self._sync_cond.notify()

    def _ensure_sync_worker(self) -> None:
        """Start the extraction worker if it isn't running. Call under the lock."""
        if self._sync_thread is not None and self._sync_thread.is_alive():
            return
        self._sync_thread = threading.Thread(
            target=self._sync_worker, daemon=True, name="mem0-hermes-sync"
        )
        self._sync_thread.start()

    def _sync_worker(self) -> None:
        """Drain queued turns one at a time, in order.

        One worker rather than one thread per turn: Mem0's add is a
        read-decide-write cycle over a shared store, so overlapping extractions
        can each work from a snapshot taken before the other's write and
        duplicate a fact.
        """
        while True:
            with self._sync_cond:
                while not self._sync_queue and not self._sync_stopping:
                    self._sync_cond.wait()
                if self._sync_stopping:
                    return
                turn = self._sync_queue.popleft()
                self._sync_busy = True
            try:
                self._write_turn(turn)
            except Exception as exc:  # pragma: no cover - _write_turn catches
                logger.warning("mem0_hermes: sync worker error: %s", exc)
            finally:
                with self._sync_cond:
                    self._sync_busy = False
                    self._sync_cond.notify_all()

    def _write_turn(self, turn: Tuple[str, str]) -> None:
        user_content, assistant_content = turn
        backend = self._await_backend(_BACKEND_WAIT_SECS)
        if backend is None:
            return
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

    def _stop_sync_worker(self) -> None:
        with self._sync_cond:
            self._sync_stopping = True
            thread = self._sync_thread
            self._sync_cond.notify_all()
        if thread is not None and thread.is_alive():
            thread.join(timeout=_SYNC_SHUTDOWN_WAIT_SECS)
        with self._sync_cond:
            pending = len(self._sync_queue)
            self._sync_queue.clear()
        if pending:
            logger.warning(
                "mem0_hermes: %d queued turn(s) were not extracted before "
                "shutdown", pending,
            )

    def sync_idle(self) -> bool:
        """True when nothing is queued and no extraction is in flight."""
        with self._sync_cond:
            return not self._sync_queue and not self._sync_busy

    # -- Tools --------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, ADD_SCHEMA, UPDATE_SCHEMA, DELETE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        # Tool calls run inline on the agent loop (tool_executor.py), so this is
        # the one place a wait is visible to the user. It only bites on a call
        # that lands while the embedder is still loading; afterwards the backend
        # is already published and this returns immediately.
        backend = self._await_backend(_BACKEND_WAIT_SECS)
        if backend is None:
            if self._init_started and not self._backend_ready.is_set():
                return _tool_error(
                    "Memory is still starting up (loading the embedding model). "
                    "Try again in a moment."
                )
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
                results = backend.search(
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
                backend.add(
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
                result = backend.update(memory_id, text)
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
                result = backend.delete(memory_id)
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
