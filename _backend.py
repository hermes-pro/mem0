# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Builds a Mem0 OSS ``Memory`` whose LLM is the Hermes-routed adapter.

Two obstacles have to be worked around to inject a custom LLM into Mem0 2.x:

1. ``mem0.llms.configs.LlmConfig`` validates ``provider`` against a hardcoded
   allowlist, so ``Memory.from_config({"llm": {"provider": "hermes_routed"}})``
   raises ``ValidationError`` before the factory is ever consulted.
2. ``Memory.__init__`` builds the LLM itself (``LlmFactory.create(...)``), and
   the default ``openai`` branch constructs an ``openai.OpenAI`` client eagerly
   — which raises when no ``OPENAI_API_KEY`` exists.

So the config is validated with the allowlisted ``openai`` provider and the
provider id is then reassigned to the registered ``hermes_routed`` entry (Mem0's
Pydantic models don't validate on assignment). If a future Mem0 turns
``validate_assignment`` on, the fallback path builds ``Memory`` normally with a
placeholder key and swaps ``memory.llm`` afterwards. Either way the returned
``Memory`` ends up with a :class:`~._hermes_llm.HermesRoutedLLM`, verified
before it is handed back.
"""

from __future__ import annotations

import copy
import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _fastembed_cache_dir() -> Path:
    """A durable cache location for fastembed's ONNX weights.

    fastembed defaults to ``%TEMP%/fastembed_cache`` (or the platform temp dir),
    where a Disk Cleanup run silently deletes the weights and the next session
    re-downloads them — or fails, if the box is offline by then. This keeps them
    in the user's cache directory instead: durable, shared across Hermes
    profiles, and deliberately NOT under HERMES_HOME, which ``hermes backup``
    walks (nobody wants 65 MB of model weights in every backup).
    """
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(root) / "fastembed"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "fastembed"


def _prepare_environment(config: Dict[str, Any]) -> None:
    """Set Mem0/fastembed env knobs that are only read at import time."""
    if not config.get("telemetry", False):
        # Read once in mem0.memory.telemetry at import; must be set before the
        # first `import mem0`. setdefault so an explicit user value wins.
        os.environ.setdefault("MEM0_TELEMETRY", "false")

    if (config.get("embedder") or {}).get("provider") == "fastembed":
        # Mem0 passes only ``model_name`` to fastembed, so the cache location has
        # to come from the environment. setdefault: an operator who already set
        # FASTEMBED_CACHE_PATH keeps their choice.
        cache = _fastembed_cache_dir()
        try:
            cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("FASTEMBED_CACHE_PATH", str(cache))
        except OSError as exc:
            logger.debug("mem0_hermes: leaving fastembed's default cache: %s", exc)


def _ensure_mem0_installed() -> None:
    """Lazy-install ``mem0ai`` through Hermes's gated installer if missing."""
    try:
        import mem0  # noqa: F401

        return
    except ImportError:
        pass
    try:
        from tools.lazy_deps import ensure

        # Shares the bundled mem0 plugin's allowlist entry — same package.
        ensure("memory.mem0", prompt=False)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.debug("mem0_hermes: lazy install of mem0ai unavailable: %s", exc)


def _ensure_embedder_installed(config: Dict[str, Any]) -> None:
    """Install the selected embedder's packages if they went missing.

    ``hermes memory setup`` already installs them at selection time, so this is
    the repair path: a rebuilt venv (``hermes update``) drops them, and
    ``hermes plugins update`` only refreshes what ``plugin.yaml`` declares. Left
    alone, the next session would fail on ``import fastembed`` with no hint that
    a reinstall fixes it.
    """
    from . import _config as cfgmod

    if not cfgmod.embedder_pip_requirements(config):
        return
    provider = cfgmod.embedder_provider(config)
    logger.warning(
        "mem0_hermes: embedder '%s' is selected but its packages are missing; "
        "installing (first run may take a minute)", provider,
    )
    ok, message = cfgmod.ensure_embedder_dependencies(config)
    if not ok:
        # Raised, not swallowed: Memory() would otherwise fail deeper in Mem0
        # with an ImportError that doesn't mention how to fix it.
        raise RuntimeError(message)
    if message:
        logger.info("mem0_hermes: %s", message)


def _reconcile_embedding_dims(config: Dict[str, Any]) -> None:
    """Trust fastembed's registry over our table for the vector width.

    A wrong ``embedding_dims`` doesn't fail loudly — it creates a vector
    collection at the wrong width, and every write then fails on a dimension
    mismatch. Once fastembed is importable, its own model registry settles it.
    Mutates ``config`` in place so the corrected width reaches both the vector
    store config and ``_reset_collection_if_dims_changed``.
    """
    from . import _config as cfgmod

    changed, note = cfgmod.sync_fastembed_dims(config)
    if not note:
        return
    if changed:
        logger.info("mem0_hermes: %s", note)
    else:
        logger.warning("mem0_hermes: %s", note)


def _expand(value: Any) -> Any:
    return os.path.expanduser(str(value)) if isinstance(value, str) else value


def build_memory_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Translate ``mem0_hermes.json`` into a Mem0 ``MemoryConfig`` dict."""
    llm_block = dict(config.get("llm") or {})
    embedder = copy.deepcopy(config.get("embedder") or {})
    vector_store = copy.deepcopy(config.get("vector_store") or {})

    vs_config = dict(vector_store.get("config") or {})
    if vs_config.get("path"):
        vs_config["path"] = _expand(vs_config["path"])
        Path(vs_config["path"]).parent.mkdir(parents=True, exist_ok=True)

    # Qdrant needs the vector width at collection-creation time; Mem0 only
    # infers it for models it knows.
    dims = (embedder.get("config") or {}).get("embedding_dims")
    if dims and not vs_config.get("embedding_model_dims"):
        vs_config["embedding_model_dims"] = int(dims)
    vector_store["config"] = vs_config

    history_db_path = _expand(config.get("history_db_path") or "")
    if history_db_path:
        Path(history_db_path).parent.mkdir(parents=True, exist_ok=True)

    memory_config: Dict[str, Any] = {
        # Validated as "openai" and reassigned in build_memory(); the config
        # dict itself is what HermesRoutedLLM receives.
        "llm": {"provider": "openai", "config": llm_block},
        "embedder": embedder,
        "vector_store": vector_store,
        "version": "v1.1",
    }
    if history_db_path:
        memory_config["history_db_path"] = history_db_path
    if config.get("custom_instructions"):
        memory_config["custom_instructions"] = config["custom_instructions"]
    if config.get("reranker"):
        # Passed to Mem0 verbatim. An ``llm_reranker`` whose nested llm block
        # names the registered ``hermes_routed`` provider goes through Hermes
        # too, since Mem0 builds it with the same LlmFactory (see README).
        memory_config["reranker"] = copy.deepcopy(config["reranker"])
    return memory_config


# ---------------------------------------------------------------------------
# Local Qdrant is single-owner
# ---------------------------------------------------------------------------
#
# ``QdrantClient(path=...)`` is not a server: it is qdrant-client's embedded
# QdrantLocal, which takes an EXCLUSIVE, non-blocking portalocker lock on
# ``<path>/.lock`` for the lifetime of the client object and raises
#
#   RuntimeError: Storage folder <path> is already accessed by another instance
#   of Qdrant client. If you require concurrent access, use Qdrant server
#   instead.
#
# for anyone else. The lock is deliberate: QdrantLocal loads collection state
# into the owning process, so two live instances would each mutate the directory
# from their own stale snapshot and lose writes. Hermes routinely has more than
# one candidate owner — CLI, gateway, Desktop, cron, or simply two provider
# objects in one process.
#
# Three mitigations, in order of how much they buy:
#
#   1. One owner per storage path *per process* (_acquire_shared / registry
#      below): removes the self-inflicted case entirely.
#   2. Per-operation leasing (_StoreLease): hold the OS lock only while an
#      operation runs, so cooperating processes interleave instead of the first
#      one to start owning the directory forever. Costs one QdrantLocal reopen
#      per operation — measured at ~10 ms per 100 points, ~45 ms per 1000.
#   3. Bounded retry with backoff around acquisition, so a brief overlap (a cron
#      run finishing, another process between leases) waits instead of failing.
#
# What none of this can fix: a process that holds the directory permanently and
# does not lease — for example the bundled ``mem0`` plugin, or this plugin with
# ``concurrency.lease_local_store: false``. Leasing only works when every
# participant leases. For genuinely concurrent multi-process access, run a
# Qdrant server and point ``vector_store.config.url`` at it.

_LOCK_ERROR_MARKERS = ("already accessed by another instance", "already accessed by")

DEFAULT_LOCK_RETRIES = 5
DEFAULT_LOCK_BACKOFF = 0.25
_MAX_LOCK_BACKOFF = 2.0
# Window used when concurrency.lease_idle_release is switched on. Long enough to
# span a turn's storage calls (prefetch → tool calls → extraction write), short
# enough that another process's retry budget outlasts it.
DEFAULT_LEASE_IDLE_SECONDS = 2.0


def is_lock_conflict(exc: BaseException) -> bool:
    """True for QdrantLocal's "another instance owns this directory" error."""
    text = str(exc).lower()
    return any(marker in text for marker in _LOCK_ERROR_MARKERS)


def _lock_conflict_message(path: str, holder_hint: str = "") -> str:
    return (
        f"the local Qdrant store at {path} is held by another Qdrant client"
        f"{holder_hint}. Embedded Qdrant allows exactly one live owner per "
        "directory. Close the other Hermes process (CLI, gateway, Desktop or "
        "cron), or run a Qdrant server and set vector_store.config.url to it. "
        "Do not delete the .lock file — the lock lives on the open file handle, "
        "not the file, and removing it permits concurrent writers and "
        "corruption."
    )


def _open_local_client(path: str, *, retries: int, backoff: float):
    """Open an embedded Qdrant client, retrying while another owner releases.

    Deliberately keeps no reference to the failed attempt's exception. QdrantLocal
    opens ``<path>/.lock`` *before* trying to lock it and does not close that
    handle when the lock is refused, so holding the exception (and therefore its
    traceback, its frames, and the half-built client) keeps a file handle open
    per retry. On Windows that also blocks deleting or moving the directory.
    """
    from qdrant_client import QdrantClient

    delay = max(0.0, backoff)
    attempts = max(1, retries + 1)
    for attempt in range(attempts):
        try:
            return QdrantClient(path=path)
        except Exception as exc:
            if not is_lock_conflict(exc):
                raise
            del exc  # drop the traceback → the leaked lock handle is collected
            if attempt >= attempts - 1:
                break
            logger.debug(
                "mem0_hermes: local store busy (attempt %d/%d); retrying in %.2fs",
                attempt + 1, attempts, delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, _MAX_LOCK_BACKOFF)
    raise RuntimeError(
        _lock_conflict_message(path, " and did not release in time")
    ) from None


@contextmanager
def _retrying_local_client(retries: int, backoff: float):
    """Make Mem0's own ``QdrantClient(path=...)`` call retry on a lock conflict.

    Building ``Memory`` acquires the lock — Mem0 constructs its vector store in
    ``__init__`` — so two Hermes processes starting at the same moment collide
    before either has an operation to lease. Retrying at construction closes
    that window: the loser waits out the winner's first lease, which lasts
    milliseconds.

    Patched for the duration of construction only, and only for the local path
    form; a server URL goes through Mem0's untouched constructor.
    """
    import mem0.vector_stores.qdrant as qdrant_module

    original = qdrant_module.QdrantClient

    def factory(*args, **kwargs):
        path = kwargs.get("path")
        if path and not args:
            return _open_local_client(path, retries=retries, backoff=backoff)
        return original(*args, **kwargs)

    qdrant_module.QdrantClient = factory
    try:
        yield
    finally:
        qdrant_module.QdrantClient = original


class _StoreLease:
    """Owns the embedded-Qdrant directory for the duration of one storage call.

    Each of Mem0's vector-store methods is wrapped so it opens the store, runs,
    and closes again. Between calls the directory is unlocked and another Hermes
    process can use it — including during the extraction LLM call inside
    ``Memory.add``, which is the longest part of a turn and would otherwise hold
    the lock for seconds.

    Granularity is deliberately per storage call rather than per memory
    operation: that is what a real Qdrant server gives you (it serializes
    individual requests, not your whole read-think-write cycle). The trade is
    that two processes extracting at the same time can each work from a snapshot
    taken before the other's write, which can duplicate a fact. Blocking every
    other process's recall for the length of a model call is the worse failure.

    Between leases ``memory.vector_store.client`` holds a *closed* client, so any
    unwrapped access raises "QdrantLocal instance is closed" instead of silently
    reading a stale snapshot. Local persistence commits per point, so closing
    between calls cannot lose data.

    Re-entrant: Mem0's store methods call one another, and an inner lease must
    not close the client an outer one is still using. The lock also serializes
    this process's own storage calls, which QdrantLocal requires anyway.

    Optionally (``concurrency.lease_idle_release``, off by default) the store is
    kept open for a short idle window after the last call instead of closing
    immediately, so a whole turn's prefetch, tool calls and extraction write
    share one open. See :meth:`held` for what that trades away.
    """

    def __init__(
        self, memory: Any, path: str, *, retries: int, backoff: float,
        idle_seconds: float = 0.0,
    ):
        self._memory = memory
        self._path = path
        self._retries = retries
        self._backoff = backoff
        self._idle = max(0.0, idle_seconds)
        self._lock = threading.RLock()
        self._depth = 0
        self._open = False
        self._timer: Optional[threading.Timer] = None
        # Bumped every time the store is opened. An idle-release timer captures
        # the value it was scheduled under and does nothing if a later lease has
        # since reopened the store — otherwise a timer from a previous cycle
        # could close a client that is in use.
        self._generation = 0
        self.leases = 0  # observable in tests / debugging

    @property
    def path(self) -> str:
        return self._path

    @property
    def idle_seconds(self) -> float:
        return self._idle

    def _cancel_timer(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()

    def release_now(self) -> None:
        """Close the client so other processes can take the directory."""
        with self._lock:
            self._cancel_timer()
            self._open = False
            client = getattr(getattr(self._memory, "vector_store", None), "client", None)
            if client is None:
                return
            try:
                client.close()
            except Exception as exc:
                logger.debug("mem0_hermes: closing local store client: %s", exc)

    def _schedule_release(self) -> None:
        generation = self._generation

        def _fire() -> None:
            with self._lock:
                # Another call may have started, or reopened the store, between
                # the timer firing and it winning the lock.
                if self._depth == 0 and self._generation == generation and self._open:
                    self.release_now()

        self._cancel_timer()
        timer = threading.Timer(self._idle, _fire)
        # Daemon: a pending release must never hold up interpreter exit.
        timer.daemon = True
        self._timer = timer
        timer.start()

    @contextmanager
    def held(self):
        """Own the store for one storage call.

        With ``idle_seconds`` at 0 (the default) the store is closed on the way
        out, so it is unlocked between every call — maximum fairness to other
        Hermes processes, at the cost of one reopen per call, which grows with
        the number of stored points.

        With an idle window, the store stays open until the window passes
        without another call. Consecutive calls then cost nothing extra, but two
        things become true: another process may wait up to that window (plus its
        retry backoff) to get in, and the store stays open into the beginning of
        Mem0's extraction LLM call rather than being released before it.
        """
        with self._lock:
            if self._depth == 0:
                self._cancel_timer()
                if not self._open:
                    self._memory.vector_store.client = _open_local_client(
                        self._path, retries=self._retries, backoff=self._backoff
                    )
                    self._open = True
                    self._generation += 1
                    self.leases += 1
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
                if self._depth == 0:
                    if self._idle > 0:
                        self._schedule_release()
                    else:
                        self.release_now()

    def wrap(self, method):
        def leased(*args, **kwargs):
            with self.held():
                return method(*args, **kwargs)

        leased.__name__ = getattr(method, "__name__", "leased")
        leased.__doc__ = getattr(method, "__doc__", None)
        return leased


def _install_lease(
    memory: Any, path: str, *, retries: int, backoff: float, idle_seconds: float = 0.0,
) -> _StoreLease:
    """Wrap every public storage method of Mem0's vector store with a lease.

    Discovered from the class rather than a hardcoded name list, so a Mem0
    release that adds a storage method doesn't quietly leave it unleased —
    calling an unwrapped method between leases fails loudly on the closed
    client instead of corrupting anything.
    """
    lease = _StoreLease(
        memory, path, retries=retries, backoff=backoff, idle_seconds=idle_seconds,
    )
    store = memory.vector_store
    for name in dir(type(store)):
        if name.startswith("_"):
            continue
        attribute = getattr(type(store), name, None)
        if not callable(attribute) or isinstance(attribute, (property, type)):
            continue
        bound = getattr(store, name, None)
        if not callable(bound):
            continue
        setattr(store, name, lease.wrap(bound))
    # Constructing Memory took the lock; hand the directory back right away.
    lease.release_now()
    return lease


def _reset_collection_if_dims_changed(
    vector_store: Dict[str, Any], *, retries: int = DEFAULT_LOCK_RETRIES,
    backoff: float = DEFAULT_LOCK_BACKOFF,
) -> None:
    """Drop a local Qdrant collection whose vector width no longer matches.

    Switching embedders (e.g. OpenAI 1536 → Ollama 768) otherwise fails every
    upsert with a dimension mismatch. Same guard as the bundled mem0 plugin's
    OSS backend, narrowed to the local Qdrant default used here. Retries on a
    lock conflict: skipping the check because another process happened to hold
    the store would leave a stale-width collection in place.
    """
    if vector_store.get("provider") != "qdrant":
        return
    vs_config = vector_store.get("config") or {}
    expected = vs_config.get("embedding_model_dims")
    collection = vs_config.get("collection_name", "mem0")
    if not expected:
        return
    try:
        from qdrant_client import QdrantClient

        if vs_config.get("path"):
            client = _open_local_client(
                vs_config["path"], retries=retries, backoff=backoff
            )
        elif vs_config.get("url"):
            client = QdrantClient(url=vs_config["url"], api_key=vs_config.get("api_key"))
        else:
            return
        try:
            if not client.collection_exists(collection):
                return
            vectors = client.get_collection(collection).config.params.vectors
            if isinstance(vectors, dict):
                first = next(iter(vectors.values()), None)
                current = getattr(first, "size", None)
            else:
                current = getattr(vectors, "size", None)
            if current is not None and int(current) != int(expected):
                logger.warning(
                    "mem0_hermes: embedding dims changed (%s → %s); recreating "
                    "Qdrant collection %r",
                    current, expected, collection,
                )
                client.delete_collection(collection)
        finally:
            client.close()
    except Exception as exc:
        logger.debug("mem0_hermes: dims check skipped: %s", exc)


def _concurrency_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    block = config.get("concurrency")
    block = block if isinstance(block, dict) else {}
    lease = block.get("lease_local_store", True)
    if isinstance(lease, str):
        lease = lease.strip().lower() not in ("false", "0", "no")
    try:
        retries = int(block.get("lock_retries", DEFAULT_LOCK_RETRIES))
    except (TypeError, ValueError):
        retries = DEFAULT_LOCK_RETRIES
    try:
        backoff = float(block.get("lock_retry_backoff", DEFAULT_LOCK_BACKOFF))
    except (TypeError, ValueError):
        backoff = DEFAULT_LOCK_BACKOFF

    # Off by default: releasing on every call is the fair behavior, and the
    # idle window trades some of that fairness for latency on large stores.
    idle_enabled = block.get("lease_idle_release", False)
    if isinstance(idle_enabled, str):
        idle_enabled = idle_enabled.strip().lower() not in ("false", "0", "no", "")
    idle_seconds = 0.0
    if idle_enabled:
        try:
            idle_seconds = float(block.get("lease_idle_seconds", DEFAULT_LEASE_IDLE_SECONDS))
        except (TypeError, ValueError):
            idle_seconds = DEFAULT_LEASE_IDLE_SECONDS
        idle_seconds = max(0.0, idle_seconds)

    return {
        "lease": bool(lease),
        "retries": max(0, retries),
        "backoff": max(0.0, backoff),
        "idle_seconds": idle_seconds,
    }


def local_store_path(config: Dict[str, Any]) -> str:
    """Filesystem path of an embedded Qdrant store, or ``""`` for a server."""
    vector_store = config.get("vector_store") or {}
    if vector_store.get("provider") != "qdrant":
        return ""
    vs_config = vector_store.get("config") or {}
    if vs_config.get("url") or vs_config.get("host"):
        return ""  # real server: no lock, no leasing needed
    path = vs_config.get("path")
    return os.path.abspath(_expand(path)) if path else ""


def build_memory(config: Dict[str, Any]):
    """Return a Mem0 ``Memory`` wired to :class:`HermesRoutedLLM`.

    Raises on failure — the caller reports the message to the user.
    """
    _prepare_environment(config)
    _ensure_mem0_installed()
    _ensure_embedder_installed(config)
    # After the install, so fastembed's registry is available to check against.
    _reconcile_embedding_dims(config)

    from ._hermes_llm import HermesRoutedLLM, RoutedLlmConfig, register_with_mem0

    memory_config = build_memory_config(config)
    _settings = _concurrency_settings(config)
    _reset_collection_if_dims_changed(
        memory_config["vector_store"],
        retries=_settings["retries"], backoff=_settings["backoff"],
    )

    from mem0.configs.base import MemoryConfig
    from mem0.memory.main import Memory

    provider_name = register_with_mem0()
    routed_config = RoutedLlmConfig(**(memory_config["llm"]["config"] or {}))

    validated = MemoryConfig(**memory_config)
    # Only the reassignment is guarded. Letting a Memory() failure fall through
    # to the second construction would build the vector store twice — with a
    # local Qdrant path that fails on its own file lock, masking the real cause.
    try:
        validated.llm.provider = provider_name
        injected = validated.llm.provider == provider_name
    except Exception as exc:
        logger.debug("mem0_hermes: factory injection unavailable: %s", exc)
        injected = False

    store_path = local_store_path(config)
    settings = _concurrency_settings(config)
    try:
        with _retrying_local_client(settings["retries"], settings["backoff"]):
            if injected:
                memory = Memory(validated)
            else:
                logger.debug("mem0_hermes: falling back to post-construction LLM swap")
                fallback = copy.deepcopy(memory_config)
                # Placeholder key: only stops openai.OpenAI() from raising during
                # construction. The client is discarded by the swap below and
                # never sees a request.
                fallback["llm"]["config"] = dict(fallback["llm"]["config"] or {})
                fallback["llm"]["config"]["api_key"] = "unused-hermes-routed"
                memory = Memory(MemoryConfig(**fallback))
    except Exception as exc:
        # Construction takes the embedded-Qdrant lock. Translate the raw
        # "already accessed" RuntimeError into something that names the cause
        # and the two real fixes.
        if store_path and is_lock_conflict(exc):
            raise RuntimeError(_lock_conflict_message(store_path)) from exc
        raise

    routed = HermesRoutedLLM(routed_config)
    # Unconditional: guarantees the routed adapter is in place regardless of
    # which construction path ran above.
    memory.llm = routed
    if getattr(memory, "llm", None) is not routed:  # pragma: no cover - paranoia
        raise RuntimeError("mem0_hermes: failed to install the Hermes-routed LLM")
    logger.info(
        "mem0_hermes: memory extraction routed through %s", routed_config.describe(),
    )
    _prepare_history_db(memory, config)

    if store_path and settings["lease"] and getattr(memory.vector_store, "is_local", False):
        memory._hermes_lease = _install_lease(
            memory, store_path,
            retries=settings["retries"], backoff=settings["backoff"],
            idle_seconds=settings["idle_seconds"],
        )
        if settings["idle_seconds"] > 0:
            logger.info(
                "mem0_hermes: embedded Qdrant at %s is leased with a %.1fs idle "
                "release — consecutive calls share one open, and other Hermes "
                "processes may wait that long to get in",
                store_path, settings["idle_seconds"],
            )
        else:
            logger.info(
                "mem0_hermes: embedded Qdrant at %s is leased per storage call, so "
                "other Hermes processes can share it", store_path,
            )
    elif store_path:
        logger.info(
            "mem0_hermes: holding the embedded Qdrant lock at %s for this "
            "process's lifetime (concurrency.lease_local_store is off)", store_path,
        )
    return memory


DEFAULT_SQLITE_BUSY_TIMEOUT_MS = 15_000


def _prepare_history_db(memory: Any, config: Dict[str, Any]) -> str:
    """Make Mem0's history DB tolerate other Hermes processes.

    ``history.db`` is the second thing several Hermes processes open at once,
    and Mem0 connects with ``sqlite3.connect(path, check_same_thread=False)``
    and nothing else. Two defaults come with that:

    * **A 5 s busy timeout.** Mem0's writes are short (``BEGIN`` → one INSERT →
      ``COMMIT``), so contention is normally absorbed — measured here at 1200
      writes across 3 processes with zero failures, worst single wait 549 ms.
      But 5 s is a hard cliff: past it the write raises and takes the whole
      memory operation with it. Raising the ceiling costs nothing when
      uncontended.
    * **Rollback journalling**, where a writer blocks readers for the length of
      the write.

    Journal mode is delegated to ``hermes_state.apply_wal_with_fallback`` rather
    than setting ``PRAGMA journal_mode=WAL`` here, so this database follows the
    same policy Hermes applies to its own ``state.db``: WAL where it helps,
    DELETE on filesystems that reject WAL (NFS/SMB/FUSE), and — importantly on
    the SQLite shipped with this venv — *no* WAL on library builds carrying the
    WAL-reset corruption bug (3.7.0–3.51.2, fixed in 3.51.3 / 3.50.7 / 3.44.6).
    Enabling WAL ourselves would hand multi-process users a corruption risk
    Hermes deliberately refuses elsewhere.

    Returns the journal mode in force, or ``""`` when nothing could be applied.
    """
    connection = getattr(getattr(memory, "db", None), "connection", None)
    if connection is None:
        return ""

    settings = config.get("concurrency")
    settings = settings if isinstance(settings, dict) else {}
    try:
        timeout_ms = int(settings.get("sqlite_busy_timeout_ms", DEFAULT_SQLITE_BUSY_TIMEOUT_MS))
    except (TypeError, ValueError):
        timeout_ms = DEFAULT_SQLITE_BUSY_TIMEOUT_MS
    timeout_ms = max(0, timeout_ms)

    try:
        connection.execute(f"PRAGMA busy_timeout={timeout_ms}")
    except Exception as exc:
        logger.debug("mem0_hermes: could not set history busy_timeout: %s", exc)

    mode = ""
    if settings.get("sqlite_wal", True):
        try:
            from hermes_state import apply_wal_with_fallback  # type: ignore

            mode = apply_wal_with_fallback(connection, db_label="mem0_hermes history.db")
        except ImportError:
            logger.debug("mem0_hermes: hermes_state unavailable; leaving journal mode as-is")
        except Exception as exc:
            # Never fatal: the store works in any journal mode.
            logger.warning("mem0_hermes: history journal mode unchanged: %s", exc)
    if mode:
        logger.debug(
            "mem0_hermes: history.db journal=%s busy_timeout=%dms", mode, timeout_ms
        )
    return mode


def _unwrap_results(response: Any) -> List[dict]:
    if isinstance(response, dict):
        return response.get("results", []) or []
    if isinstance(response, list):
        return response
    return []


class HermesRoutedMem0Backend:
    """Thin operations wrapper around Mem0 OSS, mirroring the bundled plugin."""

    def __init__(self, config: Dict[str, Any], memory: Any = None):
        self._config = config
        self._memory = memory if memory is not None else build_memory(config)
        self._lease = getattr(self._memory, "_hermes_lease", None)
        # Shared-owner bookkeeping (see _acquire_shared).
        self._share_key = ""
        self._refcount = 0

    @property
    def memory(self) -> Any:
        return self._memory

    @property
    def leases(self) -> int:
        """Number of times the embedded store has been leased (diagnostics)."""
        return getattr(self._lease, "leases", 0)

    @contextmanager
    def _store(self):
        """Translate a lock conflict into something a user can act on.

        The lease itself is installed on Mem0's vector store, so the OS lock is
        taken and released per storage call inside this block.
        """
        try:
            yield
        except Exception as exc:
            if is_lock_conflict(exc):
                path = getattr(self._lease, "path", "") or local_store_path(self._config)
                raise RuntimeError(_lock_conflict_message(path)) from exc
            raise

    @property
    def routing(self) -> str:
        llm = getattr(self._memory, "llm", None)
        cfg = getattr(llm, "config", None)
        if cfg is not None and hasattr(cfg, "describe"):
            return cfg.describe()
        return "unknown"

    @property
    def last_model(self) -> str:
        return str(getattr(getattr(self._memory, "llm", None), "last_model", "") or "")

    def search(
        self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = False
    ) -> List[dict]:
        with self._store():
            return _unwrap_results(
                self._memory.search(query, filters=filters, top_k=top_k, rerank=rerank)
            )

    def add(
        self,
        messages: list,
        *,
        user_id: str,
        agent_id: str,
        infer: bool = False,
        metadata: Optional[dict] = None,
    ) -> dict:
        kwargs: Dict[str, Any] = {
            "user_id": user_id,
            "agent_id": agent_id,
            "infer": infer,
        }
        if metadata:
            kwargs["metadata"] = metadata
        with self._store():
            return self._memory.add(messages, **kwargs)

    def update(self, memory_id: str, text: str) -> dict:
        with self._store():
            self._memory.update(memory_id, text=text)
        return {"result": "Memory updated.", "memory_id": memory_id}

    def delete(self, memory_id: str) -> dict:
        with self._store():
            self._memory.delete(memory_id)
        return {"result": "Memory deleted.", "memory_id": memory_id}

    def close(self) -> None:
        memory = self._memory
        if memory is None:
            return
        # Before anything else: cancel a pending idle release and drop the OS
        # lock. Leaving a timer armed would reopen nothing but would fire against
        # a torn-down store.
        if self._lease is not None:
            try:
                self._lease.release_now()
            except Exception:
                pass
        try:
            telemetry = getattr(memory, "telemetry", None)
            posthog = getattr(telemetry, "posthog", None)
            if posthog is not None:
                posthog.shutdown()
        except Exception:
            pass
        for target in (memory, getattr(memory, "vector_store", None)):
            try:
                closer = getattr(target, "close", None)
                if callable(closer):
                    closer()
            except Exception:
                pass
        try:
            client = getattr(getattr(memory, "vector_store", None), "client", None)
            closer = getattr(client, "close", None)
            if callable(closer):
                closer()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# One owner per storage path, per process
# ---------------------------------------------------------------------------
#
# Hermes can build more than one memory provider in a single process (a gateway
# serving several sessions, a cached agent, a re-``initialize()`` after a
# provider error). Each would construct its own Memory and the second would hit
# QdrantLocal's lock — the plugin fighting itself. Sharing one backend per
# storage path removes that case entirely; recall and writes stay correctly
# scoped because ``user_id``/``agent_id`` are passed per call, not held on the
# backend.

_OWNERS: Dict[str, "HermesRoutedMem0Backend"] = {}
_OWNERS_LOCK = threading.Lock()


def _share_key(config: Dict[str, Any]) -> str:
    """Identity of the underlying store, or ``""`` when sharing doesn't apply."""
    path = local_store_path(config)
    if not path:
        return ""  # a server handles concurrency itself; no need to share
    collection = ((config.get("vector_store") or {}).get("config") or {}).get(
        "collection_name", "mem0"
    )
    return f"{os.path.normcase(path)}::{collection}"


def acquire_backend(config: Dict[str, Any]) -> "HermesRoutedMem0Backend":
    """Return a backend for ``config``, reusing the process's existing owner."""
    key = _share_key(config)
    if not key:
        return HermesRoutedMem0Backend(config)
    with _OWNERS_LOCK:
        backend = _OWNERS.get(key)
        if backend is None:
            backend = HermesRoutedMem0Backend(config)
            backend._share_key = key
            _OWNERS[key] = backend
        else:
            logger.debug("mem0_hermes: reusing this process's owner of %s", key)
        backend._refcount += 1
        return backend


def release_backend(backend: Optional["HermesRoutedMem0Backend"]) -> None:
    """Drop a reference; close the store once nothing in the process holds it."""
    if backend is None:
        return
    key = getattr(backend, "_share_key", "")
    if not key:
        backend.close()
        return
    with _OWNERS_LOCK:
        backend._refcount -= 1
        if backend._refcount > 0:
            return
        if _OWNERS.get(key) is backend:
            del _OWNERS[key]
    backend.close()
