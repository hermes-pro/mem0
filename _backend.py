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


def _reset_collection_if_dims_changed(vector_store: Dict[str, Any]) -> None:
    """Drop a local Qdrant collection whose vector width no longer matches.

    Switching embedders (e.g. OpenAI 1536 → Ollama 768) otherwise fails every
    upsert with a dimension mismatch. Same guard as the bundled mem0 plugin's
    OSS backend, narrowed to the local Qdrant default used here.
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
            client = QdrantClient(path=vs_config["path"])
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
    _reset_collection_if_dims_changed(memory_config["vector_store"])

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

    if injected:
        memory = Memory(validated)
    else:
        logger.debug("mem0_hermes: falling back to post-construction LLM swap")
        fallback = copy.deepcopy(memory_config)
        # Placeholder key: only stops openai.OpenAI() from raising during
        # construction. The client is discarded by the swap below and never
        # sees a request.
        fallback["llm"]["config"] = dict(fallback["llm"]["config"] or {})
        fallback["llm"]["config"]["api_key"] = "unused-hermes-routed"
        memory = Memory(MemoryConfig(**fallback))

    routed = HermesRoutedLLM(routed_config)
    # Unconditional: guarantees the routed adapter is in place regardless of
    # which construction path ran above.
    memory.llm = routed
    if getattr(memory, "llm", None) is not routed:  # pragma: no cover - paranoia
        raise RuntimeError("mem0_hermes: failed to install the Hermes-routed LLM")
    logger.info(
        "mem0_hermes: memory extraction routed through %s", routed_config.describe(),
    )
    return memory


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

    @property
    def memory(self) -> Any:
        return self._memory

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
        return self._memory.add(messages, **kwargs)

    def update(self, memory_id: str, text: str) -> dict:
        self._memory.update(memory_id, text=text)
        return {"result": "Memory updated.", "memory_id": memory_id}

    def delete(self, memory_id: str) -> dict:
        self._memory.delete(memory_id)
        return {"result": "Memory deleted.", "memory_id": memory_id}

    def close(self) -> None:
        memory = self._memory
        if memory is None:
            return
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
