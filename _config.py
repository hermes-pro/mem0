# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Configuration for the ``mem0_hermes`` memory provider.

Canonical file: ``$HERMES_HOME/mem0_hermes.json``.

Shape (every key optional — the defaults below are used for anything absent)::

    {
      "user_id": "hermes-user",
      "agent_id": "hermes",
      "rerank": false,
      "telemetry": false,
      "custom_instructions": null,
      "llm": {
        "task": "mem0_hermes_extraction",
        "provider": "",          # "" = whatever `hermes model` is set to
        "model": "",             # "" = the main chat model
        "base_url": "",
        "api_key": "",
        "temperature": 0.1,
        "max_tokens": null,      # null = provider default
        "timeout": 120,
        "json_mode": "prompt"    # prompt | response_format | off
      },
      "embedder": {"provider": "openai", "config": {...}},
      "vector_store": {"provider": "qdrant", "config": {...}},
      "history_db_path": "<hermes_home>/mem0_hermes/history.db"
    }

Only the ``llm`` block is Hermes-specific: it configures the adapter in
:mod:`._hermes_llm`, which routes Mem0's extraction/update calls through
``agent.auxiliary_client.call_llm`` instead of Mem0's own OpenAI client. The
``embedder`` and ``vector_store`` blocks are passed through to Mem0 verbatim
(Hermes has no embedding path of its own, so embeddings still go to whichever
provider you configure there).
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Sentinel user_id. Treated as "operator did not configure one", so the
# gateway-native id (Telegram numeric id, Discord snowflake, …) still wins.
# Same convention as the bundled mem0 plugin.
DEFAULT_USER_ID = "hermes-user"
DEFAULT_AGENT_ID = "hermes"

CONFIG_FILENAME = "mem0_hermes.json"
STATE_DIRNAME = "mem0_hermes"

# Auxiliary-task key. Unconfigured, auxiliary_client resolves it to the main
# provider + main model (see _resolve_auto in agent/auxiliary_client.py), which
# is the whole point of this plugin. Users who want extraction on a cheaper
# model can pin `auxiliary.mem0_hermes_extraction.{provider,model}` in
# config.yaml, or set llm.provider / llm.model here.
DEFAULT_AUX_TASK = "mem0_hermes_extraction"

# Embedder providers Mem0 accepts, with the ones that need a key flagged.
# Used for the setup wizard and for is_available().
EMBEDDER_KEY_ENV: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "together": "TOGETHER_API_KEY",
}
EMBEDDER_CHOICES = [
    "fastembed",   # default — local ONNX, no server, no key, no data leaves the box
    "ollama",      # nomic-embed-text — needs a local Ollama server
    "openai",      # text-embedding-3-small — needs OPENAI_API_KEY
    "huggingface",
    "azure_openai",
    "gemini",
    "lmstudio",
    "together",
    "aws_bedrock",
]

# fastembed's own default model: 384 dims, 67 MB of ONNX weights on first use.
# Mem0's FastEmbedEmbedding would otherwise default to thenlper/gte-large at
# 1.2 GB — a lot to download for a memory sidecar. Every width below is checked
# against fastembed's registry at setup time (resolve_fastembed_dims), so a
# wrong entry here can't silently create a mis-sized vector collection.
DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Set by the test suite (and usable in CI) to refuse runtime pip installs, so a
# test run can never mutate the interpreter it happens to be running under.
NO_INSTALL_ENV = "MEM0_HERMES_NO_INSTALL"

EMBEDDER_DEFAULT_MODEL: Dict[str, str] = {
    "fastembed": DEFAULT_FASTEMBED_MODEL,
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
}

# Python packages each embedder needs, installed only for the provider actually
# selected (see ensure_embedder_dependencies). ``fastembed>=0.3.1`` matches the
# constraint in mem0ai's own ``extras`` extra, so this can't fight Mem0's
# resolver — and it doubles as enabling Mem0's BM25 keyword search, which is
# skipped when fastembed is absent.
EMBEDDER_PIP_DEPS: Dict[str, tuple] = {
    "fastembed": ("fastembed>=0.3.1",),
    "ollama": ("ollama",),
    "huggingface": ("sentence-transformers",),
}
# pip name → import name, where they differ.
_PIP_IMPORT_NAMES = {"sentence-transformers": "sentence_transformers"}

# Embedding dimensions for models where Mem0 does not report them up front.
# Needed so the Qdrant collection is created with the right vector size.
KNOWN_DIMS: Dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "thenlper/gte-large": 1024,
    "intfloat/multilingual-e5-large": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
}
EMBEDDER_BASE_URL_KEY: Dict[str, str] = {
    "openai": "openai_base_url",
    "ollama": "ollama_base_url",
    "lmstudio": "lmstudio_base_url",
}

VECTOR_CHOICES = ["qdrant", "pgvector", "chroma", "faiss"]

JSON_MODE_CHOICES = ["prompt", "response_format", "off"]


def hermes_home() -> Path:
    """Return the active HERMES_HOME, without requiring the agent to be running."""
    try:
        from hermes_constants import get_hermes_home  # type: ignore

        return Path(get_hermes_home())
    except Exception:
        env = os.environ.get("HERMES_HOME", "").strip()
        return Path(env) if env else Path.home() / ".hermes"


def default_config(home: Optional[Path] = None) -> Dict[str, Any]:
    """Return the built-in defaults, resolved against ``home``."""
    root = Path(home) if home else hermes_home()
    state = root / STATE_DIRNAME
    return {
        "agent_id": DEFAULT_AGENT_ID,
        "rerank": False,
        # Mem0's PostHog telemetry is off by default here: memory content is
        # private and the extra telemetry vector store it spins up costs a
        # second collection on disk for nothing.
        "telemetry": False,
        "custom_instructions": None,
        "llm": {
            "task": DEFAULT_AUX_TASK,
            "provider": "",
            "model": "",
            "base_url": "",
            "api_key": "",
            "temperature": 0.1,
            "max_tokens": None,
            "timeout": 120,
            "json_mode": "prompt",
        },
        # Local by default: embeddings are the one part of this plugin that
        # can't ride Hermes's auth, so the default must not require a key —
        # otherwise a Codex/OAuth user (who has no OPENAI_API_KEY at all) is
        # stuck at the first step. fastembed runs ONNX in-process.
        "embedder": {
            "provider": "fastembed",
            "config": {
                "model": DEFAULT_FASTEMBED_MODEL,
                "embedding_dims": KNOWN_DIMS[DEFAULT_FASTEMBED_MODEL],
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {"path": str(state / "qdrant"), "collection_name": "mem0_hermes"},
        },
        "history_db_path": str(state / "history.db"),
    }


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` (returns a new dict)."""
    out = copy.deepcopy(base)
    for key, value in (overlay or {}).items():
        if value is None and key in ("custom_instructions", "max_tokens"):
            out[key] = None  # explicit null is meaningful for these
            continue
        if value is None or value == "":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _inherited_from_bundled_mem0(home: Path) -> Dict[str, Any]:
    """Reuse an existing ``mem0.json`` OSS setup for embedder + vector store.

    Someone who already ran ``hermes memory setup`` for the bundled ``mem0``
    plugin in OSS mode has a working embedder and vector store; inheriting them
    means switching to this plugin keeps reading the same memories instead of
    starting from an empty collection. The bundled ``oss.llm`` block is
    deliberately ignored — replacing it is the point of this plugin.
    """
    bundled = _read_json(home / "mem0.json")
    inherited: Dict[str, Any] = {}
    oss = bundled.get("oss") if isinstance(bundled.get("oss"), dict) else {}
    for key in ("embedder", "vector_store"):
        block = oss.get(key)
        if isinstance(block, dict) and block.get("provider"):
            inherited[key] = copy.deepcopy(block)
    for key in ("user_id", "agent_id", "rerank"):
        if bundled.get(key) not in (None, ""):
            inherited[key] = bundled[key]
    return inherited


def _env_overrides() -> Dict[str, Any]:
    """Env-var escape hatches (highest precedence)."""
    out: Dict[str, Any] = {}
    llm: Dict[str, Any] = {}

    # MEM0_* names are honored too so an existing bundled-mem0 environment
    # keeps working after switching providers.
    for env_name in ("MEM0_HERMES_USER_ID", "MEM0_USER_ID"):
        value = os.environ.get(env_name, "").strip()
        if value:
            out["user_id"] = value
            break
    for env_name in ("MEM0_HERMES_AGENT_ID", "MEM0_AGENT_ID"):
        value = os.environ.get(env_name, "").strip()
        if value:
            out["agent_id"] = value
            break

    for env_name, key in (
        ("MEM0_HERMES_LLM_PROVIDER", "provider"),
        ("MEM0_HERMES_LLM_MODEL", "model"),
        ("MEM0_HERMES_LLM_BASE_URL", "base_url"),
        ("MEM0_HERMES_LLM_TASK", "task"),
        ("MEM0_HERMES_JSON_MODE", "json_mode"),
    ):
        value = os.environ.get(env_name, "").strip()
        if value:
            llm[key] = value
    if llm:
        out["llm"] = llm
    return out


def _drop_inherited_dims(base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
    """Forget an inherited ``embedding_dims`` when the overlay picks a new model.

    Dimensions belong to a specific embedding model, so a layer that changes the
    embedder without restating them must not keep the lower layer's number — a
    1536-wide default silently applied to a 768-wide model makes every upsert
    fail on vector width.
    """
    embedder = overlay.get("embedder")
    if not isinstance(embedder, dict):
        return
    overlay_config = embedder.get("config")
    overlay_config = overlay_config if isinstance(overlay_config, dict) else {}
    if not (embedder.get("provider") or overlay_config.get("model")):
        return
    if overlay_config.get("embedding_dims"):
        return  # stated explicitly alongside the model — respect it
    base_config = (base.get("embedder") or {}).get("config")
    if isinstance(base_config, dict):
        base_config.pop("embedding_dims", None)


def load_config(home: Optional[Path] = None) -> Dict[str, Any]:
    """Resolve the effective config.

    Precedence (lowest → highest): built-in defaults → an existing bundled
    ``mem0.json`` OSS setup → ``mem0_hermes.json`` → environment variables.
    """
    root = Path(home) if home else hermes_home()
    config = default_config(root)
    for overlay in (
        _inherited_from_bundled_mem0(root),
        _read_json(root / CONFIG_FILENAME),
        _env_overrides(),
    ):
        _drop_inherited_dims(config, overlay)
        config = _deep_merge(config, overlay)
    _backfill_embedding_dims(config)
    return config


def _backfill_embedding_dims(config: Dict[str, Any]) -> None:
    """Fill in ``embedding_dims`` for models we know, when not set explicitly."""
    embedder = config.get("embedder") or {}
    block = embedder.get("config")
    if not isinstance(block, dict):
        return
    if block.get("embedding_dims"):
        return
    dims = KNOWN_DIMS.get(str(block.get("model") or ""))
    if dims:
        block["embedding_dims"] = dims


# ---------------------------------------------------------------------------
# Embedder dependencies
# ---------------------------------------------------------------------------

def embedder_provider(config: Dict[str, Any]) -> str:
    return str((config.get("embedder") or {}).get("provider") or "")


def _importable(pip_spec: str) -> bool:
    """True if the package behind ``pip_spec`` can already be imported."""
    import importlib.util
    import re

    match = re.match(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*", pip_spec)
    base = match.group(0) if match else pip_spec
    module = _PIP_IMPORT_NAMES.get(base, base.replace("-", "_"))
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        # A partially-installed package can raise here rather than return None;
        # treat that as "not usable" so the installer gets a chance to repair it.
        return False


def embedder_pip_requirements(config: Dict[str, Any]) -> tuple:
    """pip specs the selected embedder needs and that aren't importable yet."""
    specs = EMBEDDER_PIP_DEPS.get(embedder_provider(config), ())
    return tuple(spec for spec in specs if not _importable(spec))


def ensure_embedder_dependencies(config: Dict[str, Any]) -> tuple:
    """Install the selected embedder's packages. Returns ``(ok, message)``.

    Declaring these in ``plugin.yaml`` isn't an option: ``hermes memory setup``
    installs manifest dependencies *before* walking the config schema, so it
    would download every embedder's stack — including fastembed's ONNX runtime —
    for users who picked OpenAI. Installing per selection keeps the cost on the
    provider actually chosen.

    Routed through ``tools.lazy_deps.install_specs``, which is what the memory
    setup wizard itself uses: venv-scoped, redirected to the durable target on
    sealed images, and gated by ``security.allow_lazy_installs``.
    """
    missing = embedder_pip_requirements(config)
    if not missing:
        return True, ""

    provider = embedder_provider(config)
    if os.environ.get(NO_INSTALL_ENV, "").strip():
        return False, (
            f"embedder '{provider}' needs {', '.join(missing)}, but installs are "
            f"disabled by {NO_INSTALL_ENV}"
        )
    try:
        from tools.lazy_deps import install_specs  # type: ignore
    except ImportError:
        return False, (
            f"embedder '{provider}' needs {', '.join(missing)} — install it with "
            f"pip, or run `hermes memory setup` from a Hermes environment"
        )

    result = install_specs(list(missing))
    if getattr(result, "ok", False):
        return True, f"installed {', '.join(missing)}"
    reason = getattr(result, "reason", "") or getattr(result, "stderr", "") or "install failed"
    return False, (
        f"could not install {', '.join(missing)} for embedder '{provider}': {reason}"
    )


def resolve_fastembed_dims(model: str) -> Optional[int]:
    """Look up a fastembed model's true vector width, or ``None`` if unknown.

    Authoritative when fastembed is installed — it reads fastembed's own model
    registry, so the vector store can never be created at a guessed width. Falls
    back to :data:`KNOWN_DIMS` when fastembed isn't importable yet.
    """
    try:
        from fastembed import TextEmbedding  # type: ignore

        for entry in TextEmbedding.list_supported_models():
            name = entry.get("model") if isinstance(entry, dict) else getattr(entry, "model", "")
            if str(name) == model:
                dims = entry.get("dim") if isinstance(entry, dict) else getattr(entry, "dim", None)
                return int(dims) if dims else None
        return None  # fastembed is present and does not know this model
    except Exception:
        return KNOWN_DIMS.get(model)


def fastembed_supported_models(limit: int = 8) -> list:
    """A few model names fastembed accepts, for error messages."""
    try:
        from fastembed import TextEmbedding  # type: ignore

        names = []
        for entry in TextEmbedding.list_supported_models():
            name = entry.get("model") if isinstance(entry, dict) else getattr(entry, "model", "")
            if name:
                names.append(str(name))
        return sorted(names)[:limit]
    except Exception:
        return []


def sync_fastembed_dims(config: Dict[str, Any]) -> tuple:
    """Reconcile ``embedding_dims`` with fastembed's registry. ``(changed, note)``.

    Only meaningful once fastembed is installed. If the configured model isn't in
    the registry the config is left alone and the caller gets a note to show —
    the mistake surfaces as a message naming valid models rather than as a
    dimension mismatch on the first write.
    """
    if embedder_provider(config) != "fastembed":
        return False, ""
    block = (config.get("embedder") or {}).get("config")
    if not isinstance(block, dict):
        return False, ""
    model = str(block.get("model") or DEFAULT_FASTEMBED_MODEL)
    block.setdefault("model", model)

    actual = resolve_fastembed_dims(model)
    if actual is None:
        available = fastembed_supported_models()
        if not available:
            return False, ""  # fastembed not importable; nothing verified
        return False, (
            f"fastembed does not offer model '{model}'. Supported models include: "
            + ", ".join(available)
        )
    if int(block.get("embedding_dims") or 0) == actual:
        return False, ""
    previous = block.get("embedding_dims")
    block["embedding_dims"] = actual
    if previous:
        return True, f"corrected embedding_dims for {model}: {previous} → {actual}"
    return True, f"embedding_dims for {model}: {actual}"


def resolved_user_id(config: Dict[str, Any], gateway_user_id: str = "") -> str:
    """Pick the user_id: operator-configured → gateway-native → default.

    An explicitly configured ``DEFAULT_USER_ID`` is treated as unset so that
    someone who accepted the wizard's suggested default still gets
    gateway-native ids rather than every platform bucketed into one principal.
    """
    configured = str(config.get("user_id") or "").strip()
    if configured == DEFAULT_USER_ID:
        configured = ""
    return configured or (gateway_user_id or "").strip() or DEFAULT_USER_ID


# ---------------------------------------------------------------------------
# Setup wizard (`hermes memory setup`) support
# ---------------------------------------------------------------------------

# Flat wizard key → dotted path in mem0_hermes.json.
_WIZARD_KEY_MAP: Dict[str, str] = {
    "user_id": "user_id",
    "agent_id": "agent_id",
    "llm_provider": "llm.provider",
    "llm_model": "llm.model",
    "llm_base_url": "llm.base_url",
    "temperature": "llm.temperature",
    "max_tokens": "llm.max_tokens",
    "json_mode": "llm.json_mode",
    "embedder_provider": "embedder.provider",
    "embedder_model": "embedder.config.model",
    "embedder_url": "embedder.config.__base_url__",
    "vector_provider": "vector_store.provider",
    "vector_path": "vector_store.config.path",
    "vector_url": "vector_store.config.url",
}

_NUMERIC_KEYS = {"temperature": float, "max_tokens": int}


def config_schema(config: Optional[Dict[str, Any]] = None) -> list:
    """Field descriptors for ``hermes memory setup``."""
    cfg = config or load_config()
    embedder = (cfg.get("embedder") or {}).get("provider") or "fastembed"
    return [
        {
            "key": "user_id",
            "description": "User identifier (shared across gateways)",
            "default": cfg.get("user_id") or DEFAULT_USER_ID,
        },
        {
            "key": "agent_id",
            "description": "Agent identifier",
            "default": cfg.get("agent_id") or DEFAULT_AGENT_ID,
        },
        {
            "key": "llm_model",
            "description": (
                "Model for memory extraction (blank = your main Hermes model)"
            ),
            "default": (cfg.get("llm") or {}).get("model", ""),
        },
        {
            "key": "json_mode",
            "description": "How to ask the model for JSON",
            "choices": JSON_MODE_CHOICES,
            "default": (cfg.get("llm") or {}).get("json_mode", "prompt"),
        },
        {
            "key": "embedder_provider",
            "description": (
                "Embedding provider — installed on selection; embeddings do NOT "
                "go through Hermes"
            ),
            "choices": EMBEDDER_CHOICES,
            "default": embedder,
        },
        {
            "key": "embedder_model",
            "description": "Embedding model",
            "default_from": {"field": "embedder_provider", "map": EMBEDDER_DEFAULT_MODEL},
            "default": EMBEDDER_DEFAULT_MODEL.get(embedder, ""),
        },
        {
            "key": "embedder_url",
            "description": "Embedding server URL",
            "when": {"embedder_provider": "ollama"},
            "default": "http://localhost:11434",
        },
        {
            "key": "openai_api_key",
            "description": "OpenAI API key (embeddings only)",
            "secret": True,
            "env_var": "OPENAI_API_KEY",
            "when": {"embedder_provider": "openai"},
            "url": "https://platform.openai.com/api-keys",
        },
        {
            "key": "vector_path",
            "description": "Local vector store path",
            "default": ((cfg.get("vector_store") or {}).get("config") or {}).get("path", ""),
        },
    ]


def _assign(target: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = target
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def wizard_values_to_config(
    values: Dict[str, Any], existing: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Translate the wizard's flat answers into the nested config shape."""
    out: Dict[str, Any] = copy.deepcopy(existing or {})
    embedder_provider = str(
        values.get("embedder_provider")
        or ((out.get("embedder") or {}).get("provider"))
        or "openai"
    )

    for flat_key, dotted in _WIZARD_KEY_MAP.items():
        if flat_key not in values:
            continue
        value = values[flat_key]
        if value in (None, ""):
            continue
        if flat_key in _NUMERIC_KEYS:
            try:
                value = _NUMERIC_KEYS[flat_key](value)
            except (TypeError, ValueError):
                continue
        if dotted.endswith("__base_url__"):
            base_key = EMBEDDER_BASE_URL_KEY.get(embedder_provider)
            if not base_key:
                continue
            dotted = f"embedder.config.{base_key}"
        _assign(out, dotted, value)

    # Changing the embedder usually changes the vector width; drop a stale
    # explicit embedding_dims so it is recomputed from the new model.
    if "embedder_provider" in values or "embedder_model" in values:
        block = (out.get("embedder") or {}).get("config")
        if isinstance(block, dict):
            model = str(block.get("model") or "")
            dims = KNOWN_DIMS.get(model)
            if dims:
                block["embedding_dims"] = dims
            else:
                block.pop("embedding_dims", None)
    return out


def _write_config(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from utils import atomic_json_write  # type: ignore

        atomic_json_write(path, data, mode=0o600)
    except Exception:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass


def save_config(
    values: Dict[str, Any], home: Path, *, install: bool = True, log=print
) -> Path:
    """Merge wizard answers into ``$HERMES_HOME/mem0_hermes.json``.

    Then install the packages the chosen embedder needs, because this is the
    first point in ``hermes memory setup`` where that choice is known — manifest
    dependencies are installed before the schema walk. The config is written
    *before* installing, so a failed or blocked install still leaves the
    selection saved and re-runnable.

    ``install=False`` skips it (used by tests and by callers that only want the
    file written). ``log`` receives one-line progress messages; the wizard's own
    dependency step prints the same way.
    """
    root = Path(home)
    path = root / CONFIG_FILENAME
    merged = wizard_values_to_config(values, _read_json(path))
    _write_config(path, merged)
    if not install:
        return path

    # ``merged`` holds only what has been explicitly configured, so consult the
    # fully-resolved config to learn which embedder is actually in force.
    effective = load_config(root)
    missing = embedder_pip_requirements(effective)
    if missing:
        log(f"  Installing {embedder_provider(effective)} embedder: {', '.join(missing)}")
        ok, message = ensure_embedder_dependencies(effective)
        if message:
            log(f"  {'✓' if ok else '⚠'} {message}")

    # With fastembed now importable, take its registry's word for the vector
    # width instead of the table in this module.
    changed, note = sync_fastembed_dims(effective)
    if note:
        log(f"  {'✓' if changed else '⚠'} {note}")
    if changed:
        block = (effective.get("embedder") or {}).get("config") or {}
        _assign(merged, "embedder.provider", embedder_provider(effective))
        _assign(merged, "embedder.config.model", block.get("model"))
        _assign(merged, "embedder.config.embedding_dims", block.get("embedding_dims"))
        _write_config(path, merged)
    return path


def embedder_key_missing(config: Dict[str, Any]) -> str:
    """Return the name of the missing embedder API-key env var, or ``""``.

    The routed LLM needs no credentials of its own (it borrows the main
    Hermes provider's), and the default embedder (fastembed) is local — so this
    is empty unless someone switches to a hosted embedder.
    """
    embedder = embedder_provider(config)
    inline = ((config.get("embedder") or {}).get("config") or {}).get("api_key")
    if inline:
        return ""
    env_var = EMBEDDER_KEY_ENV.get(str(embedder))
    if env_var and not os.environ.get(env_var, "").strip():
        return env_var
    return ""


def external_state_paths(
    config: Dict[str, Any], home: Optional[Path] = None
) -> list:
    """Paths this provider writes that ``hermes backup`` would otherwise miss.

    ``backup_paths()`` is for state kept *outside* HERMES_HOME — the backup
    command already walks HERMES_HOME itself, and anything returned here is
    copied into a separate ``_external/`` subtree of the archive. The defaults
    live under HERMES_HOME, so normally this is empty; it only reports the
    vector store / history DB after someone points them elsewhere.
    """
    root = Path(home) if home else hermes_home()
    try:
        root_resolved = root.resolve()
    except OSError:
        root_resolved = root

    candidates = []
    history = config.get("history_db_path")
    if history:
        candidates.append(os.path.expanduser(str(history)))
    vs_config = (config.get("vector_store") or {}).get("config") or {}
    if vs_config.get("path"):
        candidates.append(os.path.expanduser(str(vs_config["path"])))

    external = []
    for candidate in candidates:
        path = Path(candidate)
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if root_resolved in resolved.parents or resolved == root_resolved:
            continue  # already inside HERMES_HOME; the backup walk covers it
        external.append(str(resolved))
    return external
