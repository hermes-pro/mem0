# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Mem0 LLM adapter that routes generation through Hermes's own model path.

Mem0's OSS ``Memory`` calls ``self.llm.generate_response(...)`` for every piece
of LLM work it does — fact extraction on ``add(infer=True)``, the add/update/
delete decision pass, and procedural summaries. Out of the box that object is
``mem0.llms.openai.OpenAILLM``, which owns its own ``openai.OpenAI`` client and
talks straight to ``api.openai.com`` with ``OPENAI_API_KEY``.

:class:`HermesRoutedLLM` is a drop-in replacement that implements the same
duck-typed contract but sends the request through
``agent.auxiliary_client.call_llm`` — the same centralized path Hermes uses for
compression, vision, titles and every other side task. That means memory
extraction inherits, for free:

* the provider/model configured by ``hermes model`` (Anthropic, OpenRouter,
  Bedrock, Codex, a local endpoint, …) — no OpenAI key required anywhere;
* Hermes's auth handling: credential pools, OAuth refresh, Anthropic/Bedrock/
  Codex request shaping, header attribution;
* its retry, transient-blip and provider-fallback chain;
* its usage/cost accounting and ``Auxiliary <task>: using <provider> (<model>)``
  logging, so routed memory calls are visible like any other aux call.

The class deliberately does **not** subclass ``mem0.llms.base.LLMBase``: this
module is imported by Hermes's plugin loader before ``mem0ai`` is guaranteed to
be installed, and a module-level ``mem0`` import would leave a half-initialized
module in ``sys.modules``. Mem0 only ever duck-types this object, so a plain
class with ``.config`` and ``.generate_response()`` is sufficient.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Provider id registered into mem0's LlmFactory, and a stable sys.modules alias
# so the factory's dotted-path import resolves no matter which synthetic
# package name the plugin loader gave this module.
PROVIDER_NAME = "hermes_routed"
_MODULE_ALIAS = "_hermes_mem0_routed_llm"
_CLASS_PATH = f"{_MODULE_ALIAS}.HermesRoutedLLM"

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class RoutedLlmConfig:
    """Stand-in for ``mem0.configs.llms.base.BaseLlmConfig``.

    Mem0 instantiates this with the ``llm.config`` dict from the memory config,
    so unknown keys must be tolerated rather than raising ``TypeError``.
    """

    def __init__(
        self,
        *,
        task: str = "mem0_hermes_extraction",
        provider: str = "",
        model: str = "",
        base_url: str = "",
        api_key: str = "",
        temperature: Optional[float] = 0.1,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = 120,
        json_mode: str = "prompt",
        vision_details: str = "auto",
        enable_vision: bool = False,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **extra: Any,
    ) -> None:
        self.task = str(task or "mem0_hermes_extraction")
        # "" means "let auxiliary_client resolve it", which lands on the main
        # Hermes provider/model. Never fall back to an OpenAI default here.
        self.provider = str(provider or "").strip()
        self.model = str(model or "").strip()
        self.base_url = str(base_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.json_mode = str(json_mode or "prompt").strip().lower()
        self.vision_details = vision_details
        self.enable_vision = enable_vision
        self.top_p = top_p
        self.top_k = top_k
        self.extra = dict(extra)

    def describe(self) -> str:
        """Human-readable routing summary for status output."""
        if self.provider or self.model:
            return f"{self.provider or 'auto'}/{self.model or 'default'}"
        return f"main Hermes model (auxiliary.{self.task})"


def _message_text(content: Any) -> str:
    """Flatten a message/response content value into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _extract_text(response: Any) -> str:
    """Pull assistant text out of an OpenAI-shaped response object."""
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return ""
    text = _message_text(getattr(message, "content", None))
    if text:
        return text
    # Some reasoning endpoints put the payload in reasoning_content when the
    # visible content came back empty.
    return _message_text(getattr(message, "reasoning_content", None))


def _extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
    """Normalize tool calls into Mem0's ``{"name", "arguments"}`` shape."""
    out: List[Dict[str, Any]] = []
    try:
        raw_calls = response.choices[0].message.tool_calls or []
    except (AttributeError, IndexError, TypeError):
        return out
    for call in raw_calls:
        function = getattr(call, "function", None)
        if function is None and isinstance(call, dict):
            function = call.get("function")
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
        if name is None and isinstance(function, dict):
            name = function.get("name")
            arguments = function.get("arguments")
        if not name:
            continue
        parsed: Any = {}
        if isinstance(arguments, str) and arguments.strip():
            try:
                parsed = json.loads(coerce_json(arguments), strict=False)
            except Exception:
                parsed = {"_raw": arguments}
        elif isinstance(arguments, dict):
            parsed = arguments
        out.append({"name": name, "arguments": parsed})
    return out


def coerce_json(text: str) -> str:
    """Reduce model output to the JSON object it contains.

    Mem0's prompts already demand bare JSON, but models reached through
    arbitrary providers wrap it in prose, code fences or ``<think>`` blocks.
    Mem0 tolerates fences on the extraction path only; doing it here makes
    every call site safe.
    """
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text).strip()
    fenced = _FENCE_RE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        return cleaned[start : end + 1]
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _wants_json(response_format: Any) -> bool:
    if isinstance(response_format, dict):
        return str(response_format.get("type", "")).startswith("json")
    return bool(response_format)


class HermesRoutedLLM:
    """Mem0-compatible LLM that runs on Hermes's configured model."""

    def __init__(self, config: Any = None):
        if config is None:
            self.config = RoutedLlmConfig()
        elif isinstance(config, dict):
            self.config = RoutedLlmConfig(**config)
        else:
            self.config = config
        # Populated after the first successful call — surfaced in provider
        # status output so users can confirm where extraction actually ran.
        self.last_model: str = ""
        self.call_count: int = 0

    # -- Mem0 contract ------------------------------------------------------

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        response_format: Any = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs: Any,
    ) -> Any:
        """Run one completion through ``agent.auxiliary_client.call_llm``.

        Returns the assistant text, or — when ``tools`` are supplied — Mem0's
        ``{"content": ..., "tool_calls": [...]}`` dict.
        """
        if not messages:
            raise ValueError("Hermes-routed memory LLM called with no messages")

        cfg = self.config
        params: Dict[str, Any] = {
            "task": cfg.task,
            "messages": [dict(m) for m in messages],
        }
        # Explicit pins win; leaving them unset is what makes the call land on
        # the user's main provider/model via auxiliary_client's auto chain.
        if cfg.provider:
            params["provider"] = cfg.provider
        if cfg.model:
            params["model"] = cfg.model
        if cfg.base_url:
            params["base_url"] = cfg.base_url
        if cfg.api_key:
            params["api_key"] = cfg.api_key
        if cfg.temperature is not None:
            params["temperature"] = cfg.temperature
        if cfg.max_tokens:
            params["max_tokens"] = int(cfg.max_tokens)
        if cfg.timeout:
            params["timeout"] = float(cfg.timeout)
        if tools:
            params["tools"] = tools

        wants_json = _wants_json(response_format)
        # json_mode="response_format" forwards the constraint to the provider
        # (only safe on OpenAI-compatible endpoints); the default "prompt" mode
        # relies on Mem0's own "return only JSON" instructions plus the
        # coerce_json() cleanup below, which works on every provider.
        if wants_json and cfg.json_mode == "response_format":
            extra_body = dict(params.get("extra_body") or {})
            extra_body["response_format"] = (
                response_format if isinstance(response_format, dict) else {"type": "json_object"}
            )
            params["extra_body"] = extra_body

        from agent.auxiliary_client import call_llm

        try:
            response = call_llm(**params)
        except Exception as exc:
            # Surface a message that names this plugin: it lands in Mem0's
            # LLMError and then in the provider's circuit-breaker log.
            raise RuntimeError(
                f"Hermes-routed memory LLM call failed ({cfg.describe()}): {exc}"
            ) from exc

        self.call_count += 1
        self.last_model = str(getattr(response, "model", "") or cfg.model or "")
        text = _extract_text(response)

        if tools:
            return {"content": text, "tool_calls": _extract_tool_calls(response)}

        if not text.strip():
            raise RuntimeError(
                "Hermes-routed memory LLM returned an empty response "
                f"({cfg.describe()}, model={self.last_model or 'unknown'})"
            )

        if wants_json and cfg.json_mode != "off":
            text = coerce_json(text)
        logger.debug(
            "mem0_hermes: routed %s call through %s (%d chars)",
            cfg.task, self.last_model or cfg.describe(), len(text),
        )
        return text


def register_with_mem0() -> str:
    """Register this adapter with Mem0's ``LlmFactory`` and return its id.

    Mem0 resolves factory entries by dotted import path, so the module is
    aliased under a stable name first — the plugin loader imports it as
    ``_hermes_user_memory.mem0_hermes._hermes_llm``, which the factory could
    not reconstruct on its own.
    """
    sys.modules.setdefault(_MODULE_ALIAS, sys.modules[__name__])
    from mem0.utils.factory import LlmFactory

    LlmFactory.register_provider(PROVIDER_NAME, _CLASS_PATH, RoutedLlmConfig)
    return PROVIDER_NAME
