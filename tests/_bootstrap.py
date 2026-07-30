# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Import bootstrap shared by the test modules.

The plugin lives at the repo root so that ``hermes plugins install owner/repo``
finds ``plugin.yaml`` there. That means the package's directory name is whatever
the checkout is called, which is not importable as ``mem0_hermes`` — so this
module loads the repo root under that fixed name explicitly, the same way
Hermes's own plugin loader does. It also puts the ``hermes-agent`` checkout on
``sys.path`` (for ``agent.*``); when that isn't available, minimal stubs are
installed instead so the pure-logic tests still run under a bare interpreter.

Run the suite with::

    python -m unittest discover -s tests -v

Use the Hermes venv's interpreter to include the Mem0 integration test:

    & "$env:LOCALAPPDATA\\hermes\\hermes-agent\\venv\\Scripts\\python.exe" -m unittest discover -s tests -v
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HERMES_AGENT_DIR = Path(
    os.environ.get("HERMES_AGENT_DIR") or (REPO_ROOT / "hermes-agent")
)


def _prepend(path: Path) -> None:
    text = str(path)
    if path.is_dir() and text not in sys.path:
        sys.path.insert(0, text)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _install_memory_provider_stub() -> None:
    """Minimal ``agent.memory_provider`` so the plugin package imports."""
    agent_pkg = sys.modules.get("agent")
    if agent_pkg is None:
        agent_pkg = types.ModuleType("agent")
        agent_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["agent"] = agent_pkg

    module = types.ModuleType("agent.memory_provider")

    class MemoryProvider:  # noqa: D401 - stub mirrors the real ABC loosely
        """Stub stand-in for the real MemoryProvider ABC."""

    module.MemoryProvider = MemoryProvider  # type: ignore[attr-defined]
    sys.modules["agent.memory_provider"] = module
    agent_pkg.memory_provider = module  # type: ignore[attr-defined]


def install_call_llm(fake) -> None:
    """Point ``agent.auxiliary_client.call_llm`` at ``fake``.

    Patches the real module when it is importable, otherwise creates a stub —
    the adapter imports the symbol lazily, so either works.
    """
    if _module_available("agent.auxiliary_client"):
        module = importlib.import_module("agent.auxiliary_client")
    else:
        module = sys.modules.get("agent.auxiliary_client")
        if module is None:
            module = types.ModuleType("agent.auxiliary_client")
            sys.modules["agent.auxiliary_client"] = module
            agent_pkg = sys.modules.get("agent")
            if agent_pkg is not None:
                agent_pkg.auxiliary_client = module  # type: ignore[attr-defined]
    module.call_llm = fake  # type: ignore[attr-defined]


PLUGIN_PACKAGE = "mem0_hermes"

HERMES_AGENT_AVAILABLE = False


def _load_plugin_package() -> None:
    """Import the repo root as the ``mem0_hermes`` package."""
    if PLUGIN_PACKAGE in sys.modules:
        return
    init_file = REPO_ROOT / "__init__.py"
    if not init_file.is_file():
        raise ImportError(f"plugin package not found at {init_file}")
    spec = importlib.util.spec_from_file_location(
        PLUGIN_PACKAGE, str(init_file), submodule_search_locations=[str(REPO_ROOT)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build a spec for {init_file}")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so ``from . import _config`` inside the package
    # resolves against a parent that already exists.
    sys.modules[PLUGIN_PACKAGE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(PLUGIN_PACKAGE, None)
        raise


def bootstrap() -> None:
    global HERMES_AGENT_AVAILABLE
    # Hard stop on runtime pip installs. Selecting an embedder normally installs
    # its packages, and a test that reached that path would silently mutate
    # whichever interpreter is running the suite (it did, once).
    os.environ.setdefault("MEM0_HERMES_NO_INSTALL", "1")
    _prepend(HERMES_AGENT_DIR)
    HERMES_AGENT_AVAILABLE = _module_available("agent.memory_provider")
    if not HERMES_AGENT_AVAILABLE:
        _install_memory_provider_stub()
    _load_plugin_package()


bootstrap()


class FakeMessage:
    def __init__(self, content, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeResponse:
    """Minimal OpenAI-shaped response object."""

    def __init__(self, content="", tool_calls=None, model="fake-model",
                 reasoning_content=None):
        self.choices = [FakeChoice(FakeMessage(content, tool_calls, reasoning_content))]
        self.model = model


class FakeToolCall:
    def __init__(self, name, arguments):
        self.function = types.SimpleNamespace(name=name, arguments=arguments)


class RecordingCallLlm:
    """Callable that records kwargs and returns a canned response."""

    def __init__(self, response=None, error=None):
        self.response = response if response is not None else FakeResponse("{}")
        self.error = error
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response

    @property
    def last(self):
        return self.calls[-1]
