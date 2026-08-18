# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Tests for the Mem0 → Hermes LLM adapter."""

from __future__ import annotations

import json
import unittest

from _bootstrap import FakeResponse, FakeToolCall, RecordingCallLlm, install_call_llm

from mem0_hermes import _hermes_llm


class RoutedConfigTests(unittest.TestCase):
    def test_defaults_pin_nothing(self):
        cfg = _hermes_llm.RoutedLlmConfig()
        self.assertEqual(cfg.provider, "")
        self.assertEqual(cfg.model, "")
        self.assertEqual(cfg.task, "mem0_hermes_extraction")
        self.assertEqual(cfg.json_mode, "prompt")
        self.assertIn("main Hermes model", cfg.describe())

    def test_unknown_keys_are_tolerated(self):
        cfg = _hermes_llm.RoutedLlmConfig(**{"model": "m", "openai_base_url": "x"})
        self.assertEqual(cfg.model, "m")
        self.assertEqual(cfg.extra["openai_base_url"], "x")

    def test_describe_names_explicit_pin(self):
        cfg = _hermes_llm.RoutedLlmConfig(provider="anthropic", model="claude-opus-5")
        self.assertEqual(cfg.describe(), "anthropic/claude-opus-5")


class CoerceJsonTests(unittest.TestCase):
    def test_strips_code_fence(self):
        self.assertEqual(
            _hermes_llm.coerce_json('```json\n{"memory": []}\n```'), '{"memory": []}'
        )

    def test_strips_think_block_and_prose(self):
        raw = '<think>hmm, what to keep</think>Sure:\n{"memory": [{"text": "a"}]} done'
        self.assertEqual(_hermes_llm.coerce_json(raw), '{"memory": [{"text": "a"}]}')

    def test_handles_json_array(self):
        self.assertEqual(_hermes_llm.coerce_json('prefix [1, 2] suffix'), "[1, 2]")

    def test_keeps_an_array_of_objects_intact(self):
        # Slicing from the first "{" to the last "}" would return
        # '{"a": 1}, {"b": 2}' — not JSON at all.
        raw = '[{"a": 1}, {"b": 2}]'
        self.assertEqual(_hermes_llm.coerce_json(raw), raw)
        self.assertEqual(json.loads(_hermes_llm.coerce_json(raw)), [{"a": 1}, {"b": 2}])

    def test_single_element_array_does_not_collapse_to_an_object(self):
        # The dangerous case: preferring braces yields '{"text": "a"}', which
        # parses cleanly and silently loses the surrounding list.
        raw = 'here you go:\n[{"text": "a"}]'
        self.assertEqual(json.loads(_hermes_llm.coerce_json(raw)), [{"text": "a"}])

    def test_array_of_objects_inside_a_fence(self):
        raw = '```json\n[{"text": "a"}, {"text": "b"}]\n```'
        self.assertEqual(
            json.loads(_hermes_llm.coerce_json(raw)), [{"text": "a"}, {"text": "b"}]
        )

    def test_prose_brackets_do_not_beat_the_real_payload(self):
        # "[see above]" opens first but isn't the payload; the object is.
        raw = 'Note [see above]: {"memory": []}'
        self.assertEqual(_hermes_llm.coerce_json(raw), '{"memory": []}')

    def test_object_containing_an_array_is_unchanged(self):
        raw = '{"facts": ["a", "b"]}'
        self.assertEqual(_hermes_llm.coerce_json(raw), raw)

    def test_malformed_json_is_left_for_mem0_to_report(self):
        # Neither candidate parses; the outermost span is returned as before
        # rather than swallowing the response.
        self.assertEqual(_hermes_llm.coerce_json('{"a": 1,}'), '{"a": 1,}')

    def test_passes_through_plain_text(self):
        self.assertEqual(_hermes_llm.coerce_json("no json here"), "no json here")


class GenerateResponseTests(unittest.TestCase):
    def _run(self, *, config=None, response=None, **call_kwargs):
        fake = RecordingCallLlm(response=response)
        install_call_llm(fake)
        llm = _hermes_llm.HermesRoutedLLM(config or {})
        result = llm.generate_response(
            messages=call_kwargs.pop("messages", [{"role": "user", "content": "hi"}]),
            **call_kwargs,
        )
        return llm, fake, result

    def test_routes_to_main_model_by_default(self):
        _llm, fake, result = self._run(response=FakeResponse('{"memory": []}'))
        self.assertEqual(result, '{"memory": []}')
        self.assertEqual(fake.last["task"], "mem0_hermes_extraction")
        # No provider/model/api_key pinned → auxiliary_client resolves to the
        # user's main Hermes provider and model.
        for key in ("provider", "model", "base_url", "api_key"):
            self.assertNotIn(key, fake.last)
        self.assertEqual(fake.last["temperature"], 0.1)
        self.assertEqual(fake.last["timeout"], 120.0)
        self.assertNotIn("max_tokens", fake.last)

    def test_explicit_pins_are_forwarded(self):
        config = {
            "task": "custom_task",
            "provider": "anthropic",
            "model": "claude-opus-5",
            "base_url": "https://example.test/v1",
            "api_key": "secret",
            "temperature": 0,
            "max_tokens": 512,
            "timeout": 45,
        }
        _llm, fake, _ = self._run(config=config)
        self.assertEqual(fake.last["task"], "custom_task")
        self.assertEqual(fake.last["provider"], "anthropic")
        self.assertEqual(fake.last["model"], "claude-opus-5")
        self.assertEqual(fake.last["base_url"], "https://example.test/v1")
        self.assertEqual(fake.last["api_key"], "secret")
        self.assertEqual(fake.last["max_tokens"], 512)
        self.assertEqual(fake.last["timeout"], 45.0)

    def test_json_mode_prompt_does_not_send_response_format(self):
        _llm, fake, result = self._run(
            response=FakeResponse('```json\n{"memory": [1]}\n```'),
            response_format={"type": "json_object"},
        )
        self.assertNotIn("extra_body", fake.last)
        self.assertEqual(result, '{"memory": [1]}')

    def test_json_mode_response_format_forwards_via_extra_body(self):
        _llm, fake, _ = self._run(
            config={"json_mode": "response_format"},
            response=FakeResponse('{"memory": []}'),
            response_format={"type": "json_object"},
        )
        self.assertEqual(
            fake.last["extra_body"]["response_format"], {"type": "json_object"}
        )

    def test_json_mode_off_returns_text_untouched(self):
        raw = 'Here you go: {"memory": []}'
        _llm, _fake, result = self._run(
            config={"json_mode": "off"},
            response=FakeResponse(raw),
            response_format={"type": "json_object"},
        )
        self.assertEqual(result, raw)

    def test_tool_calls_returned_in_mem0_shape(self):
        response = FakeResponse(
            "",
            tool_calls=[FakeToolCall("add_memory", '{"text": "likes tea"}')],
        )
        _llm, fake, result = self._run(
            response=response, tools=[{"type": "function", "function": {"name": "add_memory"}}]
        )
        self.assertIn("tools", fake.last)
        self.assertEqual(result["content"], "")
        self.assertEqual(
            result["tool_calls"], [{"name": "add_memory", "arguments": {"text": "likes tea"}}]
        )

    def test_list_content_blocks_are_flattened(self):
        response = FakeResponse([{"type": "text", "text": '{"memory": '}, {"text": "[]}"}])
        _llm, _fake, result = self._run(response=response)
        self.assertEqual(result, '{"memory": []}')

    def test_reasoning_content_is_used_when_content_empty(self):
        response = FakeResponse("", reasoning_content='{"memory": []}')
        _llm, _fake, result = self._run(response=response)
        self.assertEqual(result, '{"memory": []}')

    def test_empty_response_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run(response=FakeResponse("   "))
        self.assertIn("empty response", str(ctx.exception))

    def test_provider_error_is_wrapped_with_routing_context(self):
        fake = RecordingCallLlm(error=ValueError("no credentials"))
        install_call_llm(fake)
        llm = _hermes_llm.HermesRoutedLLM({"model": "some-model"})
        with self.assertRaises(RuntimeError) as ctx:
            llm.generate_response(messages=[{"role": "user", "content": "hi"}])
        message = str(ctx.exception)
        self.assertIn("Hermes-routed memory LLM call failed", message)
        self.assertIn("some-model", message)
        self.assertIsInstance(ctx.exception.__cause__, ValueError)

    def test_no_messages_raises_before_calling_out(self):
        fake = RecordingCallLlm()
        install_call_llm(fake)
        llm = _hermes_llm.HermesRoutedLLM()
        with self.assertRaises(ValueError):
            llm.generate_response(messages=[])
        self.assertEqual(fake.calls, [])

    def test_call_metadata_recorded(self):
        llm, _fake, _ = self._run(response=FakeResponse('{}', model="gpt-x"))
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(llm.last_model, "gpt-x")


class FactoryRegistrationTests(unittest.TestCase):
    def test_registers_provider_and_module_alias(self):
        try:
            from mem0.utils.factory import LlmFactory  # noqa: F401
        except Exception:
            self.skipTest("mem0ai is not installed")
        import sys

        name = _hermes_llm.register_with_mem0()
        self.assertEqual(name, _hermes_llm.PROVIDER_NAME)
        self.assertIn(_hermes_llm._MODULE_ALIAS, sys.modules)

        from mem0.utils.factory import LlmFactory

        self.assertIn(name, LlmFactory.provider_to_class)
        # The dotted path the factory stores must resolve back to our class.
        created = LlmFactory.create(name, {"model": "pinned"})
        self.assertIsInstance(created, _hermes_llm.HermesRoutedLLM)
        self.assertEqual(created.config.model, "pinned")


if __name__ == "__main__":
    unittest.main()
