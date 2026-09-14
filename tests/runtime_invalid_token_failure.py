"""CPU regressions for out-of-vocabulary generated token failures."""
import asyncio
from array import array
from http import HTTPStatus
import json
from types import SimpleNamespace
import unittest

from fastapi import HTTPException

from runtime_chat_effort import ChatEffortTest
from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.entrypoints.context import SimpleContext
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    RequestResponseMetadata,
    ResponsesRequest,
)
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.tokenizer_manager import TokenizerManager


def scheduled(ids, cap=20, eos=True, accepted=None):
    req = Req.__new__(Req)
    req.output_ids = array("q", ids)
    req.vocab_size = 100
    req.sampling_params = SimpleNamespace(
        stop_token_ids={2} if eos else set(),
        max_new_tokens=cap,
        stop_strs=[],
        stop_regex_strs=[],
        ignore_eos=False,
    )
    req.eos_token_ids = {2} if eos else set()
    req.finished_reason = None
    req.finished_len = None
    req.to_finish = None
    req.grammar = None
    req.tokenizer = None
    req.update_finish_state(new_accepted_len=accepted or len(ids))
    return req


def invalid_finish(serialized=False):
    finish = {
        "type": "abort",
        "message": "Generation produced an invalid token ID.",
        "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
        "err_type": "InvalidTokenError",
    }
    if serialized:
        finish["status_code"] = int(finish["status_code"])
    return finish


def engine_chunk(finish):
    return {
        "text": "Planning only.",
        "output_ids": [5],
        "meta_info": {
            "id": "fixture-rid",
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "cached_tokens": 0,
            "reasoning_tokens": 1,
            "finish_reason": finish,
        },
    }


def data_payloads(frames):
    return [
        json.loads(line[6:])
        for frame in frames
        for line in frame.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


class InvalidTokenFailureTest(unittest.TestCase):
    setUpClass = classmethod(ChatEffortTest.setUpClass.__func__)

    def setUp(self):
        ChatEffortTest.setUp(self)
        manager = self.chat.tokenizer_manager
        manager.model_config.hf_config.model_type = "qwen3_8_flash_next"
        manager.model_config.context_len = 32768
        manager.num_reserved_tokens = 0
        manager.server_args.incremental_streaming_output = False
        self.responses = OpenAIServingResponses(manager, self.chat.template_manager)
        self.responses.reasoning_parser = None
        self.responses.tool_call_parser = None

    def test_invalid_token_is_failure_even_past_length_cap(self):
        for bad_id in (-1, 100, 123456):
            for cap in (1, 2, 20):
                for eos in (False, True):
                    with self.subTest(bad_id=bad_id, cap=cap, eos=eos):
                        req = scheduled([5, 6, bad_id, 9], cap, eos)
                        finish = req.finished_reason.to_json()
                        self.assertEqual(finish["type"], "abort")
                        self.assertEqual(
                            finish["status_code"], HTTPStatus.INTERNAL_SERVER_ERROR
                        )
                        self.assertEqual(finish["err_type"], "InvalidTokenError")
                        self.assertEqual(list(req.output_ids_through_stop), [5, 6][:cap])

    def test_invalid_first_token_never_reaches_decode(self):
        req = scheduled([-1], eos=False)
        self.assertEqual(list(req.output_ids_through_stop), [])
        self.assertEqual(req.finished_reason.to_json()["type"], "abort")

    def test_ordinary_scheduler_finishes_are_unchanged(self):
        for ids, cap, expected in (
            ([5, 2], 20, "stop"),
            ([5, 6, 2], 1, "length"),
            ([5, 6], 2, "length"),
            ([5, 6], 20, None),
        ):
            with self.subTest(ids=ids, cap=cap):
                req = scheduled(ids, cap)
                actual = (
                    req.finished_reason.to_json()["type"]
                    if req.finished_reason
                    else None
                )
                self.assertEqual(actual, expected)

    def test_abort_cleanup_handles_serialized_status(self):
        for is_stream in (False, True):
            with self.subTest(is_stream=is_stream):
                item = engine_chunk(invalid_finish(serialized=True))
                manager = SimpleNamespace(
                    rid_to_state={"fixture-rid": object()}, enable_lora=False
                )
                state = SimpleNamespace(obj=SimpleNamespace(rid="fixture-rid"))

                async def run():
                    return await TokenizerManager._handle_abort_finish_reason(
                        manager, item, state, is_stream
                    )

                if is_stream:
                    self.assertIs(asyncio.run(run()), item)
                else:
                    with self.assertRaises(HTTPException) as raised:
                        asyncio.run(run())
                    self.assertEqual(raised.exception.status_code, 500)
                self.assertNotIn("fixture-rid", manager.rid_to_state)

    def test_chat_and_messages_stream_serialized_failure(self):
        finish = invalid_finish(serialized=True)

        async def generate(*args, **kwargs):
            yield engine_chunk(finish)

        self.chat.tokenizer_manager.generate_request = generate
        adapted = SimpleNamespace(rid="fixture-rid")
        chat_request = ChatCompletionRequest(
            model="fixture-qwen",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        async def collect_chat():
            return [
                frame
                async for frame in self.chat._generate_chat_stream(
                    adapted, chat_request, None
                )
            ]

        chat_frames = asyncio.run(collect_chat())
        self.assertEqual(chat_frames[-1], "data: [DONE]\n\n")
        chat_payloads = data_payloads(chat_frames)
        self.assertEqual(chat_payloads[-1]["error"]["code"], 500)

        anthropic_request = AnthropicMessagesRequest(
            model="fixture-qwen",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=64,
            stream=True,
        )
        anthropic = AnthropicServing(self.chat)

        async def collect_messages():
            return [
                frame
                async for frame in anthropic._generate_anthropic_stream(
                    adapted, chat_request, anthropic_request, None
                )
            ]

        message_payloads = data_payloads(asyncio.run(collect_messages()))
        self.assertTrue(
            any(
                payload.get("type") == "error"
                and payload["error"]["type"] == "api_error"
                for payload in message_payloads
            )
        )

    def test_responses_full_and_stream_expose_failure(self):
        finish = invalid_finish(serialized=True)
        chunk = engine_chunk(finish)
        request = ResponsesRequest(
            model="fixture-qwen", input="Hi", stream=False, store=True
        )
        metadata = RequestResponseMetadata(request_id=request.request_id)
        context = SimpleContext()

        async def full_result():
            context.append_output(chunk)
            yield context

        response = asyncio.run(
            self.responses.responses_full_generator(
                request,
                {},
                full_result(),
                context,
                "fixture-qwen",
                self.chat.tokenizer_manager.tokenizer,
                metadata,
                require_reasoning=False,
            )
        )
        self.assertEqual(response.status, "failed")
        self.assertEqual(response.error["code"], "server_error")
        self.assertIn("invalid token ID", response.error["message"])

        stream_request = ResponsesRequest(
            model="fixture-qwen", input="Hi", stream=True, store=True
        )
        stream_metadata = RequestResponseMetadata(request_id=stream_request.request_id)

        async def stream_result():
            yield chunk

        async def collect_responses():
            return [
                frame
                async for frame in self.responses.responses_stream_generator_non_harmony(
                    stream_request,
                    {},
                    stream_result(),
                    "fixture-qwen",
                    self.chat.tokenizer_manager.tokenizer,
                    stream_metadata,
                    require_reasoning=False,
                )
            ]

        events = data_payloads(asyncio.run(collect_responses()))
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertEqual(events[-1]["response"]["status"], "failed")
        self.assertIn(
            "invalid token ID", events[-1]["response"]["error"]["message"]
        )
        self.assertFalse(any(event["type"] == "response.completed" for event in events))

    def test_graceful_abort_remains_cancelled(self):
        finish = {"type": "abort", "message": "Request cancelled."}
        self.assertEqual(self.responses._status_from_finish_reason(finish), "cancelled")
        self.assertIsNone(self.responses._error_from_finish_reason(finish))


if __name__ == "__main__":
    unittest.main()
