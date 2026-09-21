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
from sglang.srt.entrypoints.context import SimpleContext, StreamingHarmonyContext
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    RequestResponseMetadata,
    ResponsesRequest,
)
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
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
        self.completions = OpenAIServingCompletion(manager, self.chat.template_manager)
        self.responses = OpenAIServingResponses(manager, self.chat.template_manager)
        self.responses.reasoning_parser = None
        self.responses.tool_call_parser = None

    def _store_failed_response(self):
        request = ResponsesRequest(
            model="fixture-qwen", input="Hi", stream=False, store=True
        )

        async def generate(*args, **kwargs):
            yield engine_chunk(invalid_finish(serialized=True))

        self.responses.tokenizer_manager.generate_request = generate
        failed = asyncio.run(self.responses.create_responses(request))
        self.assertEqual(failed.status, "failed")
        return failed

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

    def test_completions_stream_serialized_failure_is_sse_error(self):
        async def generate(*args, **kwargs):
            yield engine_chunk(invalid_finish(serialized=True))

        self.completions.tokenizer_manager.generate_request = generate
        request = CompletionRequest(
            model="fixture-qwen",
            prompt="Hi",
            stream=True,
            stream_options={"include_usage": True},
        )

        async def collect():
            return [
                frame
                async for frame in self.completions._generate_completion_stream(
                    SimpleNamespace(rid="fixture-rid"), request, None
                )
            ]

        frames = asyncio.run(collect())
        self.assertEqual(frames[-1], "data: [DONE]\n\n")
        payloads = data_payloads(frames)
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[-1]["error"]["type"], "InvalidTokenError")
        self.assertEqual(payloads[-1]["error"]["code"], 500)
        self.assertFalse(
            any(
                choice.get("finish_reason") == "abort"
                for payload in payloads
                for choice in payload.get("choices", [])
            )
        )

    def test_completions_stream_graceful_abort_remains_choice(self):
        async def generate(*args, **kwargs):
            yield engine_chunk({"type": "abort", "message": "Request cancelled."})

        self.completions.tokenizer_manager.generate_request = generate
        request = CompletionRequest(model="fixture-qwen", prompt="Hi", stream=True)

        async def collect():
            return [
                frame
                async for frame in self.completions._generate_completion_stream(
                    SimpleNamespace(rid="fixture-rid"), request, None
                )
            ]

        frames = asyncio.run(collect())
        self.assertEqual(frames[-1], "data: [DONE]\n\n")
        payloads = data_payloads(frames)
        self.assertFalse(any("error" in payload for payload in payloads))
        choice_payloads = [payload for payload in payloads if payload.get("choices")]
        self.assertEqual(len(choice_payloads), 1)
        self.assertEqual(
            choice_payloads[0]["choices"][0]["finish_reason"], "abort"
        )
        self.assertEqual(payloads[0], choice_payloads[0])
        usage_payloads = [payload for payload in payloads if not payload.get("choices")]
        self.assertLessEqual(len(usage_payloads), 1)
        if usage_payloads:
            self.assertEqual(usage_payloads[0]["choices"], [])
            self.assertIn("usage", usage_payloads[0])
            self.assertEqual(payloads[-1], usage_payloads[0])

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
        self.assertEqual(
            chat_payloads[-1]["error"]["type"], "InvalidTokenError"
        )

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

    def test_harmony_responses_stream_failure_uses_failed_terminal_event(self):
        request = ResponsesRequest(
            model="fixture-qwen", input="Hi", stream=True, store=True
        )
        metadata = RequestResponseMetadata(request_id=request.request_id)
        self.responses.use_harmony = True
        self.responses._make_response_output_items_with_harmony = lambda context: []
        context = StreamingHarmonyContext.__new__(StreamingHarmonyContext)
        context.parser = SimpleNamespace(messages=[], last_content_delta=None)
        context.is_expecting_start = lambda: False
        context.is_assistant_action_turn = lambda: False
        context.num_init_messages = 0
        context.num_prompt_tokens = 0
        context.num_cached_tokens = 0
        context.num_output_tokens = 0
        context.num_reasoning_tokens = 0
        context.finish_reason = invalid_finish(serialized=True)

        async def stream_result():
            yield context

        async def collect_responses():
            return [
                frame
                async for frame in self.responses.responses_stream_generator(
                    request,
                    {},
                    stream_result(),
                    context,
                    "fixture-qwen",
                    self.chat.tokenizer_manager.tokenizer,
                    metadata,
                    require_reasoning=False,
                )
            ]

        events = data_payloads(asyncio.run(collect_responses()))
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertEqual(events[-1]["response"]["status"], "failed")
        self.assertFalse(any(event["type"] == "response.completed" for event in events))

    def test_failed_response_retrieval_preserves_failure_and_partial_output(self):
        failed = self._store_failed_response()

        retrieved = asyncio.run(self.responses.retrieve_responses(failed.id))

        self.assertEqual(retrieved.status, "failed")
        self.assertEqual(retrieved.error["code"], "server_error")
        self.assertIn("invalid token ID", retrieved.error["message"])
        self.assertTrue(retrieved.output)
        self.assertIn(
            "Planning only.",
            json.dumps([item.model_dump() for item in retrieved.output]),
        )

    def test_failed_response_cannot_be_replayed_as_predecessor(self):
        failed = self._store_failed_response()

        async def preprocessing_must_not_run(*args, **kwargs):
            self.fail("failed predecessor reached preprocessing")

        self.responses._make_request = preprocessing_must_not_run
        replay = ResponsesRequest(
            model="fixture-qwen",
            input="Continue",
            previous_response_id=failed.id,
            store=True,
        )

        rejected = asyncio.run(self.responses.create_responses(replay))

        self.assertEqual(rejected.status_code, 400)
        error = json.loads(rejected.body)["error"]
        self.assertEqual(error["param"], "previous_response_id")
        self.assertIn("status is 'failed'", error["message"])

    def test_graceful_abort_remains_cancelled(self):
        finish = {"type": "abort", "message": "Request cancelled."}
        self.assertEqual(self.responses._status_from_finish_reason(finish), "cancelled")
        self.assertIsNone(self.responses._error_from_finish_reason(finish))


if __name__ == "__main__":
    unittest.main()
