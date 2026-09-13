"""GPU-free actual Chat conversion and deployed-tokenizer regression.

Run inside the runtime with QWEN_TOKENIZER_PATH pointing to a read-only tokenizer.
Only the manager/scheduler is a fixture; Chat, protocol and tokenizer are imported.
"""
import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from transformers import AutoTokenizer
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat


class ChatEffortTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            os.environ['QWEN_TOKENIZER_PATH'], local_files_only=True,
            trust_remote_code=False,
        )

    def setUp(self):
        tm = Mock()
        tm.tokenizer = self.tokenizer
        tm.model_config.is_multimodal = False
        tm.model_config.hf_config.model_type = 'qwen4'
        tm.model_config.hf_config.architectures = ['Qwen4ForCausalLM']
        tm.model_config.get_default_sampling_params.return_value = {}
        tm.server_args.default_chat_template_kwargs = {'reasoning_effort': 'medium'}
        tm.config_value.side_effect = lambda name: None
        tm.model_path = 'fixture-qwen'
        tm.served_model_name = 'fixture-qwen'
        tm.server_args.enable_cache_report = False
        template = SimpleNamespace(chat_template_name=None,
            jinja_template_content_format='string', reasoning_config=None,
            force_reasoning=False)
        self.chat = OpenAIServingChat(tm, template)

    def test_top_level_xhigh_reaches_real_tokenizer(self):
        messages = [{'role': 'user', 'content': 'Hi'}]
        for stream in (False, True):
            with self.subTest(stream=stream):
                req = ChatCompletionRequest(model='fixture-qwen', messages=messages,
                    reasoning_effort='xhigh', stream=stream, max_tokens=64)
                internal, _ = self.chat._convert_to_internal_request(req)
                expected = self.tokenizer.apply_chat_template(messages,
                    tokenize=True, return_dict=False, add_generation_prompt=True, reasoning_effort='xhigh')
                medium = self.tokenizer.apply_chat_template(messages,
                    tokenize=True, return_dict=False, add_generation_prompt=True, reasoning_effort='medium')
                self.assertNotEqual(expected, medium)
                self.assertEqual(internal.input_ids, expected)


    def test_effort_matrix_and_no_cross_request_leak(self):
        messages = [{'role': 'user', 'content': 'Hi'}]
        cases = [({}, 'medium'), ({'reasoning_effort': None}, 'medium')]
        for effort in ('xhigh', 'high', 'medium'):
            cases.extend([({'reasoning_effort': effort}, effort),
                ({'chat_template_kwargs': {'reasoning_effort': effort}}, effort)])
        cases.extend([
            ({'reasoning_effort': 'high', 'chat_template_kwargs': {'reasoning_effort': 'xhigh'}}, 'xhigh'),
            ({'reasoning_effort': 'xhigh', 'chat_template_kwargs': {'reasoning_effort': 'medium'}}, 'medium'),
            ({'reasoning_effort': 'xhigh', 'chat_template_kwargs': {'reasoning_effort': None}}, 'xhigh'),
            ({'chat_template_kwargs': {'reasoning_effort': None}}, 'medium'),
            ({}, 'medium'),
        ])
        for fields, effort in cases:
            for stream in (False, True):
                with self.subTest(fields=fields, stream=stream):
                    fields = copy.deepcopy(fields)
                    original = copy.deepcopy(fields)
                    req = ChatCompletionRequest(model='fixture-qwen', messages=messages,
                        stream=stream, max_completion_tokens=64, **fields)
                    self.assertEqual(fields, original, 'protocol mutated caller kwargs')
                    shared = req.chat_template_kwargs
                    shared_before = copy.deepcopy(shared)
                    if effort == 'high':
                        # The unchanged Qwen template rejects high (only xhigh,
                        # medium, low). Do not silently replace it with medium.
                        with self.assertRaisesRegex(ValueError, 'Unexpected reasoning effort high'):
                            self.chat._convert_to_internal_request(req)
                        self.assertEqual(shared, shared_before)
                        continue
                    internal, normalized = self.chat._convert_to_internal_request(req)
                    self.assertEqual(shared, shared_before, 'conversion mutated shared kwargs')
                    expected = self.tokenizer.apply_chat_template(messages, tokenize=True,
                        return_dict=False, add_generation_prompt=True, reasoning_effort=effort)
                    self.assertEqual(internal.input_ids, expected)
                    self.assertEqual(normalized.reasoning_effort, effort)
                    self.assertEqual(self.chat.default_chat_template_kwargs, {'reasoning_effort': 'medium'})

    def test_invalid_effort_rejected_in_both_forms(self):
        for value in ('unsupported', True, False, -1, 1.0, float('nan'), [], {}):
            for nested in (False, True):
                with self.subTest(value=repr(value), nested=nested):
                    fields = {'chat_template_kwargs': {'reasoning_effort': value}} if nested else {'reasoning_effort': value}
                    with self.assertRaises(ValueError):
                        req = ChatCompletionRequest(model='fixture-qwen',
                            messages=[{'role': 'user', 'content': 'Hi'}], **fields)
                        self.chat._convert_to_internal_request(req)

    def test_direct_renderer_policy_unchanged(self):
        req = ChatCompletionRequest(model='fixture-qwen', messages=[{'role': 'user', 'content': 'Hi'}], reasoning_effort='xhigh')
        processed = self.chat._process_messages(req, False)
        expected = self.tokenizer.apply_chat_template([{'role': 'user', 'content': 'Hi'}],
            tokenize=True, return_dict=False, add_generation_prompt=True, reasoning_effort='medium')
        self.assertEqual(processed.prompt_ids, expected)


if __name__ == '__main__':
    unittest.main()
