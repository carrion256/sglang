"""CPU-only imported API conversion tests with an unmodified pinned tokenizer."""
import copy
import unittest
from runtime_chat_effort import ChatEffortTest
from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, ResponsesRequest
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses


class QwenAliasTest(unittest.TestCase):
    setUpClass = classmethod(ChatEffortTest.setUpClass.__func__)

    def setUp(self):
        ChatEffortTest.setUp(self)
        self.chat.tokenizer_manager.model_config.hf_config.model_type = 'qwen3_8_flash_next'

    def test_chat_alias_tokens_and_literal_provenance(self):
        messages = [{'role': 'user', 'content': 'Hi'}]
        for alias, effective in (('minimal', 'low'), ('high', 'xhigh'), ('max', 'xhigh')):
            expected = self.tokenizer.apply_chat_template(messages, tokenize=True,
                return_dict=False, add_generation_prompt=True, reasoning_effort=effective)
            for stream in (False, True):
                for fields in ({'reasoning_effort': alias},
                               {'chat_template_kwargs': {'reasoning_effort': alias}},
                               {'reasoning': {'effort': alias}}):
                    with self.subTest(alias=alias, stream=stream, fields=fields):
                        req = ChatCompletionRequest(model='fixture-qwen', messages=messages,
                            stream=stream, **copy.deepcopy(fields))
                        before = req.model_dump()
                        internal, normalized = self.chat._convert_to_internal_request(req)
                        self.assertEqual(internal.input_ids, expected)
                        self.assertEqual(req.model_dump(), before)
                        self.assertEqual(normalized.reasoning_effort, alias)

    def test_supported_values_precedence_null_and_no_leak(self):
        messages = [{'role': 'user', 'content': 'Hi'}]
        cases = [({}, 'medium'), ({'reasoning_effort': None}, 'medium'),
                 ({'chat_template_kwargs': {'reasoning_effort': None}}, 'medium')]
        for effort in ('minimal', 'low', 'medium', 'xhigh', 'high', 'max'):
            effective = {'minimal': 'low', 'high': 'xhigh', 'max': 'xhigh'}.get(effort, effort)
            cases.extend([({'reasoning_effort': effort}, effective),
                ({'reasoning_effort': effort, 'chat_template_kwargs': {'reasoning_effort': None}}, effective),
                ({'reasoning_effort': 'max', 'chat_template_kwargs': {'reasoning_effort': effort}}, effective)])
        for model_type in ('qwen3_8_flash_next', 'qwen3_8_flash_next_text'):
            self.chat.tokenizer_manager.model_config.hf_config.model_type = model_type
            for fields, effective in cases:
                with self.subTest(model_type=model_type, fields=fields):
                    req = ChatCompletionRequest(model='arbitrary-client-name', messages=messages, **fields)
                    before = req.model_dump()
                    internal, _ = self.chat._convert_to_internal_request(req)
                    expected = self.tokenizer.apply_chat_template(messages, tokenize=True,
                        return_dict=False, add_generation_prompt=True, reasoning_effort=effective)
                    self.assertEqual(internal.input_ids, expected)
                    self.assertEqual(req.model_dump(), before)
                    self.assertEqual(self.chat.default_chat_template_kwargs, {'reasoning_effort': 'medium'})

    def test_responses_real_conversion_and_literal_effort(self):
        import asyncio
        responses = OpenAIServingResponses.__new__(OpenAIServingResponses)
        responses.__dict__.update(self.chat.__dict__)
        for stream in (False, True):
            for effort in ('minimal', 'high', 'max', 'low', 'medium', 'xhigh', None):
                for nested in (None, 'minimal', 'low', 'high', 'max'):
                    with self.subTest(stream=stream, effort=effort, nested=nested):
                        req = ResponsesRequest(model='fixture-qwen', input='Hi', stream=stream,
                            reasoning={'effort': effort}, chat_template_kwargs={'reasoning_effort': nested})
                        before = req.model_dump()
                        messages, _, prompts, _ = asyncio.run(responses._make_request(req, None, self.tokenizer))
                        effective = nested or effort or 'medium'
                        effective = {'minimal': 'low', 'high': 'xhigh', 'max': 'xhigh'}.get(
                            effective, effective)
                        expected = self.tokenizer.apply_chat_template(messages, tokenize=True,
                            return_dict=False, add_generation_prompt=True, reasoning_effort=effective)
                        self.assertEqual(prompts, [expected])
                        self.assertEqual(req.model_dump(), before)

    def test_anthropic_messages_effort_uses_shared_aliases(self):
        serving = AnthropicServing(self.chat)
        messages = [{'role': 'user', 'content': 'Hi'}]
        for stream in (False, True):
            for effort, literal, effective in (
                ('minimal', 'minimal', 'low'),
                ('low', 'low', 'low'),
                ('medium', 'medium', 'medium'),
                ('high', 'high', 'xhigh'),
                ('xhigh', 'max', 'xhigh'),
                ('max', 'max', 'xhigh'),
            ):
                with self.subTest(stream=stream, effort=effort):
                    request = AnthropicMessagesRequest(model='fixture-qwen',
                        messages=messages, max_tokens=64, stream=stream,
                        output_config={'effort': effort})
                    chat_request = serving._convert_to_chat_completion_request(request)
                    before = chat_request.model_dump()
                    internal, normalized = self.chat._convert_to_internal_request(chat_request)
                    expected = self.tokenizer.apply_chat_template(messages, tokenize=True,
                        return_dict=False, add_generation_prompt=True,
                        reasoning_effort=effective)
                    self.assertEqual(internal.input_ids, expected)
                    self.assertEqual(chat_request.model_dump(), before)
                    self.assertEqual(normalized.reasoning_effort, literal)

    def test_unrelated_model_native_efforts_are_not_aliased(self):
        # A real tokenizer with a tiny native-effort template, not a mocked renderer.
        tokenizer = copy.deepcopy(self.tokenizer)
        tokenizer.chat_template = '{{ reasoning_effort }}'
        self.chat.tokenizer_manager.tokenizer = tokenizer
        for model_type in ('qwen4', 'qwen3', 'llama', 'deepseek_v3'):
            self.chat.tokenizer_manager.model_config.hf_config.model_type = model_type
            for effort in ('minimal', 'high'):
                req = ChatCompletionRequest(model='qwen3_8_flash_next',
                    messages=[{'role': 'user', 'content': 'Hi'}], reasoning_effort=effort)
                internal, normalized = self.chat._convert_to_internal_request(req)
                self.assertEqual(internal.input_ids,
                    tokenizer.encode(effort, add_special_tokens=False))
                self.assertEqual(normalized.reasoning_effort, effort)

    def test_tokenize_and_multimodal_render_paths(self):
        from sglang.srt.entrypoints.openai.protocol import TokenizeRequest
        from sglang.srt.entrypoints.openai.serving_tokenize import OpenAIServingTokenize
        serving = OpenAIServingTokenize(self.chat.tokenizer_manager, self.chat.template_manager)
        messages = [{'role': 'user', 'content': 'Hi'}]
        for effort, effective in (('minimal', 'low'), ('high', 'xhigh'),
                                  ('max', 'xhigh'), ('xhigh', 'xhigh')):
            with self.subTest(effort=effort):
                req = TokenizeRequest(messages=messages, reasoning_effort=effort)
                expected = self.tokenizer.apply_chat_template(messages, tokenize=True,
                    return_dict=False, add_generation_prompt=True, reasoning_effort=effective)
                self.assertEqual(serving._tokenize_chat_request(req), expected)
                self.chat.tokenizer_manager.model_config.is_multimodal = True
                chat_req = ChatCompletionRequest(messages=messages, reasoning_effort=effort)
                internal, _ = self.chat._convert_to_internal_request(chat_req)
                self.assertEqual(internal.text, self.tokenizer.apply_chat_template(messages,
                    tokenize=False, add_generation_prompt=True, reasoning_effort=effective))
                self.chat.tokenizer_manager.model_config.is_multimodal = False

    def test_server_default_and_absence_semantics(self):
        messages = [{'role': 'user', 'content': 'Hi'}]
        for defaults, expected_effort in (({}, 'xhigh'), ({'reasoning_effort': 'minimal'}, 'low'),
                ({'reasoning_effort': 'high'}, 'xhigh'),
                ({'reasoning_effort': 'max'}, 'xhigh'), ({'reasoning_effort': 'medium'}, 'medium')):
            self.chat.default_chat_template_kwargs = defaults
            for fields in ({}, {'reasoning_effort': None},
                           {'chat_template_kwargs': {'reasoning_effort': None}}):
                with self.subTest(defaults=defaults, fields=fields):
                    req = ChatCompletionRequest(messages=messages, **fields)
                    before = req.model_dump()
                    internal, _ = self.chat._convert_to_internal_request(req)
                    expected = self.tokenizer.apply_chat_template(messages, tokenize=True,
                        return_dict=False, add_generation_prompt=True, reasoning_effort=expected_effort)
                    self.assertEqual(internal.input_ids, expected)
                    self.assertEqual(req.model_dump(), before)

    def test_tokenize_precedence_null_and_provenance(self):
        from sglang.srt.entrypoints.openai.protocol import TokenizeRequest
        from sglang.srt.entrypoints.openai.serving_tokenize import OpenAIServingTokenize
        serving = OpenAIServingTokenize(self.chat.tokenizer_manager, self.chat.template_manager)
        messages = [{'role': 'user', 'content': 'Hi'}]
        for fields, effort in (({'reasoning_effort': 'max', 'chat_template_kwargs': {'reasoning_effort': 'minimal'}}, 'low'),
                ({'reasoning_effort': 'minimal', 'chat_template_kwargs': {'reasoning_effort': None}}, 'low'),
                ({'reasoning_effort': 'high', 'chat_template_kwargs': {'reasoning_effort': None}}, 'xhigh'),
                ({'chat_template_kwargs': {'reasoning_effort': None}}, 'medium')):
            req = TokenizeRequest(messages=messages, **fields)
            before = req.model_dump()
            expected = self.tokenizer.apply_chat_template(messages, tokenize=True,
                return_dict=False, add_generation_prompt=True, reasoning_effort=effort)
            self.assertEqual(serving._tokenize_chat_request(req), expected)
            self.assertEqual(req.model_dump(), before)
        serving.tokenizer_manager.model_config.hf_config.model_type = 'llama'
        tokenizer = copy.deepcopy(self.tokenizer)
        tokenizer.chat_template = '{{ reasoning_effort }}'
        serving.tokenizer_manager.tokenizer = tokenizer
        # Direct-tokenize's pre-existing server-default precedence is unchanged.
        req = TokenizeRequest(messages=messages, reasoning_effort='high',
            chat_template_kwargs={'reasoning_effort': 'high'})
        self.assertEqual(serving._tokenize_chat_request(req), tokenizer.encode('high', add_special_tokens=False))

    def test_processing_special_token_state_survives_render_copy(self):
        tools = [{'type': 'function', 'function': {'name': 'lookup',
            'parameters': {'type': 'object', 'properties': {}}}}]
        for effort in ('minimal', 'medium', 'high', 'max'):
            for parser, request_tools in ((None, tools), ('mistral', None)):
                with self.subTest(effort=effort, parser=parser):
                    self.chat.reasoning_parser = parser
                    req = ChatCompletionRequest(messages=[{'role': 'user', 'content': 'Hi'}],
                        reasoning_effort=effort, tools=request_tools)
                    before = req.model_dump()
                    internal, normalized = self.chat._convert_to_internal_request(req)
                    self.assertFalse(normalized.skip_special_tokens)
                    self.assertFalse(internal.sampling_params['skip_special_tokens'])
                    self.assertEqual(normalized.reasoning_effort, effort)
                    self.assertEqual(req.model_dump(), before)

    def test_invalid_values_still_fail(self):
        ChatEffortTest.test_invalid_effort_rejected_in_both_forms(self)
        for effort in ('MAX', ' high', 'x-high', True, [], {}):
            with self.subTest(effort=effort), self.assertRaises(ValueError):
                ResponsesRequest(model='fixture-qwen', input='Hi', reasoning={'effort': effort})


if __name__ == '__main__':
    unittest.main()
