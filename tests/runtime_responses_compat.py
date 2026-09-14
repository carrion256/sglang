"""Imported exact-runtime CPU tests. No model/GPU conformance claim."""
import asyncio
import copy
import inspect
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from runtime_chat_effort import ChatEffortTest

from sglang.srt.entrypoints.openai.protocol import (
    RequestResponseMetadata,
    ResponseOutputMessage,
    ResponsesRequest,
    ResponsesResponse,
)
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses


class ResponsesCompatTest(unittest.TestCase):
    @staticmethod
    def phase_serving(model_type='qwen3_8_flash_next'):
        serving = object.__new__(OpenAIServingResponses)
        serving.msg_store = {}
        serving.tokenizer_manager = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type=model_type)
            )
        )
        return serving

    def test_message_phase_survives_response_models(self):
        from openai.types.responses import ResponseOutputText
        from sglang.srt.entrypoints.openai.responses_compat import ToolRegistry

        message = ResponseOutputMessage(
            id='msg_phase', type='message', role='assistant', status='completed',
            phase='commentary', content=[ResponseOutputText(
                type='output_text', text='Checking', annotations=[], logprobs=None)],
        )
        response = ResponsesResponse(
            id='resp_phase', model='fixture', status='completed', output=[message]
        )
        self.assertEqual(response.model_dump()['output'][0]['phase'], 'commentary')
        converted = ToolRegistry([]).response_model(response)
        self.assertEqual(converted.model_dump()['output'][0]['phase'], 'commentary')

    def test_qwen_replay_preserves_assistant_stage_order(self):
        serving = self.phase_serving()
        request = ResponsesRequest(model='fixture', input=[
            {'role': 'user', 'content': 'Inspect and report'},
            {'type': 'reasoning', 'summary': [
                {'type': 'summary_text', 'text': 'PLAN'}]},
            {'role': 'assistant', 'content': 'CHECKING', 'phase': 'commentary'},
            {'type': 'function_call', 'name': 'inspect', 'call_id': 'call_1',
             'arguments': '{}'},
            {'type': 'reasoning', 'summary': [
                {'type': 'summary_text', 'text': 'SECOND'}]},
            {'role': 'assistant', 'content': 'REPORT', 'phase': 'final_answer'},
        ])
        messages = serving._construct_input_messages(request)
        assistants = [message for message in messages if message['role'] == 'assistant']
        self.assertEqual(len(assistants), 2)
        self.assertEqual(
            (assistants[0]['reasoning_content'], assistants[0]['content'],
             assistants[0]['phase'], len(assistants[0]['tool_calls'])),
            ('PLAN', 'CHECKING', 'commentary', 1),
        )
        self.assertEqual(
            (assistants[1]['reasoning_content'], assistants[1]['content'],
             assistants[1]['phase']),
            ('SECOND', 'REPORT', 'final_answer'),
        )

    def test_stored_response_replays_reasoning_phase_and_call_together(self):
        serving = self.phase_serving()
        serving.msg_store['resp_prior'] = [
            {'role': 'user', 'content': 'Inspect and report'}
        ]
        previous = ResponsesResponse.model_validate({
            'id': 'resp_prior', 'model': 'fixture', 'status': 'completed',
            'output': [
                {'id': 'rs_1', 'type': 'reasoning', 'status': 'completed',
                 'summary': [{'type': 'summary_text', 'text': 'PLAN'}],
                 'content': []},
                {'id': 'msg_1', 'type': 'message', 'role': 'assistant',
                 'status': 'completed', 'phase': 'commentary',
                 'content': [{'type': 'output_text', 'text': 'CHECKING',
                              'annotations': []}]},
                {'id': 'fc_1', 'type': 'function_call', 'status': 'completed',
                 'name': 'inspect', 'call_id': 'call_1', 'arguments': '{}'},
            ],
        })
        request = ResponsesRequest(model='fixture', previous_response_id='resp_prior',
                                   input=[{'type': 'function_call_output',
                                           'call_id': 'call_1', 'output': 'HEALTHY'}])
        messages = serving._construct_input_messages(request, previous)
        self.assertEqual([message['role'] for message in messages],
                         ['user', 'assistant', 'tool'])
        assistant = messages[1]
        self.assertEqual(assistant['reasoning_content'], 'PLAN')
        self.assertEqual(assistant['content'], [{'type': 'text', 'text': 'CHECKING'}])
        self.assertEqual(assistant['phase'], 'commentary')
        self.assertEqual(assistant['tool_calls'][0]['id'], 'call_1')

    def test_harmony_same_request_call_replay(self):
        serving = object.__new__(OpenAIServingResponses)
        serving.tool_server = None
        request = ResponsesRequest(input=[
            {'type': 'function_call', 'name': 'lookup', 'call_id': 'call_harmony', 'arguments': '{}'},
            {'type': 'function_call_output', 'call_id': 'call_harmony', 'output': 'Found'}],
            tools=[{'type': 'function', 'name': 'lookup', 'description': 'Lookup', 'parameters': {'type': 'object'}}])
        messages = serving._construct_input_messages_with_harmony(request, None)
        self.assertEqual(messages[-1].author.name, 'functions.lookup')
    def test_namespace_declaration_reaches_chat(self):
        request = ResponsesRequest(input='hello', tools=[{
            'type': 'namespace', 'name': 'workspace', 'description': 'Workspace',
            'tools': [{'type': 'function', 'name': 'read',
                       'parameters': {'type': 'object', 'properties': {}}}]}])
        tools = OpenAIServingResponses._response_tools_to_chat_tools(request)
        self.assertEqual([tool.function.name for tool in tools], ['workspace.read'])

    def test_custom_declaration_reaches_chat(self):
        request = ResponsesRequest(input='hello', tools=[{
            'type': 'custom', 'name': 'patch', 'description': 'Raw patch'}])
        tools = OpenAIServingResponses._response_tools_to_chat_tools(request)
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].function.parameters['required'], ['input'])

    def test_qualified_replay(self):
        message = OpenAIServingResponses._normalize_response_message_for_chat({
            'type': 'function_call', 'name': 'read', 'namespace': 'workspace',
            'call_id': 'call_fixture', 'arguments': '{}'})
        self.assertEqual(message['tool_calls'][0]['function']['name'], 'workspace.read')

    def test_image_result_preserved(self):
        message = OpenAIServingResponses._normalize_response_message_for_chat({
            'type': 'function_call_output', 'call_id': 'call_fixture',
            'output': [{'type': 'input_image', 'image_url': 'data:image/png;base64,AA=='}]})
        self.assertIsInstance(message['content'], list)
        self.assertEqual(message['content'][0]['type'], 'image_url')

    def test_collisions_and_malformed_members(self):
        function = {'type': 'function', 'name': 'read', 'parameters': {'type': 'object'}}
        namespace = {'type': 'namespace', 'name': 'workspace', 'tools': [function]}
        cases = [[namespace, {'type': 'function', 'name': 'workspace.read'}],
                 [namespace, namespace],
                 [{'type': 'namespace', 'name': 'workspace', 'tools': [{'type': 'web_search', 'name': 'read'}]}],
                 [{'type': 'namespace', 'name': 'workspace', 'tools': [{'type': 'function', 'name': ''}]}],
                 [{'type': 'namespace', 'name': 'workspace', 'tools': [{'type': 'custom', 'name': 'patch', 'format': False}]}],
                 [{'type': 'namespace', 'name': 'workspace', 'tools': [{'type': 'custom', 'name': 'patch', 'format': {}}]}],
                 [{'type': 'namespace', 'name': 'workspace', 'tools': [{'type': 'function', 'name': 'read', 'description': False}]}],
                 [{'type': 'namespace', 'name': 'a.b', 'tools': [{'type': 'function', 'name': 'c'}]},
                  {'type': 'namespace', 'name': 'a', 'tools': [{'type': 'function', 'name': 'b.c'}]}]]
        for tools in cases:
            with self.subTest(tools=tools), self.assertRaises(ValueError):
                request = ResponsesRequest(input='hello', tools=tools)
                OpenAIServingResponses._response_tools_to_chat_tools(request)

    def test_custom_grammar_visible(self):
        request = ResponsesRequest(input='hello', tools=[{'type': 'custom', 'name': 'patch',
            'format': {'type': 'grammar', 'syntax': 'lark', 'definition': 'start: "patch"'}}])
        tools = OpenAIServingResponses._response_tools_to_chat_tools(request)
        self.assertIn('start: "patch"', tools[0].function.description)


class MockHTTPTest(unittest.TestCase):
    """MOCK generation via injected CPU manager; real http_server ASGI endpoints."""
    setUpClass = classmethod(ChatEffortTest.setUpClass.__func__)

    def setUp(self):
        from fastapi.testclient import TestClient
        from sglang.srt.entrypoints.http_server import app

        ChatEffortTest.setUp(self)
        manager = self.chat.tokenizer_manager
        manager.model_config.hf_config.model_type = 'qwen3_8_flash_next'
        manager.model_config.context_len = 32768
        manager.num_reserved_tokens = 0
        manager.server_args.incremental_streaming_output = False
        self.generated = []
        self.text = '[{"name":"workspace.read","parameters":{"path":"file"}}]'

        async def generate(request, *args, **kwargs):
            self.generated.append(request)
            pieces = range(1, len(self.text) + 1) if request.stream else [len(self.text)]
            for size in pieces:
                yield {'text': self.text[:size], 'output_ids': [1] * size,
                       'meta_info': {'prompt_tokens': 10, 'completion_tokens': size,
                                     'finish_reason': {'type': 'stop'} if size == len(self.text) else None}}

        manager.generate_request = generate
        self.serving = OpenAIServingResponses(manager, self.chat.template_manager)
        self.serving.tool_call_parser = None
        self.serving.reasoning_parser = None
        self.previous_serving = getattr(app.state, 'openai_serving_responses', None)
        app.state.openai_serving_responses = self.serving
        self.app = app
        self.client = TestClient(app)
        self.tools = [{'type': 'namespace', 'name': 'workspace', 'description': 'Workspace',
                       'tools': [{'type': 'function', 'name': 'read',
                                  'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}}}]}]

    def tearDown(self):
        self.client.close()
        self.app.state.openai_serving_responses = self.previous_serving

    def send(self, **kwargs):
        payload = dict(model='fixture-qwen', input='Read file', tools=self.tools,
                       tool_choice='required', max_output_tokens=128,
                       reasoning={'effort': 'medium'})
        payload.update(kwargs)
        return self.client.post('/v1/responses', json=payload)

    def events(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        return [json.loads(line[6:]) for line in response.text.splitlines()
                if line.startswith('data: ') and line != 'data: [DONE]']

    @staticmethod
    def phase_semantics(output):
        result = []
        for item in output:
            if item['type'] == 'message':
                result.append(('message', item['phase'],
                               ''.join(part['text'] for part in item['content'])))
            elif item['type'] == 'reasoning':
                result.append(('reasoning', ''.join(
                    part['text'] for part in item.get('content', []))))
            else:
                result.append(('function_call', item['name'],
                               json.loads(item['arguments'])))
        return result

    def test_qwen_stream_preserves_text_tool_text_order_for_any_chunking(self):
        self.serving.reasoning_parser = None
        self.serving.tool_call_parser = 'qwen3_coder'
        tools = [{'type': 'function', 'name': 'inspect',
                  'parameters': {'type': 'object', 'properties': {}}}]
        raw = ('Checking.<tool_call><function=inspect></function></tool_call>'
               'Final answer.')
        for incremental in (False, True):
            for delivery in ('coalesced', 'characters'):
                with self.subTest(incremental=incremental, delivery=delivery):
                    self.serving.tokenizer_manager.server_args.incremental_streaming_output = incremental
                    parts = [raw] if delivery == 'coalesced' else list(raw)

                    async def generate(request, *args, **kwargs):
                        cumulative = ''
                        for index, part in enumerate(parts):
                            cumulative += part
                            yield {
                                'text': part if incremental else cumulative,
                                'output_ids': [1] * (index + 1),
                                'meta_info': {
                                    'prompt_tokens': 10,
                                    'completion_tokens': index + 1,
                                    'finish_reason': (
                                        {'type': 'stop'} if index == len(parts) - 1 else None
                                    ),
                                },
                            }

                    self.serving.tokenizer_manager.generate_request = generate
                    events = self.events(self.send(
                        stream=True, tools=tools, tool_choice='auto'))
                    output = next(event['response']['output'] for event in events
                                  if event['type'] == 'response.completed')
                    self.assertEqual(self.phase_semantics(output), [
                        ('message', 'commentary', 'Checking.'),
                        ('function_call', 'inspect', {}),
                        ('message', 'final_answer', 'Final answer.'),
                    ])
                    done = [event['item'] for event in events
                            if event['type'] == 'response.output_item.done']
                    self.assertEqual(done, output)
                    added_messages = [event['item'] for event in events
                                      if event['type'] == 'response.output_item.added'
                                      and event['item']['type'] == 'message']
                    self.assertTrue(all(item.get('phase') is None
                                        for item in added_messages))

    def test_qwen_nonstream_preserves_text_tool_text_order(self):
        self.serving.reasoning_parser = None
        self.serving.tool_call_parser = 'qwen3_coder'
        self.text = ('Checking.<tool_call><function=inspect></function></tool_call>'
                     'Final answer.')
        tools = [{'type': 'function', 'name': 'inspect',
                  'parameters': {'type': 'object', 'properties': {}}}]
        output = self.send(tools=tools, tool_choice='auto').json()['output']
        self.assertEqual(self.phase_semantics(output), [
            ('message', 'commentary', 'Checking.'),
            ('function_call', 'inspect', {}),
            ('message', 'final_answer', 'Final answer.'),
        ])

    def test_qwen_ordered_nonstream_preserves_usage_details(self):
        self.serving.reasoning_parser = None
        self.serving.tool_call_parser = 'qwen3_coder'
        self.serving.enable_prompt_tokens_details = True
        raw = ('Checking.<tool_call><function=inspect></function></tool_call>'
               'Final answer.')

        async def generate(request, *args, **kwargs):
            yield {
                'text': raw,
                'output_ids': [1] * 108,
                'meta_info': {
                    'prompt_tokens': 10,
                    'completion_tokens': 108,
                    'cached_tokens': 4,
                    'reasoning_tokens': 7,
                    'finish_reason': {'type': 'stop'},
                },
            }

        self.serving.tokenizer_manager.generate_request = generate
        tools = [{'type': 'function', 'name': 'inspect',
                  'parameters': {'type': 'object', 'properties': {}}}]
        body = self.send(tools=tools, tool_choice='auto').json()
        self.assertEqual(body['usage'], {
            'input_tokens': 10,
            'input_tokens_details': {
                'cached_tokens': 4,
                'cache_write_tokens': 0,
            },
            'output_tokens': 108,
            'output_tokens_details': {'reasoning_tokens': 7},
            'total_tokens': 118,
        })

    def test_qwen_ordered_nonstream_preserves_requested_logprobs(self):
        self.serving.reasoning_parser = None
        self.serving.tool_call_parser = 'qwen3_coder'
        raw = ('Checking.<tool_call><function=inspect></function></tool_call>'
               'Final answer.')

        async def generate(request, *args, **kwargs):
            yield {
                'text': raw,
                'output_ids': [1, 2],
                'meta_info': {
                    'prompt_tokens': 10,
                    'completion_tokens': 2,
                    'output_token_logprobs': [
                        (-0.25, 1, 'Checking.'),
                        (-0.5, 2, 'Final answer.'),
                    ],
                    'output_top_logprobs': [
                        [(-0.25, 1, 'Checking.'), (-1.0, 3, 'Inspecting.')],
                        [(-0.5, 2, 'Final answer.')],
                    ],
                    'finish_reason': {'type': 'stop'},
                },
            }

        self.serving.tokenizer_manager.generate_request = generate
        tools = [{'type': 'function', 'name': 'inspect',
                  'parameters': {'type': 'object', 'properties': {}}}]
        body = self.send(
            tools=tools,
            tool_choice='auto',
            include=['message.output_text.logprobs'],
            top_logprobs=2,
        ).json()
        messages = [item for item in body['output'] if item['type'] == 'message']
        self.assertTrue(all(item['content'][0]['logprobs'] is not None
                            for item in messages), messages)
        self.assertEqual(
            [[entry['token'] for entry in item['content'][0]['logprobs']]
             for item in messages],
            [['Checking.', 'Final answer.'], ['Checking.', 'Final answer.']],
        )
        self.assertEqual(
            messages[0]['content'][0]['logprobs'][0]['top_logprobs'][1]['token'],
            'Inspecting.',
        )

    def test_qwen_literal_angle_brackets_survive_text_tool_text(self):
        self.serving.reasoning_parser = None
        self.serving.tool_call_parser = 'qwen3_coder'
        self.text = ('Compare <left> and 1 < 2.'
                     '<tool_call><function=inspect></function></tool_call>'
                     'Final <right> and 3 > 2.')
        tools = [{'type': 'function', 'name': 'inspect',
                  'parameters': {'type': 'object', 'properties': {}}}]
        body = self.send(tools=tools, tool_choice='auto').json()
        self.assertEqual(self.phase_semantics(body['output']), [
            ('message', 'commentary', 'Compare <left> and 1 < 2.'),
            ('function_call', 'inspect', {}),
            ('message', 'final_answer', 'Final <right> and 3 > 2.'),
        ])

    def test_qwen_stream_has_no_generic_angle_boundary_split(self):
        source = inspect.getsource(
            OpenAIServingResponses.responses_stream_generator_non_harmony)
        self.assertFalse(
            're.split(r"(?=<)|(?<=>)"' in source,
            'generic angle-boundary splitting is forbidden',
        )

    def test_qwen4_exp_is_negative_control_for_ordered_nonstream(self):
        self.serving.reasoning_parser = None
        self.serving.tool_call_parser = 'qwen3_coder'
        self.serving.tokenizer_manager.model_config.hf_config.model_type = 'qwen4_exp'
        self.text = ('Checking.<tool_call><function=inspect></function></tool_call>'
                     'Final answer.')
        tools = [{'type': 'function', 'name': 'inspect',
                  'parameters': {'type': 'object', 'properties': {}}}]
        output = self.send(tools=tools, tool_choice='auto').json()['output']
        self.assertEqual(self.phase_semantics(output), [
            ('message', 'commentary', 'Checking.'),
            ('function_call', 'inspect', {}),
        ])

    def test_nonstream_message_phase_matches_remaining_tool_calls(self):
        self.text = 'Final answer.'
        final = self.send(tools=[], tool_choice='none').json()['output']
        self.assertEqual(self.phase_semantics(final), [
            ('message', 'final_answer', 'Final answer.'),
        ])

        self.serving.tool_call_parser = 'qwen3_coder'
        self.text = ('Checking.<tool_call><function=inspect></function>'
                     '</tool_call>')
        tools = [{'type': 'function', 'name': 'inspect',
                  'parameters': {'type': 'object', 'properties': {}}}]
        mixed = self.send(tools=tools, tool_choice='auto').json()['output']
        self.assertEqual(self.phase_semantics(mixed), [
            ('message', 'commentary', 'Checking.'),
            ('function_call', 'inspect', {}),
        ])

    def test_qwen_nonstream_preserves_renewed_reasoning_order(self):
        self.serving.reasoning_parser = 'qwen3'
        self.serving.tool_call_parser = None
        self.text = 'Checking.<think>Again</think>Final answer.'
        output = self.send(
            tools=[], tool_choice='none', reasoning={'effort': 'medium'}
        ).json()['output']
        self.assertEqual(self.phase_semantics(output), [
            ('message', 'commentary', 'Checking.'),
            ('reasoning', 'Again'),
            ('message', 'final_answer', 'Final answer.'),
        ])

    def test_qwen_stream_preserves_renewed_reasoning_order(self):
        self.serving.reasoning_parser = 'qwen3'
        self.serving.tool_call_parser = None
        self.text = 'Checking.<think>Again</think>Final answer.'
        events = self.events(self.send(
            stream=True, tools=[], tool_choice='none', reasoning={'effort': 'medium'}))
        output = next(event['response']['output'] for event in events
                      if event['type'] == 'response.completed')
        self.assertEqual(self.phase_semantics(output), [
            ('message', 'commentary', 'Checking.'),
            ('reasoning', 'Again'),
            ('message', 'final_answer', 'Final answer.'),
        ])

    def test_qwen_stream_preserves_split_reasoning_marker_boundaries(self):
        self.serving.reasoning_parser = 'qwen3'
        self.serving.tool_call_parser = None
        raw = 'Checking.<think>Again</think>Final answer.'
        expected = [
            ('message', 'commentary', 'Checking.'),
            ('reasoning', 'Again'),
            ('message', 'final_answer', 'Final answer.'),
        ]
        start = raw.index('<think>')
        for incremental in (False, True):
            for cut in range(start + 1, start + len('<think>')):
                with self.subTest(incremental=incremental, cut=cut):
                    self.serving.tokenizer_manager.server_args.incremental_streaming_output = incremental

                    async def generate(request, *args, **kwargs):
                        yield {'text': raw[:cut], 'output_ids': [1] * cut,
                               'meta_info': {'prompt_tokens': 10,
                                             'completion_tokens': cut,
                                             'finish_reason': None}}
                        yield {'text': raw[cut:] if incremental else raw,
                               'output_ids': [1] * len(raw),
                               'meta_info': {'prompt_tokens': 10,
                                             'completion_tokens': len(raw),
                                             'finish_reason': {'type': 'stop'}}}

                    self.serving.tokenizer_manager.generate_request = generate
                    events = self.events(self.send(
                        stream=True, tools=[], tool_choice='none',
                        reasoning={'effort': 'medium'}))
                    output = next(event['response']['output'] for event in events
                                  if event['type'] == 'response.completed')
                    self.assertEqual(self.phase_semantics(output), expected)

    def test_qwen_tool_then_renewed_reasoning_preserves_order(self):
        self.serving.reasoning_parser = 'qwen3'
        self.serving.tool_call_parser = 'qwen3_coder'
        self.text = ('Checking.<tool_call><function=inspect></function></tool_call>'
                     '<think>Again</think>Final answer.')
        tools = [{'type': 'function', 'name': 'inspect',
                  'parameters': {'type': 'object', 'properties': {}}}]
        expected = [
            ('message', 'commentary', 'Checking.'),
            ('function_call', 'inspect', {}),
            ('reasoning', 'Again'),
            ('message', 'final_answer', 'Final answer.'),
        ]
        for stream in (False, True):
            with self.subTest(stream=stream):
                response = self.send(
                    stream=stream, tools=tools, tool_choice='auto',
                    reasoning={'effort': 'medium'})
                body = (next(event['response'] for event in self.events(response)
                             if event['type'] == 'response.completed')
                        if stream else response.json())
                self.assertEqual(self.phase_semantics(body['output']), expected)

    def test_qwen_second_reasoning_block_preserves_order(self):
        self.serving.reasoning_parser = 'qwen3'
        self.serving.tool_call_parser = None
        self.text = ('<think>First</think>Checking.'
                     '<think>Again</think>Final answer.')
        expected = [
            ('reasoning', 'First'),
            ('message', 'commentary', 'Checking.'),
            ('reasoning', 'Again'),
            ('message', 'final_answer', 'Final answer.'),
        ]
        for stream in (False, True):
            with self.subTest(stream=stream):
                response = self.send(
                    stream=stream, tools=[], tool_choice='none',
                    reasoning={'effort': 'medium'})
                body = (next(event['response'] for event in self.events(response)
                             if event['type'] == 'response.completed')
                        if stream else response.json())
                self.assertEqual(self.phase_semantics(body['output']), expected)

    def test_qwen_adjacent_reasoning_blocks_remain_distinct(self):
        self.serving.reasoning_parser = 'qwen3'
        self.serving.tool_call_parser = None
        self.text = '<think>First</think><think>Again</think>Final answer.'
        expected = [
            ('reasoning', 'First'),
            ('reasoning', 'Again'),
            ('message', 'final_answer', 'Final answer.'),
        ]
        for stream in (False, True):
            with self.subTest(stream=stream):
                response = self.send(
                    stream=stream, tools=[], tool_choice='none',
                    reasoning={'effort': 'medium'})
                body = (next(event['response'] for event in self.events(response)
                             if event['type'] == 'response.completed')
                        if stream else response.json())
                self.assertEqual(self.phase_semantics(body['output']), expected)

    def test_required_json_marker_like_values_remain_data(self):
        self.serving.tool_call_parser = None
        cases = [
            ([{'type': 'function', 'name': 'inspect', 'parameters': {
                'type': 'object', 'properties': {'text': {'type': 'string'}}}}],
             'inspect', {'text': '<think>literal</think>'}, 'function_call'),
            ([{'type': 'custom', 'name': 'patch'}],
             'patch', {'input': '</function></tool_call>tail'},
             'custom_tool_call'),
            (self.tools, 'workspace.read', {'path': '<tag>file</tag>'},
             'function_call'),
        ]
        for reasoning_parser in (None, 'qwen3'):
            self.serving.reasoning_parser = reasoning_parser
            for tools, generated_name, arguments, output_type in cases:
                with self.subTest(reasoning_parser=reasoning_parser,
                                  generated_name=generated_name):
                    self.text = json.dumps([
                        {'name': generated_name, 'parameters': arguments}
                    ])
                    response = self.send(
                        tools=tools, tool_choice='required',
                        reasoning={'effort': 'none'})
                    self.assertEqual(response.status_code, 200, response.text)
                    item = response.json()['output'][0]
                    self.assertEqual(item['type'], output_type)
                    if output_type == 'custom_tool_call':
                        self.assertEqual(item['input'], arguments['input'])
                    else:
                        self.assertEqual(json.loads(item['arguments']), arguments)

    def test_native_tool_payload_markers_after_reasoning_remain_data(self):
        self.serving.reasoning_parser = 'qwen3'
        self.serving.tool_call_parser = 'qwen3_coder'
        cases = [
            ([{'type': 'function', 'name': 'inspect', 'parameters': {
                'type': 'object', 'properties': {'text': {'type': 'string'}}}}],
             'inspect', 'text', '<think>literal</think>', 'function_call'),
            ([{'type': 'custom', 'name': 'patch'}],
             'patch', 'input', '<think>literal</think>', 'custom_tool_call'),
        ]
        for tools, name, parameter, value, output_type in cases:
            self.text = (
                '<think>Plan</think>Before'
                f'<tool_call><function={name}><parameter={parameter}>{value}'
                '</parameter></function></tool_call>After'
            )
            for stream in (False, True):
                with self.subTest(name=name, stream=stream):
                    response = self.send(
                        stream=stream, tools=tools, tool_choice='auto',
                        reasoning={'effort': 'medium'})
                    if stream:
                        events = self.events(response)
                        completed = [event['response'] for event in events
                                     if event['type'] == 'response.completed']
                        self.assertTrue(completed, events)
                        body = completed[0]
                    else:
                        body = response.json()
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(
                        [item['type'] for item in body['output']],
                        ['reasoning', 'message', output_type, 'message'],
                    )
                    call = body['output'][2]
                    if output_type == 'custom_tool_call':
                        self.assertEqual(call['input'], value)
                    else:
                        self.assertEqual(json.loads(call['arguments']),
                                         {parameter: value})

    def test_native_tool_implicitly_closes_open_reasoning(self):
        self.serving.reasoning_parser = 'qwen3'
        self.serving.tool_call_parser = 'qwen3_coder'
        cases = [
            ([{'type': 'function', 'name': 'inspect',
               'parameters': {'type': 'object', 'properties': {}}}],
             'inspect', '', 'function_call'),
            ([{'type': 'custom', 'name': 'patch'}],
             'patch', '<parameter=input></parameter>', 'custom_tool_call'),
        ]
        for tools, name, parameters, output_type in cases:
            self.text = (f'<think>Plan<tool_call><function={name}>{parameters}</function>'
                         '</tool_call>After')
            for stream in (False, True):
                with self.subTest(name=name, stream=stream):
                    response = self.send(
                        stream=stream, tools=tools, tool_choice='auto',
                        reasoning={'effort': 'medium'})
                    if stream:
                        events = self.events(response)
                        completed = [event['response'] for event in events
                                     if event['type'] == 'response.completed']
                        self.assertTrue(completed, events)
                        body = completed[0]
                    else:
                        self.assertEqual(response.status_code, 200, response.text)
                        body = response.json()
                    self.assertEqual(
                        [item['type'] for item in body['output']],
                        ['reasoning', output_type, 'message'],
                    )
                    self.assertEqual(self.phase_semantics(
                        [body['output'][0], body['output'][2]]), [
                            ('reasoning', 'Plan'),
                            ('message', 'final_answer', 'After'),
                        ])

    def test_qwen4_stream_reasoning_is_chunking_negative_control(self):
        self.serving.tokenizer_manager.model_config.hf_config.model_type = 'qwen4_exp'
        self.serving.reasoning_parser = 'qwen3'
        self.serving.tool_call_parser = None
        raw = '<think>First</think>Checking.<think>Second</think>Final'
        expected = [
            ('reasoning', 'First'),
            ('message', 'final_answer',
             'Checking.<think>Second</think>Final'),
        ]
        for parts in ([raw], ['<think>', 'First', '</think>', 'Checking.',
                              '<think>', 'Second', '</think>', 'Final']):
            with self.subTest(parts=parts):
                async def generate(request, *args, **kwargs):
                    cumulative = ''
                    for index, part in enumerate(parts):
                        cumulative += part
                        yield {'text': cumulative, 'output_ids': [1] * (index + 1),
                               'meta_info': {'prompt_tokens': 10,
                                             'completion_tokens': index + 1,
                                             'finish_reason': ({'type': 'stop'}
                                                               if index == len(parts) - 1
                                                               else None)}}

                self.serving.tokenizer_manager.generate_request = generate
                events = self.events(self.send(
                    stream=True, tools=[], tool_choice='none',
                    reasoning={'effort': 'medium'}))
                output = next(event['response']['output'] for event in events
                              if event['type'] == 'response.completed')
                self.assertEqual(self.phase_semantics(output), expected)

    def test_text_streams_before_phase_is_resolved(self):
        async def check():
            self.serving.reasoning_parser = None
            self.serving.tool_call_parser = 'qwen3_coder'
            request = ResponsesRequest(
                model='fixture-qwen', input='Tell a story', stream=True,
                tools=[{'type': 'function', 'name': 'inspect',
                        'parameters': {'type': 'object'}}], tool_choice='auto',
            )
            release = asyncio.Event()

            async def generate():
                yield {'text': 'Once upon a time', 'meta_info': {
                    'prompt_tokens': 10, 'completion_tokens': 4,
                    'finish_reason': None}}
                await release.wait()
                yield {'text': 'Once upon a time. The end.', 'meta_info': {
                    'prompt_tokens': 10, 'completion_tokens': 8,
                    'finish_reason': {'type': 'stop'}}}

            stream = self.serving.responses_stream_generator_non_harmony(
                request, {}, generate(), 'fixture-qwen', Mock(),
                RequestResponseMetadata(request_id=request.request_id),
                require_reasoning=False,
            )
            events = []
            try:
                async def first_text():
                    async for frame in stream:
                        event = json.loads(frame.split('data: ', 1)[1])
                        events.append(event)
                        if event['type'] == 'response.output_text.delta':
                            return event['delta']
                    self.fail('stream ended before emitting text')

                self.assertEqual(
                    await asyncio.wait_for(first_text(), timeout=1),
                    'Once upon a time',
                )
                added = next(event['item'] for event in events
                             if event['type'] == 'response.output_item.added')
                self.assertIsNone(added.get('phase'))
                release.set()
                async for frame in stream:
                    events.append(json.loads(frame.split('data: ', 1)[1]))
                done = next(event['item'] for event in events
                            if event['type'] == 'response.output_item.done')
                self.assertEqual(done['phase'], 'final_answer')
            finally:
                release.set()
                await stream.aclose()

        asyncio.run(check())

    def test_fix2_embedded_identity_rejection(self):
        for stream in (False, True):
            for embedded in (False, True):
                with self.subTest(stream=stream, embedded=embedded):
                    call = {'type': 'function_call', 'name': 'workspace.read',
                            'call_id': 'call_history', 'arguments': '{}'}
                    if embedded:
                        call = {'role': 'assistant', 'content': '', 'tool_calls': [
                            {'id': 'call_history', 'type': 'function',
                             'function': {'name': 'workspace.read', 'arguments': '{}'}}]}
                    before = len(self.generated)
                    response = self.send(stream=stream, tool_choice='none', input=[
                        {'role': 'user', 'content': 'Read'}, call,
                        {'type': 'function_call_output', 'call_id': 'call_history', 'output': 'ok'}])
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertEqual(len(self.generated), before)

    def test_fix2_embedded_flat_and_history(self):
        self.text = 'Done'
        for stream in (False, True):
            with self.subTest(stream=stream):
                response = self.send(stream=stream, tools=[], tool_choice='none', input=[
                    {'role': 'user', 'content': 'Read'},
                    {'role': 'assistant', 'content': '', 'tool_calls': [
                        {'id': 'call_history', 'type': 'function',
                         'function': {'name': 'workspace.read', 'arguments': '{}'}}]},
                    {'type': 'function_call_output', 'call_id': 'call_history', 'output': 'ok'}])
                body = next(event['response'] for event in self.events(response)
                            if event.get('type') == 'response.completed') if stream else response.json()
                self.assertEqual(body['status'], 'completed', body)
                gap = self.send(previous_response_id=body['id'], tools=[], tool_choice='none').json()
                before = len(self.generated)
                rejected = self.send(previous_response_id=gap['id'], tool_choice='none')
                self.assertEqual(rejected.status_code, 400, rejected.text)
                self.assertEqual(len(self.generated), before)

    def test_fix2_terminal_cardinality(self):
        for parser in (None, 'qwen3_coder'):
            for kind in ('function', 'custom'):
                for choice in ('required', {'type': kind, 'name': 'read', 'namespace': 'workspace'}):
                    for stream in (False, True):
                        for text in ('', '<think>Only reasoning</think>'):
                            for finish in ('stop', 'length', 'abort'):
                                with self.subTest(parser=parser, kind=kind, choice=choice,
                                                  stream=stream, text=text, finish=finish):
                                    self.serving.tool_call_parser = parser
                                    self.serving.reasoning_parser = 'qwen3'
                                    async def generate(request, *args, **kwargs):
                                        self.generated.append(request)
                                        yield {'text': text, 'output_ids': [], 'meta_info': {
                                            'prompt_tokens': 10, 'completion_tokens': 0,
                                            'finish_reason': {'type': finish}}}
                                    self.serving.tokenizer_manager.generate_request = generate
                                    existing = set(self.serving.response_store)
                                    response = self.send(stream=stream, tool_choice=choice, tools=[
                                        {'type': 'namespace', 'name': 'workspace', 'tools': [
                                            {'type': kind, 'name': 'read'}]}])
                                    if stream:
                                        events = self.events(response)
                                        self.assertFalse(any(event.get('type') == 'response.completed'
                                                             for event in events), events)
                                        if finish == 'stop':
                                            self.assertTrue(any(event.get('type') == 'error' for event in events), events)
                                        else:
                                            terminal = next(event['response'] for event in events
                                                            if event.get('type') in ('response.incomplete', 'response.failed'))
                                            self.assertEqual(terminal['status'], 'incomplete' if finish == 'length' else 'cancelled')
                                    elif finish == 'stop':
                                        self.assertEqual(response.status_code, 400, response.text)
                                    else:
                                        self.assertEqual(response.status_code, 200, response.text)
                                        self.assertEqual(response.json()['status'], 'incomplete' if finish == 'length' else 'cancelled')
                                    for response_id in set(self.serving.response_store) - existing:
                                        self.assertNotEqual(self.serving.response_store[response_id].status, 'completed')

    def test_fix3_partial_terminal_matrix(self):
        for parser in (None, 'qwen3_coder'):
            for namespace in (None, 'workspace'):
                qualified = 'workspace.read' if namespace else 'read'
                texts = ([f'[{json.dumps({"name": qualified})[:-1]},"parameters":{suffix}'
                          for suffix in ('{"input":"partial', '{"input":"ok"', '{"input":"ok"}}', '{"input":["partial')]
                         if parser is None else
                         [f'<tool_call><function={qualified}><parameter=input>partial',
                          f'<tool_call><function={qualified}><parameter=input>partial</parameter>',
                          f'<tool_call><function={qualified}><parameter=input>partial</parameter></function>'])
                for kind in ('function', 'custom'):
                    selected = {'type': kind, 'name': 'read'}
                    if namespace:
                        selected['namespace'] = namespace
                    tools = [{'type': kind, 'name': 'read'}]
                    if namespace:
                        tools = [{'type': 'namespace', 'name': namespace, 'tools': tools}]
                    for choice in ('required', selected):
                        for stream, incremental in ((False, False), (True, False), (True, True)):
                            for finish in ('length', 'abort'):
                                for text in texts:
                                    with self.subTest(parser=parser, namespace=namespace, kind=kind,
                                                      choice=choice, stream=stream, finish=finish, text=text):
                                        self.serving.tool_call_parser = parser
                                        self.serving.tokenizer_manager.server_args.incremental_streaming_output = incremental
                                        async def generate(request, *args, **kwargs):
                                            sizes = range(1, len(text) + 1) if request.stream else [len(text)]
                                            for size in sizes:
                                                yield {'text': text[size - 1:size] if incremental else text[:size], 'output_ids': [1] * size,
                                                       'meta_info': {'prompt_tokens': 10, 'completion_tokens': size,
                                                                     'reasoning_tokens': 2,
                                                                     'finish_reason': {'type': finish} if size == len(text) and not incremental else None}}
                                            if incremental:
                                                yield {'text': '', 'meta_info': {'finish_reason': {'type': finish}}}
                                        self.serving.tokenizer_manager.generate_request = generate
                                        response = self.send(stream=stream, tools=tools, tool_choice=choice,
                                                             metadata={'fixture': 'fix3'})
                                        self.assertEqual(response.status_code, 200, response.text)
                                        if stream:
                                            events = self.events(response)
                                            expected = 'response.incomplete' if finish == 'length' else 'response.failed'
                                            terminals = [event['response'] for event in events if event.get('type') == expected]
                                            self.assertEqual(len(terminals), 1, events)
                                            body = terminals[0]
                                            self.assertFalse(any(event.get('type') in (
                                                'error', 'response.completed', 'response.function_call_arguments.done',
                                                'response.custom_tool_call_input.done') for event in events), events)
                                            self.assertFalse(any(event.get('item', {}).get('type') in (
                                                'function_call', 'custom_tool_call') and event['item'].get('status') == 'completed'
                                                for event in events), events)
                                        else:
                                            body = response.json()
                                        self.assertEqual(body['status'], 'incomplete' if finish == 'length' else 'cancelled')
                                        self.assertEqual(body['metadata'], {'fixture': 'fix3'})
                                        self.assertEqual(body['usage']['input_tokens'], 10)
                                        self.assertEqual(body['usage']['output_tokens'], len(text))
                                        self.assertEqual(body['usage']['output_tokens_details']['reasoning_tokens'], 2)
                                        self.assertEqual(body['incomplete_details'],
                                                         {'reason': 'max_output_tokens'} if finish == 'length' else None)
                                        self.assertFalse(any(item['type'] in ('function_call', 'custom_tool_call') for item in body['output']))
                                        stored = self.client.get('/v1/responses/' + body['id'])
                                        self.assertEqual(stored.status_code, 200, stored.text)
                                        for key in ('id', 'status', 'output', 'usage', 'metadata', 'incomplete_details', 'reasoning'):
                                            self.assertEqual(stored.json()[key], body[key])
                                        async def replay_generate(request, *args, **kwargs):
                                            self.generated.append(request)
                                            yield {'text': 'Done', 'meta_info': {'finish_reason': {'type': 'stop'}}}
                                        self.serving.tokenizer_manager.generate_request = replay_generate
                                        replay = self.send(previous_response_id=body['id'], tools=tools, tool_choice='none')
                                        self.assertEqual(replay.status_code, 200, replay.text)
                                        self.assertEqual(replay.json()['status'], 'completed')
                                        self.assertNotIn('partial', self.tokenizer.decode(self.generated[-1].input_ids))

    def test_fix3_partial_no_store(self):
        for stream in (False, True):
            for finish in ('length', 'abort'):
                with self.subTest(stream=stream, finish=finish):
                    async def generate(request, *args, **kwargs):
                        yield {'text': '[{"name":"workspace.read","parameters":{"input":"partial',
                               'meta_info': {'prompt_tokens': 10, 'completion_tokens': 3,
                                             'finish_reason': {'type': finish}}}
                    self.serving.tokenizer_manager.generate_request = generate
                    response = self.send(stream=stream, store=False)
                    body = ([event['response'] for event in self.events(response)
                             if event.get('type') in ('response.incomplete', 'response.failed')][0]
                            if stream else response.json())
                    self.assertEqual(body['status'], 'incomplete' if finish == 'length' else 'cancelled')
                    self.assertEqual(body['output'], [])
                    self.assertEqual(self.client.get('/v1/responses/' + body['id']).status_code, 404)

    def test_fix3_terminal_text_and_no_store(self):
        for stream in (False, True):
            for finish in ('length', 'abort'):
                with self.subTest(stream=stream, finish=finish):
                    async def generate(request, *args, **kwargs):
                        yield {'text': 'partial text', 'meta_info': {
                            'prompt_tokens': 10, 'completion_tokens': 3, 'finish_reason': {'type': finish}}}
                    self.serving.tokenizer_manager.generate_request = generate
                    response = self.send(stream=stream, tool_choice='none', store=False)
                    body = ([event['response'] for event in self.events(response)
                             if event.get('type') in ('response.incomplete', 'response.failed')][0]
                            if stream else response.json())
                    self.assertEqual(body['output'][0]['content'][0]['text'], 'partial text')
                    self.assertEqual(self.client.get('/v1/responses/' + body['id']).status_code, 404)

    def test_fix3_malformed_success_remains_rejected(self):
        for stream in (False, True):
            for kind in ('function', 'custom'):
                with self.subTest(stream=stream, kind=kind):
                    self.text = '[{"name":"read","parameters":{"input":"partial'
                    response = self.send(stream=stream, tools=[{'type': kind, 'name': 'read'}])
                    if stream:
                        events = self.events(response)
                        self.assertFalse(any(event.get('type') == 'response.completed' for event in events))
                        self.assertTrue(any(event.get('type') in ('error', 'response.failed') for event in events))
                    else:
                        self.assertEqual(response.status_code, 400, response.text)

    def test_fix3_native_auto_terminal_text(self):
        self.serving.tool_call_parser = 'qwen3_coder'
        for stream in (False, True):
            for finish in ('length', 'abort'):
                with self.subTest(stream=stream, finish=finish):
                    async def generate(request, *args, **kwargs):
                        yield {'text': 'Useful partial text', 'meta_info': {
                            'prompt_tokens': 10, 'completion_tokens': 3, 'finish_reason': {'type': finish}}}
                    self.serving.tokenizer_manager.generate_request = generate
                    response = self.send(stream=stream, tool_choice='auto')
                    body = ([event['response'] for event in self.events(response)
                             if event.get('type') in ('response.incomplete', 'response.failed')][0]
                            if stream else response.json())
                    self.assertTrue(body['output'], body)
                    self.assertEqual(body['output'][0]['content'][0]['text'], 'Useful partial text')

    def test_fix2_embedded_supported_and_custom_distinctions(self):
        self.text = 'Done'
        for stream in (False, True):
            for namespace in (None, 'workspace'):
                for declarations in (False, True):
                    with self.subTest(stream=stream, namespace=namespace, declarations=declarations):
                        function = {'name': 'read', 'arguments': '{}'}
                        if namespace:
                            function['namespace'] = namespace
                        tools = [{'type': 'function', 'name': 'read'}]
                        if namespace:
                            tools = [{'type': 'namespace', 'name': namespace, 'tools': tools}]
                        response = self.send(stream=stream, tools=tools if declarations else [], tool_choice='none', input=[
                            {'role': 'user', 'content': 'Read'},
                            {'role': 'assistant', 'content': '', 'tool_calls': [
                                {'id': 'call_history', 'type': 'function', 'function': function}]},
                            {'type': 'function_call_output', 'call_id': 'call_history', 'output': 'ok'}])
                        body = next(event['response'] for event in self.events(response)
                                    if event.get('type') == 'response.completed') if stream else response.json()
                        self.assertEqual(body['status'], 'completed', body)
                        prompt = self.tokenizer.decode(self.generated[-1].input_ids)
                        self.assertIn('workspace.read' if namespace else 'read', prompt)
                        gap = self.send(previous_response_id=body['id'], tools=[], tool_choice='none').json()
                        continued = self.send(previous_response_id=gap['id'], tools=tools, tool_choice='none')
                        self.assertEqual(continued.status_code, 200, continued.text)
                        custom = [{'type': 'custom', 'name': 'read'}]
                        if namespace:
                            custom = [{'type': 'namespace', 'name': namespace, 'tools': custom}]
                        before = len(self.generated)
                        rejected = self.send(previous_response_id=gap['id'], tools=custom, tool_choice='none')
                        self.assertEqual(rejected.status_code, 400, rejected.text)
                        self.assertEqual(len(self.generated), before)
                        rejected = self.send(tools=custom, tool_choice='none', stream=stream, input=[
                            {'role': 'user', 'content': 'Read'},
                            {'role': 'assistant', 'tool_calls': [
                                {'id': 'call_history', 'type': 'function', 'function': function}]}])
                        self.assertEqual(rejected.status_code, 400, rejected.text)
                        self.assertEqual(len(self.generated), before)

    def test_fix2_unsupported_embedded_forms(self):
        cases = [
            {'role': 'assistant', 'function_call': {'name': 'read', 'arguments': '{}'}},
            {'role': 'user', 'tool_calls': []},
            {'role': 'assistant', 'tool_calls': {}},
            {'role': 'assistant', 'tool_calls': [{'id': 'call_history', 'type': 'custom',
                                               'custom': {'name': 'read', 'input': 'raw'}}]},
            {'role': 'assistant', 'tool_calls': [{'id': 'call_history', 'type': 'function',
                'namespace': 'workspace', 'function': {'name': 'read', 'arguments': '{}'}}]},
            {'role': 'assistant', 'tool_calls': [{'id': 'call_history', 'type': 'function',
                'function': {'name': 'read', 'arguments': {}}}]},
        ]
        for stream in (False, True):
            for item in cases:
                with self.subTest(stream=stream, item=item):
                    before = len(self.generated)
                    response = self.send(stream=stream, input=[{'role': 'user', 'content': 'Read'}, item])
                    self.assertIn(response.status_code, (400, 422), response.text)
                    self.assertEqual(len(self.generated), before)

    def test_fix2_selected_single_required_multiple(self):
        for kind in ('function', 'custom'):
            for parser in (None, 'qwen3_coder'):
                for stream in (False, True):
                    with self.subTest(kind=kind, parser=parser, stream=stream):
                        self.serving.tool_call_parser = parser
                        self.tools = [{'type': kind, 'name': 'read'}]
                        parameters = {'input': 'raw'} if kind == 'custom' else {'path': 'file'}
                        if parser:
                            parameter = 'input' if kind == 'custom' else 'path'
                            self.text = ('<tool_call><function=read><parameter=' + parameter +
                                         '>raw</parameter></function></tool_call>') * 2
                        else:
                            self.text = json.dumps([{'name': 'read', 'parameters': parameters}] * 2)
                        existing = set(self.serving.response_store)
                        rejected = self.send(tool_choice={'type': kind, 'name': 'read'}, stream=stream)
                        if stream:
                            events = self.events(rejected)
                            self.assertTrue(any(event.get('type') == 'error' for event in events), events)
                            self.assertFalse(any(event.get('type') == 'response.completed' for event in events), events)
                        else:
                            self.assertEqual(rejected.status_code, 400, rejected.text)
                        self.assertFalse(set(self.serving.response_store) - existing)
                        accepted = self.send(tool_choice='required', stream=stream)
                        body = next(event['response'] for event in self.events(accepted)
                                    if event.get('type') == 'response.completed') if stream else accepted.json()
                        self.assertEqual(body['status'], 'completed', body)
                        self.assertEqual(len(body['output']), 2)

    def test_fix2_custom_delimiter_limit_and_json_alternative(self):
        self.tools = [{'type': 'custom', 'name': 'patch'}]
        for delimiter in ('</parameter>', '</function>', '</tool_call>', '<parameter=other>'):
            raw = 'before' + delimiter + 'after'
            for stream in (False, True):
                with self.subTest(delimiter=delimiter, stream=stream):
                    self.serving.tool_call_parser = 'qwen3_coder'
                    self.text = '<tool_call><function=patch><parameter=input>' + raw + '</parameter></function></tool_call>'
                    response = self.send(tool_choice='auto', stream=stream)
                    if stream:
                        events = self.events(response)
                        completed = [event['response'] for event in events if event.get('type') == 'response.completed']
                        if not completed:
                            self.assertTrue(any(event.get('type') in ('error', 'response.failed') for event in events), events)
                        body = completed[0] if completed else None
                    else:
                        self.assertIn(response.status_code, (200, 400), response.text)
                        body = response.json() if response.status_code == 200 else None
                    if body:
                        calls = [item for item in body['output'] if item['type'] == 'custom_tool_call']
                        self.assertEqual(len(calls), 1, body)
                        expected = raw if stream and delimiter == '</tool_call>' else 'before'
                        self.assertEqual(calls[0]['input'], expected)
                    self.serving.tool_call_parser = None
                    self.text = json.dumps([{'name': 'patch', 'parameters': {'input': raw}}])
                    response = self.send(tool_choice='required', stream=stream)
                    body = next(event['response'] for event in self.events(response)
                                if event.get('type') == 'response.completed') if stream else response.json()
                    self.assertEqual(body['output'][0]['input'], raw)

    def test_fix2_custom_history_rejects_embedded_function_after_gap(self):
        self.text = 'Done'
        for stream in (False, True):
            with self.subTest(stream=stream):
                prior = self.send(tools=[], tool_choice='none', input=[
                    {'role': 'user', 'content': 'Patch'},
                    {'type': 'custom_tool_call', 'name': 'patch', 'call_id': 'call_custom', 'input': 'raw'},
                    {'type': 'custom_tool_call_output', 'call_id': 'call_custom', 'output': 'ok'}]).json()
                gap = self.send(previous_response_id=prior['id'], tools=[], tool_choice='none').json()
                before = len(self.generated)
                response = self.send(previous_response_id=gap['id'], tools=[], stream=stream, tool_choice='none', input=[
                    {'role': 'user', 'content': 'Patch'},
                    {'role': 'assistant', 'tool_calls': [{'id': 'call_new', 'type': 'function',
                        'function': {'name': 'patch', 'arguments': '{"input":"raw"}'}}]}])
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(len(self.generated), before)

    def test_background_requires_storage(self):
        for store in (False, None):
            with self.subTest(store=store):
                response = self.send(background=True, store=store)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(self.serving.response_store, {})
                self.assertEqual(self.serving._compat_registries, {})
                self.assertEqual(self.generated, [])

    def test_fixer_conflicting_forced_representations_before_generation(self):
        tools = [{'type': 'function', 'name': name} for name in ('read', 'write')]
        tools += [{'type': 'namespace', 'name': namespace, 'tools': [
            {'type': 'function', 'name': 'read'}]} for namespace in ('first', 'second')]
        choices = [
            {'name': 'read', 'function': {'name': 'write'}},
            {'name': 'read', 'namespace': 'first', 'function': {'name': 'read', 'namespace': 'second'}},
            {'name': [], 'function': {'name': 'write'}},
            {'namespace': [], 'function': {'name': 'read', 'namespace': 'second'}},
            {'name': 'read', 'function': None},
            {'name': 'read', 'function': []},
            {'name': 'read', 'function': {'namespace': 'first'}},
        ]
        for stream in (False, True):
            for choice in choices:
                with self.subTest(stream=stream, choice=choice):
                    before = len(self.generated)
                    response = self.send(tools=tools, stream=stream, tool_choice={'type': 'function', **choice})
                    self.assertIn(response.status_code, (400, 422), response.text)
                    self.assertEqual(len(self.generated), before)

    def test_fixer_descriptions_reach_rendered_prompt(self):
        self.text = 'Done'
        for kind in ('function', 'custom'):
            with self.subTest(kind=kind):
                member = {'type': kind, 'name': 'read', 'description': 'MEMBER instructions'}
                if kind == 'custom':
                    member['format'] = {'type': 'grammar', 'syntax': 'lark', 'definition': 'start: "GRAMMAR"'}
                response = self.send(tools=[{'type': 'namespace', 'name': 'workspace',
                    'description': 'NAMESPACE instructions', 'tools': [member]}], tool_choice='auto')
                self.assertEqual(response.status_code, 200, response.text)
                prompt = self.tokenizer.decode(self.generated[-1].input_ids)
                for expected in ('NAMESPACE instructions', 'MEMBER instructions') + (
                        ('lark', 'GRAMMAR', 'not enforced') if kind == 'custom' else ()):
                    with self.subTest(expected=expected):
                        self.assertIn(expected, prompt)

    def test_fixer_whole_history_collision_parity(self):
        cases = [
            ({'type': 'function_call', 'name': 'read', 'namespace': 'workspace', 'arguments': '{}'},
             {'type': 'function', 'name': 'workspace.read'}),
            ({'type': 'function_call', 'name': 'b.read', 'namespace': 'a', 'arguments': '{}'},
             {'type': 'namespace', 'name': 'a.b', 'tools': [{'type': 'function', 'name': 'read'}]}),
            ({'type': 'custom_tool_call', 'name': 'read', 'namespace': 'workspace', 'input': ' \nraw\n '},
             {'type': 'namespace', 'name': 'workspace', 'tools': [{'type': 'function', 'name': 'read'}]}),
            ({'type': 'function_call', 'name': 'workspace.read', 'arguments': '{}'},
             {'type': 'namespace', 'name': 'workspace', 'tools': [{'type': 'function', 'name': 'read'}]}),
            ({'type': 'function_call', 'name': 'read', 'namespace': 'a.b', 'arguments': '{}'},
             {'type': 'namespace', 'name': 'a', 'tools': [{'type': 'function', 'name': 'b.read'}]}),
            ({'type': 'function_call', 'name': 'read', 'namespace': 'workspace', 'arguments': '{}'},
             {'type': 'namespace', 'name': 'workspace', 'tools': [{'type': 'custom', 'name': 'read'}]}),
        ]
        self.text = 'Remembered'
        for historical, conflicting in cases:
            for stream in (False, True):
                with self.subTest(historical=historical, stream=stream):
                    call = dict(historical, call_id='call_history')
                    output = {'type': historical['type'] + '_output', 'call_id': call['call_id'], 'output': 'RESULT'}
                    replay = [{'role': 'user', 'content': 'remember'}, call, output]
                    response = self.send(tools=[], tool_choice='none', input=replay)
                    self.assertEqual(response.status_code, 200, response.text)
                    first = response.json()
                    second = self.send(tools=[], tool_choice='none', previous_response_id=first['id'], input='gap one').json()
                    third = self.send(tools=[], tool_choice='none', previous_response_id=second['id'], input='gap two').json()
                    for arguments in ({'input': replay}, {'input': 'continue', 'previous_response_id': third['id']}):
                        with self.subTest(stateful='previous_response_id' in arguments):
                            before = len(self.generated)
                            response = self.send(tools=[conflicting], tool_choice='auto', stream=stream, **arguments)
                            self.assertEqual(response.status_code, 400, response.text)
                            self.assertEqual(len(self.generated), before)
                    if conflicting['type'] == 'namespace':
                        new_member = conflicting['tools'][0]
                        new_call = {'type': 'custom_tool_call' if new_member['type'] == 'custom' else 'function_call',
                                    'name': new_member['name'], 'namespace': conflicting['name']}
                    else:
                        new_call = {'type': 'function_call', 'name': conflicting['name']}
                    new_call.update(call_id='call_new', arguments='{}')
                    if new_call['type'] == 'custom_tool_call':
                        new_call.pop('arguments')
                        new_call['input'] = ''
                    before = len(self.generated)
                    response = self.send(tools=[], tool_choice='none', stream=stream,
                        previous_response_id=third['id'], input=[new_call])
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertEqual(len(self.generated), before)

    def test_fixer_generated_history_survives_output_only_and_multiple_turns(self):
        first = self.send().json()
        first_call = first['output'][0]
        raw = ' \nsecond generation raw\n '
        self.text = json.dumps([{'name': 'editor.patch', 'parameters': {'input': raw}}])
        second = self.send(tools=[{'type': 'namespace', 'name': 'editor', 'tools': [
            {'type': 'custom', 'name': 'patch'}]}], previous_response_id=first['id'], input=[{
            'type': 'function_call_output', 'call_id': first_call['call_id'], 'output': 'FIRST RESULT'}]).json()
        second_call = second['output'][0]
        self.assertEqual(second_call['input'], raw)
        self.text = 'Done'
        third = self.send(tools=[], tool_choice='none', previous_response_id=second['id'], input=[{
            'type': 'custom_tool_call_output', 'call_id': second_call['call_id'], 'output': 'SECOND RESULT'}]).json()
        fourth = self.send(tools=[], tool_choice='none', previous_response_id=third['id'], input='gap').json()
        messages = self.serving.msg_store[fourth['id']]
        calls = [call for message in messages for call in message.get('tool_calls', [])]
        self.assertEqual([(call['id'], call['function']['name']) for call in calls], [
            (first_call['call_id'], 'workspace.read'), (second_call['call_id'], 'editor.patch')])
        self.assertEqual(json.loads(calls[1]['function']['arguments'])['input'], raw)
        for body in (first, second):
            self.assertEqual(self.client.get('/v1/responses/' + body['id']).json()['output'], body['output'])
        before = len(self.generated)
        rejected = self.send(tools=[{'type': 'function', 'name': 'workspace.read'}], tool_choice='auto',
            previous_response_id=fourth['id'])
        self.assertEqual(rejected.status_code, 400, rejected.text)
        self.assertEqual(len(self.generated), before)
        self.text = '[{"name":"workspace.read","parameters":{}}]'
        for stream in (False, True):
            response = self.send(tools=[{'type': 'function', 'name': 'other'}], stream=stream,
                previous_response_id=fourth['id'])
            if stream:
                self.assertFalse(any(event['type'] == 'response.completed' for event in self.events(response)))
            else:
                self.assertEqual(response.status_code, 400, response.text)

    def test_fixer_pinned_sdk_output_and_event_roundtrip_replay(self):
        import openai
        from openai.types.responses import ResponseFunctionToolCall, ResponseCustomToolCall, ResponseStreamEvent
        from openai.types.responses.response_output_item import ResponseOutputItem
        from pydantic import TypeAdapter

        self.assertEqual(openai.__version__, '2.6.1')
        self.assertNotIn('namespace', ResponseFunctionToolCall.model_fields)
        self.assertNotIn('namespace', ResponseCustomToolCall.model_fields)
        output_adapter = TypeAdapter(ResponseOutputItem)
        event_adapter = TypeAdapter(ResponseStreamEvent)
        for kind, raw in (('function', '{}'), ('custom', ''), ('custom', ' \nraw\n ')):
            for stream in (False, True):
                with self.subTest(kind=kind, raw=raw, stream=stream):
                    tools = [{'type': 'namespace', 'name': 'sdk', 'tools': [{'type': kind, 'name': 'call'}]}]
                    parameters = {'input': raw} if kind == 'custom' else {}
                    self.text = json.dumps([{'name': 'sdk.call', 'parameters': parameters}])
                    response = self.send(tools=tools, stream=stream)
                    if stream:
                        events = self.events(response)
                        body = next(event['response'] for event in events if event['type'] == 'response.completed')
                        for event in events:
                            if 'response' in event:
                                continue
                            parsed = event_adapter.validate_python(event)
                            wire = json.loads(parsed.model_dump_json(exclude_unset=True))
                            for field in ('type', 'item_id', 'name', 'namespace', 'delta', 'input', 'arguments'):
                                if field in event:
                                    self.assertEqual(wire[field], event[field])
                            if 'item' in event:
                                for field in ('call_id', 'name', 'namespace', 'input', 'arguments'):
                                    if field in event['item']:
                                        self.assertEqual(wire['item'][field], event['item'][field])
                    else:
                        self.assertEqual(response.status_code, 200, response.text)
                        body = response.json()
                    call = body['output'][0]
                    parsed = output_adapter.validate_python(call)
                    replay = json.loads(parsed.model_dump_json(exclude_unset=True))
                    self.assertEqual(replay, call)
                    self.assertEqual(replay['namespace'], 'sdk')
                    if kind == 'custom':
                        self.assertEqual(replay['input'], raw)
                    self.text = 'Replayed'
                    result = {'type': call['type'] + '_output', 'call_id': call['call_id'], 'output': 'SDK result'}
                    for arguments in ({'input': [{'role': 'user', 'content': 'replay'}, replay, result]},
                                      {'previous_response_id': body['id'], 'input': [result]}):
                        continued = self.send(tools=[], tool_choice='none', **arguments)
                        self.assertEqual(continued.status_code, 200, continued.text)
                        stored = self.serving.msg_store[continued.json()['id']]
                        calls = [item for message in stored for item in message.get('tool_calls', [])]
                        self.assertEqual(calls[-1]['id'], call['call_id'])
                        self.assertEqual(calls[-1]['function']['name'], 'sdk.call')
                        if kind == 'custom':
                            self.assertEqual(json.loads(calls[-1]['function']['arguments'])['input'], raw)

    def test_fixer_pinned_sdk_client_terminal_roundtrip(self):
        from openai import OpenAI
        from openai.types.responses import Response
        from pydantic import ValidationError

        sdk = OpenAI(api_key='mock-cpu-not-a-credential', base_url='http://testserver/v1', http_client=self.client)
        for kind in ('function', 'custom'):
            for stream in (False, True):
                with self.subTest(kind=kind, stream=stream):
                    raw = ' \nSDK terminal\n '
                    parameters = {'input': raw} if kind == 'custom' else {}
                    self.text = json.dumps([{'name': 'sdk.call', 'parameters': parameters}])
                    tools = [{'type': 'namespace', 'name': 'sdk', 'tools': [{'type': kind, 'name': 'call'}]}]
                    response = sdk.responses.create(model='fixture-qwen', input='Read', stream=stream,
                        extra_body={'tools': tools, 'tool_choice': 'required', 'max_output_tokens': 128})
                    if stream:
                        events = list(response)
                        response = next(event.response for event in events if event.type == 'response.completed')
                        done = next(event.item for event in events if event.type == 'response.output_item.done')
                        self.assertEqual(done.call_id, response.output[0].call_id)
                        self.assertEqual(done.namespace, 'sdk')
                    serialized = json.loads(response.to_json())
                    call = serialized['output'][0]
                    self.assertEqual((call['name'], call['namespace']), ('call', 'sdk'))
                    self.assertEqual(call['type'], 'custom_tool_call' if kind == 'custom' else 'function_call')
                    if kind == 'custom':
                        self.assertEqual(call['input'], raw)
                    with self.assertRaises(ValidationError) as rejected:
                        Response.model_validate(serialized)
                    self.assertTrue(all(error['loc'][0] == 'tools' for error in rejected.exception.errors()))
                    Response.model_validate({**serialized, 'tools': []})
                    self.text = 'SDK replay complete'
                    continued = sdk.responses.create(model='fixture-qwen', input=[
                        {'role': 'user', 'content': 'Replay'}, call,
                        {'type': call['type'] + '_output', 'call_id': call['call_id'], 'output': 'Result'}],
                        extra_body={'tools': [], 'tool_choice': 'none'})
                    self.assertEqual(continued.status, 'completed')

    def test_fixer_image_history_survives_no_declaration_gaps(self):
        image = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a/RsAAAAASUVORK5CYII='
        self.serving.tokenizer_manager.model_config.is_multimodal = True
        self.serving.template_manager.jinja_template_content_format = 'openai'
        self.text = 'Seen'
        response = self.send(tools=[], tool_choice='none', input=[
            {'role': 'user', 'content': 'Inspect'},
            {'type': 'custom_tool_call', 'name': 'patch', 'namespace': 'editor', 'call_id': 'image_call', 'input': ' \nraw\n '},
            {'type': 'custom_tool_call_output', 'call_id': 'image_call', 'output': [
                {'type': 'input_text', 'text': 'IMAGE HISTORY'}, {'type': 'input_image', 'image_url': image}]}])
        self.assertEqual(response.status_code, 200, response.text)
        for gap in ('first gap', 'second gap'):
            response = self.send(tools=[], tool_choice='none', previous_response_id=response.json()['id'], input=gap)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(self.generated[-1].image_data)
            self.assertIn('IMAGE HISTORY', self.generated[-1].text)
            stored = self.serving.msg_store[response.json()['id']]
            output = next(message for message in stored if message.get('tool_call_id') == 'image_call')
            self.assertEqual(output['content'][1]['image_url']['url'], image)

    def test_failed_and_disconnected_stream_preserves_stored_identity(self):
        import asyncio
        response_id = 'resp_stream_reuse_cpu'
        original = self.send(request_id=response_id).json()
        original_messages = copy.deepcopy(self.serving.msg_store[response_id])
        alternate = [{'type': 'namespace', 'name': 'other', 'tools': [
            {'type': 'function', 'name': 'read', 'parameters': {'type': 'object'}}]}]
        self.text = '[{"name":"undeclared","parameters":{}}]'
        events = self.events(self.send(request_id=response_id, tools=alternate, stream=True))
        self.assertFalse(any(event['type'] == 'response.completed' for event in events))
        self.assertEqual(self.client.get('/v1/responses/' + response_id).json(), original)
        self.assertEqual(self.serving.msg_store[response_id], original_messages)
        aborted = []
        self.serving.tokenizer_manager.abort_request = lambda **kwargs: aborted.append(kwargs['rid'])
        async def disconnect():
            request = ResponsesRequest(model='fixture-qwen', input='different', tools=alternate,
                                       stream=True, request_id=response_id)
            stream = await self.serving.create_responses(request)
            await anext(stream)
            await stream.aclose()
        asyncio.run(disconnect())
        self.assertEqual(aborted, [response_id])
        self.assertEqual(self.client.get('/v1/responses/' + response_id).json(), original)
        self.assertEqual(self.serving.msg_store[response_id], original_messages)
        closed = []
        async def unfinished(request, *args, **kwargs):
            try:
                yield {'text': 'partial text', 'output_ids': [1], 'meta_info': {
                    'prompt_tokens': 10, 'completion_tokens': 1, 'finish_reason': None}}
                raise AssertionError('Disconnected generation must not resume')
            finally:
                closed.append(True)
        self.serving.tokenizer_manager.generate_request = unfinished
        async def disconnect_active():
            request = ResponsesRequest(model='fixture-qwen', input='different', tools=alternate,
                                       tool_choice='none', stream=True, request_id=response_id)
            stream = await self.serving.create_responses(request)
            async for frame in stream:
                if 'event: response.output_text.delta\n' in frame:
                    break
            await stream.aclose()
        asyncio.run(disconnect_active())
        self.assertEqual(closed, [True])
        self.assertEqual(aborted, [response_id, response_id])
        self.assertEqual(self.client.get('/v1/responses/' + response_id).json(), original)
        self.assertEqual(self.serving.msg_store[response_id], original_messages)

    def test_mock_http_namespace_nonstream_and_stateful_replay(self):
        response = self.send()
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        call = body['output'][0]
        self.assertEqual((call['name'], call['namespace']), ('read', 'workspace'))
        self.assertEqual(json.loads(call['arguments']), {'path': 'file'})
        retrieved = self.client.get('/v1/responses/' + body['id'])
        self.assertEqual(retrieved.json()['output'], body['output'])
        cancelled = self.client.post('/v1/responses/' + body['id'] + '/cancel')
        self.assertEqual(cancelled.json()['output'], body['output'])
        self.text = 'Done'
        replay = self.send(previous_response_id=body['id'], tool_choice='none',
                           input=[{'type': 'function_call_output', 'call_id': call['call_id'], 'output': 'data'}])
        self.assertEqual(replay.status_code, 200, replay.text)
        prompt = self.tokenizer.decode(self.generated[-1].input_ids)
        self.assertIn('workspace.read', prompt)
        self.assertIn('file', prompt)
        self.assertIn('data', prompt)

    def test_mock_http_namespace_sse_lifecycle(self):
        events = self.events(self.send(stream=True))
        self.assertFalse([event for event in events if event.get('type') == 'error'], events)
        added = next(event['item'] for event in events if event.get('type') == 'response.output_item.added')
        done = next(event['item'] for event in events if event.get('type') == 'response.output_item.done')
        terminal = next(event['response'] for event in events if event.get('type') == 'response.completed')
        self.assertEqual((added['name'], added['namespace']), ('read', 'workspace'))
        self.assertEqual(added['call_id'], done['call_id'])
        self.assertEqual(terminal['output'], [done])
        arguments = ''.join(event['delta'] for event in events if event.get('type') == 'response.function_call_arguments.delta')
        self.assertEqual(arguments, done['arguments'])
        self.assertEqual(json.loads(arguments), {'path': 'file'})

    def test_mock_http_custom_raw_and_stateless_replay(self):
        raw = '*** Begin Patch\n"escaped" \\ Ω 😀\n*** End Patch\n'
        self.tools = [{'type': 'namespace', 'name': 'workspace.files', 'tools': [
            {'type': 'custom', 'name': 'apply.patch', 'description': 'Apply raw patch'}]}]
        self.text = json.dumps([{'name': 'workspace.files.apply.patch', 'parameters': {'input': raw}}])
        body = self.send().json()
        call = body['output'][0]
        self.assertEqual(call['input'], raw)
        self.assertEqual((call['type'], call['namespace'], call['name']),
                         ('custom_tool_call', 'workspace.files', 'apply.patch'))
        events = self.events(self.send(stream=True))
        deltas = [event['delta'] for event in events if event.get('type') == 'response.custom_tool_call_input.delta']
        self.assertEqual(''.join(deltas), raw)
        done = next(event['item'] for event in events if event.get('type') == 'response.output_item.done')
        self.assertEqual(done['input'], raw)
        self.assertEqual(next(event['response'] for event in events if event.get('type') == 'response.completed')['output'], [done])
        self.text = 'Done'
        replay = self.send(tool_choice='none', store=False, input=[{'role': 'user', 'content': 'Apply patch'}, call,
                           {'type': 'custom_tool_call_output', 'call_id': call['call_id'], 'output': 'Applied'}])
        self.assertEqual(replay.status_code, 200, replay.text)
        prompt = self.tokenizer.decode(self.generated[-1].input_ids)
        self.assertIn('workspace.files.apply.patch', prompt)
        self.assertIn('Applied', prompt)

    def test_mock_http_rejects_unknown_and_forced_invalid(self):
        for choice in ({'type': 'function', 'name': 'missing', 'namespace': 'workspace'},
                       {'type': 'web_search'}, {'type': 'custom', 'name': 'read', 'namespace': 'workspace'}):
            with self.subTest(choice=choice):
                self.assertEqual(self.send(tool_choice=choice).status_code, 400)
                self.assertFalse(self.generated)
        self.text = '[{"name":"workspace.NOT_DECLARED","parameters":{}}]'
        self.assertEqual(self.send().status_code, 400)
        events = self.events(self.send(stream=True))
        self.assertTrue(any(event.get('type') in ('error', 'response.failed') for event in events), events)
        self.assertFalse(any(event.get('type') == 'response.completed' for event in events), events)

    def test_mock_http_flat_regression(self):
        self.tools = [{'type': 'function', 'name': 'flat.dotted', 'parameters': {'type': 'object'}}]
        self.text = '[{"name":"flat.dotted","parameters":{}}]'
        for stream in (False, True):
            response = self.send(stream=stream)
            body = next(event['response'] for event in self.events(response) if event.get('type') == 'response.completed') if stream else response.json()
            self.assertEqual(body['output'][0]['name'], 'flat.dotted')
            self.assertNotIn('namespace', body['output'][0])

    def test_mock_http_native_auto_and_required(self):
        self.serving.tool_call_parser = 'qwen3_coder'
        self.text = '<tool_call>\n<function=workspace.read>\n<parameter=path>file</parameter>\n</function>\n</tool_call>'
        for choice in ('auto', 'required', {'type': 'function', 'name': 'read', 'namespace': 'workspace'}):
            for stream in (False, True):
                with self.subTest(choice=choice, stream=stream):
                    response = self.send(tool_choice=choice, stream=stream)
                    self.assertEqual(response.status_code, 200, response.text)
                    body = next(event['response'] for event in self.events(response) if event.get('type') == 'response.completed') if stream else response.json()
                    self.assertEqual((body['output'][0]['name'], body['output'][0]['namespace']), ('read', 'workspace'))
                    self.assertEqual(json.loads(body['output'][0]['arguments']), {'path': 'file'})

    def test_mock_http_native_unknown_fails_closed(self):
        self.serving.tool_call_parser = 'qwen3_coder'
        self.text = '<tool_call>\n<function=workspace.NOT_DECLARED>\n<parameter=path>file</parameter>\n</function>\n</tool_call>'
        self.assertEqual(self.send(tool_choice='auto').status_code, 400)
        events = self.events(self.send(tool_choice='auto', stream=True))
        self.assertTrue(any(event.get('type') in ('error', 'response.failed') for event in events), events)
        self.assertFalse(any(event.get('type') == 'response.completed' for event in events), events)

    def test_mock_http_multimodal_tool_result_and_alias_provenance(self):
        image = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a/RsAAAAASUVORK5CYII='
        manager = self.serving.tokenizer_manager
        manager.model_config.is_multimodal = True
        self.serving.template_manager.jinja_template_content_format = 'openai'
        self.text = 'Seen'
        for effort in ('medium', 'high', 'max'):
            response = self.send(tool_choice='none', reasoning={'effort': effort}, input=[
                {'role': 'user', 'content': 'Inspect file'},
                {'type': 'function_call', 'name': 'read', 'namespace': 'workspace',
                 'call_id': 'call_image', 'arguments': '{}'},
                {'type': 'function_call_output', 'call_id': 'call_image', 'output': [
                    {'type': 'input_text', 'text': 'Image result'},
                    {'type': 'input_image', 'image_url': image}]}])
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['reasoning']['effort'], effort)
            self.assertTrue(self.generated[-1].image_data)
            self.assertIn('Image result', self.generated[-1].text)

    def test_mock_http_request_provenance_and_unrelated_model(self):
        import asyncio

        self.text = 'Hello'
        for effort in ('high', 'max', 'medium'):
            request = ResponsesRequest(model='fixture-qwen', input='Hi', reasoning={'effort': effort})
            before = copy.deepcopy(request.model_dump())
            response = asyncio.run(self.serving.create_responses(request))
            self.assertIsInstance(response, ResponsesResponse)
            self.assertEqual(response.status, 'completed')
            self.assertEqual(request.model_dump(), before)
            self.assertEqual(response.reasoning['effort'], effort)
        self.serving.tokenizer_manager.model_config.hf_config.model_type = 'llama'
        tokenizer = copy.deepcopy(self.tokenizer)
        tokenizer.chat_template = '{{ reasoning_effort }}'
        self.serving.tokenizer_manager.tokenizer = tokenizer
        self.serving.default_chat_template_kwargs = {}
        response = self.send(tools=[], tool_choice='none', reasoning={'effort': 'high'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.generated[-1].input_ids, tokenizer.encode('high', add_special_tokens=False))

    def test_mock_http_parallel_dotted_and_duplicate_local_names(self):
        self.tools = [{'type': 'namespace', 'name': namespace, 'tools': [
            {'type': 'function', 'name': 'lookup.part', 'parameters': {'type': 'object'}}]}
            for namespace in ('one.deep', 'two.deep')]
        self.text = json.dumps([{'name': namespace + '.lookup.part', 'parameters': {}}
                                for namespace in ('one.deep', 'two.deep')])
        for stream in (False, True):
            response = self.send(stream=stream)
            body = next(event['response'] for event in self.events(response) if event.get('type') == 'response.completed') if stream else response.json()
            self.assertEqual([(item['namespace'], item['name']) for item in body['output']],
                             [('one.deep', 'lookup.part'), ('two.deep', 'lookup.part')])
            self.assertEqual(len({item['call_id'] for item in body['output']}), 2)

    def test_mock_http_custom_native_empty_and_escaped(self):
        self.serving.tool_call_parser = 'qwen3_coder'
        self.tools = [{'type': 'custom', 'name': 'patch'}]
        tool = {'type': 'function', 'function': {'name': 'patch', 'parameters': {
            'type': 'object', 'properties': {'input': {'type': 'string'}}, 'required': ['input']}}}
        for raw in ('', 'line one\n"quoted" \\ slash Ω 😀', '\nraw\n', 'null',
                    '\n', '\n\n', ' \nraw\n ', '{"x":1}', '[1,2]', '\r\nraw\r\n'):
            rendered = self.tokenizer.apply_chat_template([
                {'role': 'user', 'content': 'Call patch'},
                {'role': 'assistant', 'content': '', 'tool_calls': [
                    {'type': 'function', 'function': {'name': 'patch', 'arguments': {'input': raw}}}]}],
                tools=[tool], tokenize=False, add_generation_prompt=False)
            start = rendered.rindex('<tool_call>')
            self.text = rendered[start:rendered.index('</tool_call>', start) + len('</tool_call>')]
            for stream in (False, True):
                response = self.send(tool_choice='auto', stream=stream)
                body = next(event['response'] for event in self.events(response) if event.get('type') == 'response.completed') if stream else response.json()
                self.assertEqual(body['output'][0]['input'], raw)

    def test_mock_http_custom_literal_angles_preserve_order_and_payload(self):
        self.serving.tool_call_parser = 'qwen3_coder'
        self.tools = [{'type': 'custom', 'name': 'patch'}]
        raw = '<tag>keep</tag> 1 < 2 and 3 > 2'
        tool = {'type': 'function', 'function': {'name': 'patch', 'parameters': {
            'type': 'object', 'properties': {'input': {'type': 'string'}},
            'required': ['input']}}}
        rendered = self.serving.tokenizer_manager.tokenizer.apply_chat_template([
            {'role': 'user', 'content': 'Call patch'},
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'type': 'function', 'function': {
                    'name': 'patch', 'arguments': {'input': raw}}}]}],
            tools=[tool], tokenize=False, add_generation_prompt=False)
        start = rendered.rindex('<tool_call>')
        tool_block = rendered[start:rendered.index(
            '</tool_call>', start) + len('</tool_call>')]
        self.text = 'Before <literal>.' + tool_block + 'After </literal>.'
        body = self.send(tool_choice='auto').json()
        self.assertEqual(
            [(item['type'], item.get('phase')) for item in body['output']],
            [('message', 'commentary'), ('custom_tool_call', None),
             ('message', 'final_answer')],
        )
        self.assertEqual(body['output'][0]['content'][0]['text'],
                         'Before <literal>.')
        self.assertEqual(body['output'][1]['input'], raw)
        self.assertEqual(body['output'][2]['content'][0]['text'],
                         'After </literal>.')

    def test_mock_http_replay_cannot_forge_flat_dotted_identity(self):
        self.tools = [{'type': 'function', 'name': 'workspace.read'}]
        response = self.send(input=[{'type': 'function_call', 'name': 'read', 'namespace': 'workspace',
                                    'call_id': 'call_forged', 'arguments': '{}'}])
        self.assertEqual(response.status_code, 400)
        self.assertFalse(self.generated)

    def test_mock_http_flat_history_without_active_tools(self):
        self.text = 'Remembered'
        response = self.send(tools=[], tool_choice='none', input=[
            {'role': 'user', 'content': 'Remember previous result'},
            {'type': 'function_call', 'name': 'old.lookup', 'call_id': 'call_old', 'arguments': '{}'},
            {'type': 'function_call_output', 'call_id': 'call_old', 'output': 'old data'}])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn('old.lookup', self.tokenizer.decode(self.generated[-1].input_ids))

    def test_mock_http_forced_choice_cannot_emit_other_declared_tool(self):
        self.tools[0]['tools'].append({'type': 'function', 'name': 'write', 'parameters': {'type': 'object'}})
        self.text = '[{"name":"workspace.write","parameters":{}}]'
        choice = {'type': 'function', 'name': 'read', 'namespace': 'workspace'}
        self.assertEqual(self.send(tool_choice=choice).status_code, 400)
        events = self.events(self.send(tool_choice=choice, stream=True))
        self.assertTrue(any(event.get('type') in ('error', 'response.failed') for event in events), events)
        self.assertFalse(any(event.get('type') == 'response.completed' for event in events), events)

    def test_mock_http_json_schema_and_explicit_nulls(self):
        self.text = '{"value":1}'
        response = self.send(tools=[], tool_choice='none', text={'format': {
            'type': 'json_schema', 'name': 'result', 'schema': {
                'type': 'object', 'properties': {'value': {'type': 'integer'}}}}})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(json.loads(self.generated[-1].sampling_params['json_schema'])['type'], 'object')
        self.assertIn('error', response.json())
        self.assertIsNone(response.json()['error'])
        self.assertIn('incomplete_details', response.json())

    def test_direct_flat_result_retains_typed_api(self):
        import asyncio

        self.text = 'Hello'
        response = asyncio.run(self.serving.create_responses(ResponsesRequest(
            model='fixture-qwen', input='Hi')))
        self.assertIsInstance(response, ResponsesResponse)
        self.assertEqual(response.status, 'completed')
        self.assertEqual(response.output[0].content[0].text, 'Hello')

    def test_mock_http_rejected_call_cannot_be_replayed_from_store(self):
        self.serving.tool_call_parser = 'qwen3_coder'
        self.text = '<tool_call><function=workspace.NOT_DECLARED><parameter=path>file</parameter></function></tool_call>'
        response = self.send(tool_choice='auto', request_id='resp_rejected_cpu_fixture')
        self.assertEqual(response.status_code, 400)
        self.text = 'Done'
        replay = self.send(previous_response_id='resp_rejected_cpu_fixture', tool_choice='none')
        self.assertNotEqual(replay.status_code, 200)


if __name__ == '__main__':
    unittest.main()
