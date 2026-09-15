"""Run the full compatibility suite with the new failed-stream contract.

Keep the predecessor suite unchanged for its older image profiles. Override only
changed failure expectations and add the two new HTTP regression scenarios.
"""
import json
import unittest
import runtime_responses_compat as compat


class StreamStabilityTest(compat.MockHTTPTest):
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
                                            self.assertEqual(len([event for event in events if event.get('type') == 'response.failed' and event['response'].get('error', {}).get('message')]), 1, events)
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
                            self.assertEqual(len([event for event in events if event.get('type') == 'response.failed' and event['response'].get('error', {}).get('message')]), 1, events)
                            self.assertFalse(any(event.get('type') == 'response.completed' for event in events), events)
                        else:
                            self.assertEqual(rejected.status_code, 400, rejected.text)
                        added = set(self.serving.response_store) - existing
                        if stream:
                            self.assertEqual(len(added), 1)
                            self.assertEqual(self.serving.response_store[added.pop()].status, 'failed')
                        else:
                            self.assertFalse(added)
                        accepted = self.send(tool_choice='required', stream=stream)
                        body = next(event['response'] for event in self.events(accepted)
                                    if event.get('type') == 'response.completed') if stream else accepted.json()
                        self.assertEqual(body['status'], 'completed', body)
                        self.assertEqual(len(body['output']), 2)


    def test_stability_unwrapped_markup_remains_text(self):
        self.serving.tool_call_parser = 'qwen3_coder'
        self.text = 'Example syntax: <function=function></function>'
        for stream in (False, True):
            response = self.send(tool_choice='auto', stream=stream)
            body = next(e['response'] for e in self.events(response)
                        if e.get('type') == 'response.completed') if stream else response.json()
            self.assertEqual(body['output'][0]['content'][0]['text'], self.text)
            self.assertFalse(any(i['type'] == 'function_call' for i in body['output']))


    def test_stability_unknown_retains_commentary_and_error(self):
        self.serving.tool_call_parser = 'qwen3_coder'
        self.text = 'Checking.<tool_call><function=missing></function></tool_call>'
        events = self.events(self.send(tool_choice='auto', stream=True))
        failed = [e['response'] for e in events if e.get('type') == 'response.failed']
        self.assertEqual(len(failed), 1, events)
        self.assertIn('Unknown generated tool identity: missing', failed[0]['error']['message'])
        self.assertEqual(failed[0]['output'][0]['content'][0]['text'], 'Checking.')
        self.assertFalse(any(e.get('type') == 'response.completed' for e in events))
        self.assertFalse(any(e.get('item', {}).get('name') == 'missing' for e in events))


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name in dir(compat):
        cls = getattr(compat, name)
        if isinstance(cls, type) and issubclass(cls, unittest.TestCase):
            suite.addTests(loader.loadTestsFromTestCase(StreamStabilityTest if cls is compat.MockHTTPTest else cls))
    return suite


if __name__ == '__main__':
    unittest.main()
