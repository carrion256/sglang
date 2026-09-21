"""Qwen wrapper grammar across arbitrary stream boundaries."""
import pytest
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector

@pytest.mark.parametrize('text', [
    'Example syntax: <function=function></function>',
    '<parameter=name>value</parameter>',
    '<function=workspace.read><parameter=path>file</parameter></function>',
])
def test_unwrapped_markup_is_text(text):
    for cuts in [[], *[[i] for i in range(1, len(text))], list(range(1, len(text)))]:
        detector = Qwen3CoderDetector()
        parts = [text[a:b] for a, b in zip([0]+cuts, cuts+[len(text)])]
        results = [detector.parse_streaming_increment(part, []) for part in parts]
        assert not any(r.calls for r in results)
        assert ''.join(r.normal_text for r in results) == text


def test_wrapped_name_split_every_boundary():
    text = '<tool_call><function=functions.write_file></function></tool_call>'
    for cut in range(1, len(text)):
        detector = Qwen3CoderDetector()
        calls = detector.parse_streaming_increment(text[:cut], []).calls
        calls += detector.parse_streaming_increment(text[cut:], []).calls
        assert [c.name for c in calls if c.name] == ['functions.write_file']
