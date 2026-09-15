"""Run bundled Responses tests with the cumulative build's replay contract."""
import tempfile
from pathlib import Path

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()
import pytest

root = Path('/sgl-workspace/sglang/test/registered/unit/entrypoints/openai')
source = (root / 'test_serving_responses.py').read_text()
old = '"content": "first answer part\\nsecond answer part",'
new = '''"content": [
                        {"type": "text", "text": "first answer part"},
                        {"type": "text", "text": "second answer part"},
                    ],'''
assert source.count(old) == 1, 'Bundled replay fixture changed; review before adapting it'
# The cumulative API deliberately preserves content parts (including media),
# rather than injecting a newline. The original fixture also fails on the parent.
with tempfile.TemporaryDirectory(prefix='upstream-responses-') as tmp:
    adapted = Path(tmp) / 'test_serving_responses.py'
    adapted.write_text(source.replace(old, new))
    print('Adapting one replay expectation to preserve text parts; all tests remain enabled.', flush=True)
    raise SystemExit(pytest.main([
        str(root / 'test_responses_protocol.py'), str(adapted),
        str(root / 'test_serving_responses_stream.py'), '-q', '-p', 'no:cacheprovider',
    ]))
