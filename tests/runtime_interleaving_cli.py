"""Exercise actual CLI registration in the isolated CPU image."""
import argparse
from sglang.srt.server_args import ServerArgs


def test_cli_registration_and_default():
    parser=argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    explicit=parser.parse_args(['--model-path','synthetic','--prefill-batches-before-decode','2'])
    default=parser.parse_args(['--model-path','synthetic'])
    assert explicit.prefill_batches_before_decode==2
    assert default.prefill_batches_before_decode==0
