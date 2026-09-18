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


def test_production_recipe_arguments():
    import shlex
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "docs/hicache-production-recipe.md").read_text()
    block = next(block.split("```", 1)[0] for block in text.split("```sh\n")[1:]
                 if block.startswith("--tp-size="))
    flags = shlex.split(block.replace("\\\n", ""))
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed = parser.parse_args(["--model-path", "synthetic", *flags])
    assert parsed.prefill_batches_before_decode == 0.5
    assert parsed.tp_size == 2
    assert parsed.chunked_prefill_size == 6144
    assert parsed.enable_hierarchical_cache
    assert parsed.hicache_write_policy == "write_back"
