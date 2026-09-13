# Qwen Flash-Next effort aliases — candidate, not deployed

This source-only candidate is based on main `b8e8bebf2274e0099c3abce5718ac8813dd9001d`
and the published Chat precedence image
`kanadaj/sglang-qwen38fn-sm120-turbo@sha256:872a2bda228e39aa9c1af729b47cc28f7862e7859e448f1a8868b85a4051f404`.
No image was built/published and no production settings or checkpoint files were changed.
Historical production/combined manifests and patch series remain unchanged.
Do not use the historical `Dockerfile.reasoning-effort` to claim a reconstruction
of the deployed image from the candidate runtime overlay.

## Policy and scope

Only loaded `hf_config.model_type` equal to `qwen3_8_flash_next` or
`qwen3_8_flash_next_text` opts in. A client `model` name cannot opt another
checkpoint in. `high` and `max` deliberately render as `xhigh`; case/whitespace
variants are not repaired. Other models retain native `high` behavior.

Precedence: nonnull `chat_template_kwargs.reasoning_effort`, then nonnull
request effort, then server default. The production server default stays
`medium`. `low`, `medium`, `xhigh`, omitted and null retain their intended
meaning. Without a configured server default the checkpoint's own default
still applies; this patch does not introduce a new global default.

Normalization occurs on a rendering-only request copy. Caller request fields,
including conflicting top/nested literal values, remain available for diagnostic
provenance; the returned Chat request retains the selected literal tier rather
than reporting that the caller requested `xhigh`. Responses request/echo fields
are not aliased. No extra logging, prompt capture, or checkpoint template rewrite.

### API coverage

- Chat `/v1/chat/completions`: top-level `reasoning_effort`, nested template
  kwargs, and the existing `reasoning.effort` compatibility input; stream and
  nonstream share the tested conversion.
- Responses `/v1/responses`: `reasoning.effort` and nested template kwargs via
  the actual non-Harmony `_make_request` conversion. This Qwen-specific path
  also fixes request effort being hidden by the medium server default.
- `/v1/tokenize` with messages: actual `_tokenize_chat_request` path.
- Text-only and multimodal prompt rendering branches share normalization;
  CPU tests cover rendered multimodal text, not image encoding/inference.
- Raw completion/generate/tokenize-prompt inputs do not apply a chat template
  and are not normalized. Harmony/custom encoders and other model families
  are outside this patch. No HTTP server or GPU inference validation is claimed.

**Schema finding:** the exact published base already accepts `max` in
`ReasoningEffortTier`, Chat and `ResponseReasoningParam`. Both imported request
classes were exercised. No schema widening or global alias is needed. The
failure in this runtime is the unchanged Qwen template rejecting high/max.

## Reproduction

The manifest `provenance/qwen-effort-alias.json` pins the image, source before/
after hashes, patch hash, and tokenizer revision/file hashes. Download only
`config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`
from `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` at revision
`ada4da32a583a78aa47299f45a70603c950490b8` into a scratch tokenizer directory.
These are unmodified public checkpoint metadata, not a claimed fresh live-host
checkpoint export. No weights are needed.

```bash
QWEN_TOKENIZER_PATH=/absolute/scratch/tokenizer bash scripts/test_qwen_effort_alias.sh
```

The runner requires the published image locally, forbids pulling/network, mounts
the repository and tokenizer read-only plus a read-only single-file runtime
overlay, and exposes no GPUs. Tokenizer, runtime and patch hashes are checked
before execution. Expected: 13 imported API
test methods (including the four unchanged Chat regressions), then 77 existing
CPU package tests. CUDA-unavailable and deprecated-max_tokens warnings are
inherited from the baseline.

For full source reconstruction, export `/sgl-workspace/sglang/python/sglang`
from that exact image to `TREE/python/sglang`, then:

```bash
python3 scripts/verify_qwen_effort_alias.py --tree TREE --apply
python3 scripts/verify_qwen_effort_alias.py --tree TREE
```

This verifies all 4,391 base files before application, applies ordered patch
0015 after the existing production + Chat-precedence source profile, and
verifies the complete candidate path/content inventory. Source equivalence is
not byte-identical Docker-image reproducibility. The new patch is intentionally
not appended to historical production/combined series.
