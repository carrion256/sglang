# Published production-source runtime — 2026-09-11

## Pull and run without a checkout

```bash
docker pull docker.io/kanadaj/sglang-qwen38fn-sm120-turbo:production-private-draft-20260911-5413729@sha256:baf8d37cd0e6c95d6184ba6a609b410141bc3fca43db9c1c77dff1c4ba3b9421
```

This is an ordinary **Linux amd64 SGLang runtime**, not a launcher with hidden
serving defaults. The full command below keeps NEXTN **steps3 / topk1 / draft4**,
private NVFP4 W4A16 draft-head gate, TP2, packed CPU PLE, vision, Mamba512/track128,
context524288 and complete nested YaRN2 configuration external.

**Choose unoccupied GPUs and a free port.** The example binds the API to localhost;
use an authenticated gateway before exposing it. Supply a fully downloaded
Qwen3.8 Flash-Next LIL NVFP4 local checkpoint at the host mount. There are no
model weights in this image. The production-source profile deliberately excludes
HF-repo-ID fix0012: do **not** substitute a repository ID for `/model` and assume
that older loader supports the corrected acquisition path. The separately
published historical HF-path runtime and the combined source profile remain
unchanged.

```bash
# Example only: do not run on occupied GPUs/ports. Public runtime; no build required.
# Requires a complete local checkpoint; see production-image-20260911.md.
docker run --rm --gpus '"device=0,1"' --ipc=host -p 127.0.0.1:30000:30000 \
  -v /absolute/path/to/local-checkpoint:/model:ro \
  -e SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=1 \
  -e SGLANG_SM120_ONLINE_MXFP8=false \
  -e SGLANG_PLE_PACKED_NVFP4=1 \
  -e SGLANG_PLE_PACKED_FP8_REFERENCE=1 \
  -e SGLANG_PRIVATE_DRAFT_NVFP4_A16=1 \
  --entrypoint python3 docker.io/kanadaj/sglang-qwen38fn-sm120-turbo:production-private-draft-20260911-5413729@sha256:baf8d37cd0e6c95d6184ba6a609b410141bc3fca43db9c1c77dff1c4ba3b9421 -m sglang.launch_server \
  --model-path \
  /model \
  --enable-metrics \
  --uvicorn-access-log-exclude-prefixes \
  /metrics \
  --reasoning-parser=auto \
  --tool-call-parser=auto \
  --default-chat-template-kwargs \
  '{"reasoning_effort":"medium"}' \
  --linear-attn-prefill-backend=flashinfer \
  --linear-attn-decode-backend=flashinfer \
  --max-mamba-cache-size=512 \
  --mamba-radix-cache-strategy=extra_buffer \
  --mamba-track-interval=128 \
  --mamba-ssm-dtype=bfloat16 \
  --gdn-mtp-cache-mode=none \
  --tp-size=2 \
  --quantization=modelopt_mixed \
  --kv-cache-dtype=fp8_e4m3 \
  --context-length=524288 \
  --mem-fraction-static=0.93 \
  --page-size=64 \
  --chunked-prefill-size=4096 \
  --max-running-requests=16 \
  --enable-metrics \
  --enable-cache-report \
  --cuda-graph-max-bs-decode=16 \
  --moe-runner-backend=flashinfer_cutlass \
  --disable-custom-all-reduce \
  --disable-prefill-cuda-graph \
  --json-model-override-args \
  '{"text_config":{"vocab_size":248320,"hidden_size":2560,"intermediate_size":12288,"num_hidden_layers":48,"num_attention_heads":24,"num_key_value_heads":2,"hidden_act":"silu","max_position_embeddings":524288,"initializer_range":0.02,"rms_norm_eps":1e-06,"use_cache":true,"head_dim":256,"attention_bias":false,"attention_dropout":0.0,"linear_conv_kernel_dim":4,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_num_key_heads":16,"linear_num_value_heads":48,"layer_types":["linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention"],"moe_intermediate_size":640,"shared_expert_intermediate_size":640,"num_experts_per_tok":10,"num_experts":512,"output_router_logits":false,"router_aux_loss_coef":0.001,"hc_count":4,"hc_lowrank":320,"ple_layer_ids":[2],"ple_embed_dim":2560,"ple_conv_kernel_size":4,"ngram_size":3,"heads_per_ngram":8,"ngram_vocab_size_base":20000000,"make_ngram_vocab_size_divisible_by":128,"split_ngram_parts":128,"output_gate_type":"sigmoid","indexer_n_heads":4,"indexer_kv_heads":1,"indexer_head_dim":128,"indexer_budget":2048,"indexer_compress_ratio":4,"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144,"rope_theta":10000000,"mrope_interleaved":true,"mrope_section":[11,11,10],"partial_rotary_factor":0.25},"output_hidden_states":false,"return_dict":true,"dtype":"bfloat16","chunk_size_feed_forward":0,"is_encoder_decoder":false,"id2label":{"0":"LABEL_0","1":"LABEL_1"},"label2id":{"LABEL_0":0,"LABEL_1":1},"problem_type":null,"_name_or_path":"","pad_token_id":null,"bos_token_id":248044,"eos_token_id":248044,"tie_word_embeddings":false,"mamba_ssm_dtype":"float32","mtp":{"hybrid":true,"layer_types":["full_attention"],"mtp_use_hidden_state_from_layer":null,"num_hidden_layers":1,"rope_theta":10000000},"mtp_num_hidden_layers":1,"mtp_use_dedicated_embeddings":false,"model_type":"qwen3_8_flash_next_text","output_attentions":false,"ple_embedding_dtype":"nvfp4"},"max_position_embeddings":524288}' \
  --ple-offload-embedding \
  --speculative-algorithm=NEXTN \
  --speculative-num-steps=3 \
  --speculative-eagle-topk=1 \
  --speculative-num-draft-tokens=4 \
  --speculative-draft-model-quantization=modelopt_mixed \
  --speculative-moe-runner-backend=flashinfer_cutlass \
  --model-loader-extra-config \
  '{"enable_multithread_load":false,"num_threads":2}' \
  --startup-weight-load-mode=serial \
  --mm-enable-dp-encoder \
  --host \
  0.0.0.0 \
  --port \
  30000
```

The identical command is also in [`production-command.sh`](production-command.sh).
Only host mount, GPU pair and published port normally need changing. This command
was parser-checked, **not booted on GPUs** during publication.

## Immutable identity and build provenance

| Identity | Value |
|---|---|
| Registry OCI index | `sha256:baf8d37cd0e6c95d6184ba6a609b410141bc3fca43db9c1c77dff1c4ba3b9421` |
| Linux amd64 manifest | `sha256:fbcbc38448c7d01dfdec307b869c535c70fc06286ef6324af55f3a25303465cc` |
| Runtime config digest | `sha256:7f97605c237fef19d6069c69d99d6f4a4fbb09697884525b065511fa5bdacc17` |
| Build source | `5413729c1a4302f218cfb43f47b071ef18482e83` |
| Recipe | `Dockerfile.production`, `--pull --no-cache`, fresh GitHub clone |
| Pinned public base | `lmsysorg/sglang:qwen38flashnext@sha256:59f06adce6f91401adf443bd168d45fdb2044d77671fd591c7c57a29d851cbae` |

Built on a separate workstation, not the production GPU host. BuildKit's SLSA
attestation records the source revision and recipe. The versioned tag was absent
before push; no `latest`, stable tag or historical HF-fixed tag was overwritten.
Registry push and anonymous manifest reads returned the index above.

**Do not confuse an index with an image config.** The private live descriptor
`sha256:69f1f64c62ca2b5d919d69bcd60efb7fc89d7bf410ba406a5de586051f358465`
is a different OCI index, not this build's config ID. Containerd-backed Docker may
show the index in `.Id`; classic Docker normally identifies an image by config.
Use the explicitly named index/platform/config mapping above.

This is a **clean source-equivalent rebuild, not a byte-identical export or a new
production deployment**. All 4,391 attested non-bytecode SGLang source paths and
hashes match production, including the private draft head and excluding fix0012.
The runtime's public base/dependency ancestry is preserved; content attestation
is not a new native-ABI, full-model accuracy or performance qualification.
No live container was committed, restarted, reconfigured or used for a build.

## Publication-safety audit

Before push, all **77 retained layers** were scanned, covering **284,981 regular
file versions / 32,263,906,008 uncompressed file bytes**, including lower-layer
files later overwritten/deleted. Config/history and the BuildKit attestation were
also inspected. The first 68 layer diffIDs and history prefix match the pinned
public base; environment values are exactly that public base's values.
The nine added filesystem layers contain package source/provenance/licenses and
compiled Python caches, not launch profiles, checkpoints or private snapshots.
Entrypoint is `python3 -m sglang.launch_server`, with default command `--help`.

No user credentials or model checkpoint payloads were detected. The 68 pattern
hits are confined to byte-identical public-base layers: cryptographic format
strings and test fixtures, AWS example identifiers, a sandbox-only Transformers
CI constant, and upstream shared SSH host keys. **Never use the inherited shared
host keys for an SSH service.** This image runs SGLang, not SSH. The only
`.safetensors` candidate is the upstream library's int8 Hadamard transform-matrix
table (1,436,901 bytes), not learned model weights.

This is a provenance-backed publication review, **not a claim of zero key
material, a vulnerability scan, or mathematical proof that arbitrary secrets
cannot exist**. Private original snapshot metadata and raw audit material were
kept outside Git. No credential values are included in the public receipt.

## Exercised validation

- Fresh-clone no-cache Docker build: completed; reconstruction verified all4,391
  source files, followed by actual-image source verification.
- Repository CPU suite inside the built image: **77/77 passed** using
  `bash scripts/test.sh`, with its required `TRITON_INTERPRET=1` and CPU thread
  settings. Tests use tiny synthetic tensors, not checkpoints.
- Initial bare-unittest invocation failed five GPU-dispatch tests because it
  omitted the prescribed interpreter environment; the original failure was
  retained and diagnosed, not waived. Running host Python also lacked Torch;
  validation was moved into the actual runtime instead of inventing results.
- Installed `sglang`, `private_draft_head` and `packed_ple` imports passed with
  `torch.cuda.is_available() == False`; no GPU devices or model mounts exposed.
- Actual installed CLI `--help` and parser accepted all47 external argument
  tokens; asserted TP2, context524288, nested overrides and speculation3/1/4.
- After publication, Docker with an empty authentication configuration pulled
  the digest and the digest-pinned runtime repeated source and import checks.
  That Docker daemon reused its existing layers; independent clean-cache
  download evidence is recorded separately below.
- No inference, model download, standalone GPU boot, GPU benchmark or production
  deployment change was performed as part of publication. Earlier accepted
  [quality/performance evidence and limitations](production-20260911.md) remain
  historical evidence, not tests of a newly deployed clean rebuild.

### Independent anonymous pull

An independent registry client, without reading Docker/GitHub credentials,
resolved the public index and platform/attestation manifests, then downloaded
**all 74 unique config/layer/attestation blobs (13,959,689,290 bytes)** into a
new, initially empty OCI directory. Every descriptor size and SHA256 matched;
the downloaded runtime config exactly matched the audited build config.
The OCI directory was successfully streamed into `docker load`, followed by
another digest-pinned installed-runtime import and all4,391-file source check.
Docker could reuse its daemon cache at load/run time; the independent download
itself did not reuse any layer cache.

The complete manifest/blob digest receipt, build/base identity and sanitized
validation summary are in [`publication.json`](../provenance/production/publication.json).
Actual CPU/parser/source/import/load output is retained in
[`publication-validation.log`](../provenance/production/publication-validation.log).
Private original-image metadata, the full per-layer scan, original failed test
receipts and downloaded image blobs remain outside Git.
