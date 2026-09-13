# Read-only performance diagnostics

`python3 scripts/collect_diagnostics.py` is a standalone Python 3.10+ standard-library collector. It does not start inference, benchmark GPU/P2P bandwidth, install packages, tune settings, escalate privileges, restart services, or upload reports. Linux provides the fullest host data; missing tools, unsupported drivers and denied access produce explicit unavailable fields, not false or zero.

## Start with the comparison contract

The reported outsider result is **1320 tok/s decode at C16**. Our historical same-pair reference is **1705.3 tok/s**, one TP2 replica at C16 on GPU0/1 with NEXTN3 and the private draft head (two-step reference: 1664.3). These are not new measurements from this collector, a fleet result, or proof of a cause. Establish comparability before diagnosing the gap:

- Pure decode versus end-to-end including prefill; actual emitted-token counts and wall-clock denominator.
- C16 **per replica** versus fleet C16 split across replicas; requested versus observed running/queued requests, TP/DP and participating GPU pair.
- Exact marketed card edition, power/clock policy, driver/runtime, model/quantization and kernel backends.
- Input/output lengths and distributions, context limit, chunked prefill, KV and Mamba capacity; cold versus cached prefixes and concurrent arriving prefill.
- Warmup, JIT, graph capture, measurement duration and repetitions; competing activity.
- Reasoning and answer-token counting, EOS handling, output caps, sampling and truncation policy.
- Speculation algorithm/steps/topk/draft tokens and acceptance. The reference uses NEXTN **3/1/4**, `SGLANG_PRIVATE_DRAFT_NVFP4_A16=1`; configured gates are not proof of executed kernel dispatch.

## Usage

```bash
# Local host/GPU snapshot; no endpoint requests or Docker access by default.
python3 scripts/collect_diagnostics.py --output diagnostic.json

# Select the serving pair; observe an ALREADY running workload, never create one.
python3 scripts/collect_diagnostics.py --gpus 0,1 \
  --duration 10 --interval 1 --output existing-load.json

# Optional existing local Docker container and direct SGLang origin.
# DIAGNOSTIC_TOKEN must already be supplied securely in the environment.
python3 scripts/collect_diagnostics.py --gpus 0,1 \
  --container YOUR_EXISTING_CONTAINER --endpoint http://127.0.0.1:30000 \
  --token-env DIAGNOSTIC_TOKEN --duration 10 --interval 1 \
  --output serving.json

# Compare only allowlisted numeric fields of an earlier report.
python3 scripts/collect_diagnostics.py --output later.json --baseline diagnostic.json

python3 -B -m unittest discover -s tests -p test_diagnostics.py -v
```

Both `FILE` (JSON) and `FILE.txt` (readable evidence/caveats) must be new. Files are created mode 0600; errors suppress paths and supplied arguments. Review **both** before sharing. Sampling accepts 0–60 seconds, intervals 0.25–10 seconds and at most 120 intervals. Subprocess commands retain a three-second wait timeout and a 1 MiB capture limit. Each HTTP call has a three-second total request deadline, including isolated-interpreter startup, DNS resolution, connection/TLS setup, response headers and body, with a 1 MiB body limit. On expiry the parent kills and reaps the HTTP worker; OS scheduling and cleanup can add a small overrun. The worker's socket inactivity timeout is an additional safeguard, not the total deadline: trickling headers or bytes cannot extend the parent's budget. Collection is sequential, so setup and a final in-flight sample can exceed the requested window; actual elapsed intervals are recorded. This is not a hard total collector-process deadline.

The container opt-in performs a bounded Docker inspect, allowlists the result in memory and runs one isolated Python package-metadata probe (`-I -S -B`), without importing torch/SGLang or initializing CUDA. Both Docker commands are pinned to `--host unix:///var/run/docker.sock`, use an empty temporary Docker configuration, and inherit no `DOCKER_*` variables. Remote hosts, environment contexts and the configured default context cannot override this transport; alternative/rootless sockets are not automatically discovered. It requires existing Docker permissions; never add permissions just to run this diagnostic.

Serving-container cgroups are read only when local Unix-socket peer credentials identify a visible daemon PID whose PID and mount namespaces match the collector. Recognized collector-container markers, invisible peers, namespace mismatches or unavailable verification fail closed with `container_pid_namespace_unverified`; the inspect PID is not guessed against local `/proc` or used for subsequent interval samples. A host with separately namespaced Docker can therefore still provide inspect/package evidence but not serving-container cgroups. Package metadata is not loaded-library attestation. Wrapper scripts can hide effective serving arguments: use endpoint settings as additional evidence, never infer missing flags are disabled.

The endpoint opt-in performs only GET `/get_server_info` and `/metrics`. Use a direct instance origin, not a load-balanced fleet URL; endpoint-to-GPU identity is not verified. Bearer authentication comes only from the named environment variable and is passed to the fresh HTTP interpreter through its inherited environment, never command-line arguments or request-control stdin. Endpoint/route control uses an anonymous pipe, not command-line arguments. The worker starts by executing a fresh interpreter rather than running Python in a forked copy of the collector's reader threads. Worker stderr is discarded; the parent accepts only fixed failure reasons and applies the existing field allowlists before exporting response data. Malformed HTTP status/header/body exceptions become a fixed unavailable reason without response text or endpoint details. URLs with credentials, query strings, fragments or paths are rejected; redirects and environment proxies are disabled. Use HTTPS for tokens unless the connection is trusted local transport. The collector does not scrape logs, prompts or completions. Unsupported metrics and duplicate labeled series remain unavailable rather than blindly summing TP ranks.

Baseline comparison accepts only allowlisted, finite numeric values from zero through `1e20`. Oversized integers, including several-hundred-digit JSON numbers, remain unavailable rather than aborting collection; valid neighboring fields still compare normally.

## Evidence and privacy boundaries

- GPU output: strict model/version formats, VRAM, configured/default/max power, clocks, temperature, utilization, current/max PCIe, event flags/counter deltas, selected-pair topology and P2P read/write capability. Synthetic GPU0/GPU1 follow selection order; local GPU IDs, UUIDs, serials and bus addresses are not exported.
- Host output: allowlisted CPU model, CPU/NUMA affinity, memory/swap, PSI, interval swap I/O and CPU ticks; collector and optional serving-container cgroup constraints/events and configured Docker shared memory.
- Runtime output: allowlisted numeric/enumerated graph, quantization, TP, PLE, head, speculation, cache, Mamba and chunk settings; image digest and constrained package versions. No raw environment, inspect JSON, command lines, endpoints, IPs, paths, labels or arbitrary strings are saved. Hardware/configuration still fingerprints a system; this is not an anonymity guarantee.
- P2P `true` means capability, **not bandwidth** or proof of the custom all-reduce path. NODE/PHB describe different paths, not measured latency. Idle PCIe downshift is not a fault. SM activity is not saturation; power-cap events are not thermal events. Lifetime accumulated counters are not current-window throttling.
- Accepted emitted tokens per request verification differ from global batch iterations. The latter are explicitly not measured; any cadence estimate requires steady pure decode/full batches. Completion-accounted counters can lag decode; these reports cannot establish benchmark throughput or a root cause.

No production launch or deployment files are changed by this tool. Keep generated reports outside the repository; do not commit private live captures.
