# Selective HiCache restore diagnostics

Patches0030/0031 add explanations for known cache data that cannot be restored and
correlate those failures with substantial subsequent prefill. They do not change
cache matching, eviction, transfer ordering, timeout policy or scheduling decisions.
Ordinary cold misses remain quiet.

`HiCache restore fallback` reports request ID, reason, requested/matched token
counts, elapsed/budget seconds and suppressed event count. Reasons include
`lookup_timeout`, `checkpoint_missing`, `peer_miss`, `host_capacity`,
`restore_incomplete`, `prefetch_capacity`, `component_capacity` and `backup_pending`.
Matched tokens describe known availability, not successfully restored tokens.
A zero time budget means not applicable. Best-effort decisions and cancellation
are not reported as timeouts.

Enriched details identify raw KV availability, common checkpoint compatibility,
selected length, host allocation before/after eviction and per-component completed
pages. A `HiCache prefill impact` event on admission links a retained fallback to
input, reused, host-hit, storage-loaded and admitted-new token counts when at least
16,384 tokens still need prefill. These are admission counts, not completion receipts.

- Cold misses increment bounded reason/rank counters only.
- Capacity/backlog messages require at least16,384requested tokens and are limited
  to one event per reason per60seconds; suppressed counts appear on the next event.
- Request-specific matched-but-unrestorable failures remain visible.
- Rank0 emits logs; all ranks expose counters. Per-component details include local
  counts and synchronized minima, not an identification of every failing rank.
- Retained per-request correlation is bounded to4,096 entries and cleared on
  admission/abort. Request IDs are never metric labels.
- Prompt text, token IDs and cache file keys are not included in new diagnostics.
- Ordinary missing files become DEBUG; real read/I/O failures remain warnings.
- No eviction history is introduced. A cold miss cannot distinguish never-written
  data from eviction, and a successful disk lookup does not guarantee a usable
  recurrent checkpoint or RAM staging allocation.

## Build and CPU validation

```sh
docker build -f Dockerfile.hicache-wip -t local/qwen-hicache-diagnostics .
HICACHE_DIAGNOSTICS_IMAGE=local/qwen-hicache-diagnostics \
  bash scripts/test_restore_isolation_cpu.sh
```

The public pinned base is reconstructed through the declared patch series.
`verify_hicache_wip.py` checks patch order/hashes, preimages and the full resulting
source inventory. Packaging0030 removes a duplicate field annotation present in
the first local build; it has no runtime semantic effect.

The diagnostic suite has101CPU cases, including quiet cold misses, rate limiting,
rank suppression, metrics, missing-file versus bad-read behavior, common-boundary,
QSA/PLE transfer regressions and actual scheduler admission hooks. Restore-isolation
adds9CPU cases. GPU/runtime evidence and limitations are documented separately in
[restore-isolation-validation.md](restore-isolation-validation.md).

See [hicache-production-recipe.md](hicache-production-recipe.md) for the6144-token,
64-request,90GB RAM/1TB disk write-back configuration and why RAM smaller than GPU
makes write-through unsuitable for this implementation.
