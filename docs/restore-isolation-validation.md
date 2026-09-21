# HiCache restore isolation validation

Tests exercise distinct logical checkpoints through reused physical GPU/host slots,
restore them out of write order and compare exact target/draft KV, QSA, Mamba and
PLE tensors. Test storage is isolated temporary storage, never the production cache.

## Run

Build `Dockerfile.hicache-wip` as described in
[hicache-selective-diagnostics.md](hicache-selective-diagnostics.md). Supply that
image explicitly:

```sh
export HICACHE_DIAGNOSTICS_IMAGE=local/qwen-hicache-diagnostics
bash scripts/test_restore_isolation_cpu.sh
# Maintenance only: first stop inference with authorization and verify GPU idleness.
bash scripts/test_restore_isolation_gpu.sh 0
bash scripts/test_restore_isolation_gpu.sh 1
# Against a running server; no cache flush or deletion:
python3 validation/hicache/live_restore_isolation.py \
  --base-url http://127.0.0.1:5331 --model qwen3.8-flash-next \
  --output /path/to/new/results-directory
```

The GPU runner checks the local Qwen systemd unit, not arbitrary GPU occupants.
Explicitly check device occupancy before running it on another installation.
It does not stop services or reset GPUs. CPU tests require no GPU devices.

## September16 evidence

- 101existing CPU diagnostic/cache tests passed.
- 9additional CPU tests passed: file rank/key isolation, reopened storage,
 replacement isolation and6real duplicate-reclamation eligibility cases.
- 10GPU tests passed on each of two physical GPUs.
- Five complete hybrid checkpoints were written through identical physical slots;
 10out-of-order reads per metadata-cache mode per GPU gave40exact full-checkpoint
 comparisons. Existing cases also test delayed writes and early-consumer waits.
- 48/48live shared-prefix/divergent-record requests passed atC8 in8.82seconds,
 with13,114input tokens per request; zero storage hits.
- Four older persisted fixtures answered4/4correctly in7.04seconds, but no storage
 hits/load-back occurred: the required-storage coverage gate **failed**.

The isolated GPU tests used the deployed impact1 source. Public packaging differs
only by removal of a redundant duplicate dataclass annotation in0030; no transfer
algorithm changed. The portable public-base build passed full source verification and all110CPU
checks before publication. No performance feature was disabled to pass these tests.

## Limits

These are bounded correctness checks, not proof of every scheduler interleaving.
The6reclamation cases test the real eligibility predicate, not a complete host-full
cancellation/eviction cycle. The unknown-key case tests an ordinary miss, not hash
collision resistance. Separate rank-local GPU runs do not test TP2 collective
agreement. The live runs above do not newly validate disk restoration because
no disk hit occurred; correct answers alone are insufficient for that claim.

An earlier restart paused in filesystem inode waits during the large file-cache
startup scan and recovered without a reboot. It is not evidence of cache tensor
corruption. Operational startup time and full-cache behavior require separate
observation; a green health endpoint alone does not establish cache correctness.
