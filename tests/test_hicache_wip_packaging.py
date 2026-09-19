"""Fail-closed checks for the isolated, opt-in HiCache profile."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "hicache_wip_verifier", ROOT / "scripts/verify_hicache_wip.py"
)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


class HiCacheWipPackagingTest(unittest.TestCase):
    def test_patch_chain_and_result_inventory(self):
        manifest, inventory = VERIFIER.package_records()
        self.assertEqual(manifest["status"].split(":", 1)[0], "DRAFT / WIP")
        self.assertEqual(len(inventory), 4395)
        self.assertEqual(
            [row["file"] for row in manifest["patches"]],
            [
                "0020-hicache-ple-state.patch",
                "0021-hicache-file-integrity.patch",
                "0022-hicache-qsa-sidecar.patch",
                "0023-qsa-sparse-gather-memory-safety.patch",
                "0024-router-pdl-bias-order.patch",
                "0025-hicache-load-order.patch",
                "0026-qsa-short-extend-bounds.patch",
                "0027-qsa-paged-prefill.patch",
                "0029-hicache-common-boundary.patch",
                "0030-hicache-selective-diagnostics.patch",
                "0031-hicache-prefill-impact.patch",
                "0032-shared-ple-host-table.patch",
                "0033-prefill-decode-interleaving.patch",
                "0034-hicache-checkpoint-preservation.patch",
                "0035-hicache-prefetch-namespace.patch",
                "0036-hicache-writeback-admission.patch",
            ],
        )

    def test_added_sources_are_explicit(self):
        manifest, _ = VERIFIER.package_records()
        added = [name for name, row in manifest["files"].items() if row["before"] is None]
        self.assertEqual(
            sorted(added), ["python/sglang/srt/mem_cache/cache_diagnostics.py", "python/sglang/srt/mem_cache/checkpoint_coordination.py", "python/sglang/srt/mem_cache/qsa_pool_host.py"]
        )

    def test_patch_paths_match_the_manifest(self):
        manifest, _ = VERIFIER.package_records()
        for row in manifest["patches"]:
            with self.subTest(patch=row["file"]):
                lines = (ROOT / "patches" / row["file"]).read_text().splitlines()
                paths = {
                    line.removeprefix("+++ b/")
                    for line in lines
                    if line.startswith("+++ b/")
                }
                self.assertEqual(paths, set(row["files"]))

    def test_every_patch_fails_closed_on_drift(self):
        manifest, _ = VERIFIER.package_records()
        original = VERIFIER.digest
        for row in manifest["patches"]:
            with self.subTest(patch=row["file"]):
                target = ROOT / "patches" / row["file"]

                def changed(path, *, target=target):
                    return "0" * 64 if path == target else original(path)

                with patch.object(VERIFIER, "digest", side_effect=changed):
                    with self.assertRaisesRegex(ValueError, "patch hash mismatch"):
                        VERIFIER.package_records()

    def test_default_profiles_do_not_include_hicache_wip(self):
        for name in ("series", "series.production", "series.responses-compat"):
            series = (ROOT / "patches" / name).read_text()
            self.assertNotIn("hicache", series.lower())
        dockerfile = (ROOT / "Dockerfile.hicache-wip").read_text()
        self.assertIn("EXPERIMENTAL", dockerfile)
        self.assertIn(
            "sha256:f2859d1ccf824a5295088cf578eba89b0f3eeefff6ae7679c3f5d64af0689458",
            dockerfile,
        )


if __name__ == "__main__":
    unittest.main()
