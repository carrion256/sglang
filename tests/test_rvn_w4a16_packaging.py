"""Fail-closed checks for the isolated, opt-in rvn-w4a16 profile."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rvn_w4a16_verifier", ROOT / "scripts/verify_rvn_w4a16.py"
)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)

RVN_PATCHES = [
    "0047-rvn-text-config.patch",
    "0048-rvn-w4a16-dispatch.patch",
    "0049-rvn-ple-packed-loader.patch",
    "0050-rvn-ple-hooksite.patch",
    "0051-rvn-ple-offload-eligibility.patch",
    "0052-rvn-marlin-moe-release.patch",
    "0053-rvn-marlin-skip-blockscale-swizzle.patch",
    "0054-rvn-ple-recon-mode-gate.patch",
    "0055-rvn-marlin-repack-cycle-collect.patch",
    "0056-rvn-ple-encoder-version-gate.patch",
    "0057-rvn-nextn-draft-gate.patch",
]
BASE_IMAGE = "localhost/kanadaj-sglang-qwen38fn:hicache-a6d5284"
BASE_IMAGE_ID = (
    "sha256:91cee840799be19916e1ba17ed10a517923f4fc70d54f5abd0247f700d01d77a"
)
LAUNCHER = (ROOT / "deploy/rvn-w4a16-baseline/run_baseline.sh").read_text()


class RvnW4a16PackagingTest(unittest.TestCase):
    def test_series_and_manifest_agree_on_patch_order(self):
        manifest, inventory = VERIFIER.package_records()
        self.assertEqual([row["file"] for row in manifest["patches"]], RVN_PATCHES)
        self.assertEqual(
            (ROOT / "patches/series.rvn-w4a16").read_text().splitlines(), RVN_PATCHES
        )
        self.assertEqual(len(inventory), manifest["source_files_after"])

    def test_created_files_have_no_before_hash(self):
        manifest, _ = VERIFIER.package_records()
        added = [
            name for name, row in manifest["files"].items() if row["before"] is None
        ]
        self.assertEqual(
            sorted(added),
            [
                "python/sglang/srt/models/qwen4_exp_text_adapter.py",
                "python/sglang/srt/models/rvn_ple_storage.py",
            ],
        )
        for row in manifest["patches"]:
            for name, hashes in row["files"].items():
                with self.subTest(patch=row["file"], source=name):
                    if hashes["before"] is None:
                        self.assertIsNone(manifest["files"][name]["before"])

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

    def test_base_image_is_the_pinned_local_deployed_image(self):
        manifest, _ = VERIFIER.package_records()
        self.assertEqual(manifest["base_image"], BASE_IMAGE)
        self.assertEqual(manifest["base_image_id"], BASE_IMAGE_ID)
        from_lines = [
            line
            for line in (ROOT / "Dockerfile.rvn-w4a16").read_text().splitlines()
            if line.startswith("FROM ")
        ]
        self.assertEqual(from_lines, ["FROM " + manifest["base_image"]])

    def test_profile_is_isolated_from_existing_profiles(self):
        for series in sorted((ROOT / "patches").glob("series*")):
            if series.name == "series.rvn-w4a16":
                continue
            with self.subTest(series=series.name):
                self.assertNotIn("rvn", series.read_text().lower())
        for dockerfile in sorted(ROOT.glob("Dockerfile*")):
            if dockerfile.name == "Dockerfile.rvn-w4a16":
                continue
            with self.subTest(dockerfile=dockerfile.name):
                self.assertNotIn("rvn", dockerfile.read_text().lower())
        for record in sorted((ROOT / "provenance").glob("*")):
            if record.name in ("rvn-w4a16.json", "rvn-w4a16-runtime-files.json"):
                continue
            if record.is_file():
                with self.subTest(provenance=record.name):
                    text = record.read_text(errors="replace").lower()
                    self.assertNotIn("rvn", text)
                    for patch in RVN_PATCHES:
                        self.assertNotIn(patch, text)

    def test_launcher_targets_profile_image_with_entrypoint_and_gate(self):
        # finding 1: default must be the patched profile image, never the base
        self.assertIn("IMAGE=${IMAGE:-rvn-w4a16:sim}", LAUNCHER)
        self.assertNotIn("IMAGE=${IMAGE:-" + BASE_IMAGE + "}", LAUNCHER)
        # finding 2: Dockerfile.rvn-w4a16 sets ENTRYPOINT, so bash must override it
        self.assertIn('  --entrypoint /bin/bash\n  "$IMAGE"\n  -lc "$INNER"\n', LAUNCHER)
        # abort gate: runs before cmd=( is built and before any host mkdir
        gate = LAUNCHER.index("PROFILE GATE")
        self.assertLess(gate, LAUNCHER.index("cmd=("))
        self.assertLess(gate, LAUNCHER.index('mkdir -p "$CACHE_ROOT'))
        self.assertIn('--entrypoint python3 "$IMAGE" -B', LAUNCHER)
        self.assertIn("/opt/rvn-w4a16/scripts/verify_rvn_w4a16.py", LAUNCHER)


if __name__ == "__main__":
    unittest.main()
