"""Fail-closed packaging tests for invalid generated-token failures."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "invalid_token_verifier", ROOT / "scripts/verify_invalid_token_failure.py"
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class InvalidTokenPackagingTest(unittest.TestCase):
    def test_manifest_chain_and_runtime_compile(self):
        manifest, inventory = verifier.package_records()
        self.assertEqual(len(inventory), 4392)
        self.assertEqual(
            list(manifest["files"]),
            [
                "python/sglang/srt/entrypoints/openai/serving_chat.py",
                "python/sglang/srt/entrypoints/openai/serving_responses.py",
                "python/sglang/srt/managers/schedule_batch.py",
            ],
        )
        for name in manifest["files"]:
            compile(
                (ROOT / "runtime.invalid-token-failure" / name).read_bytes(),
                name,
                "exec",
            )

    def test_runtime_drift_fails_closed(self):
        original = verifier.digest

        def changed(path):
            if path.name == "schedule_batch.py":
                return "0" * 64
            return original(path)

        with patch.object(verifier, "digest", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "Packaged runtime mismatch"):
                verifier.package_records()

    def test_patch_drift_fails_closed(self):
        original = verifier.digest

        def changed(path):
            if path.name.startswith("0019-"):
                return "0" * 64
            return original(path)

        with patch.object(verifier, "digest", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "patch hash mismatch"):
                verifier.package_records()
