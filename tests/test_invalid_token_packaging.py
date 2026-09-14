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
        self.assertEqual(manifest["source_files_before"], 4392)
        self.assertEqual(manifest["source_files_after"], 4392)
        self.assertEqual(
            list(manifest["files"]),
            [
                "python/sglang/srt/entrypoints/openai/serving_chat.py",
                "python/sglang/srt/entrypoints/openai/serving_completions.py",
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

    def test_dockerfile_mounts_cumulative_overlay_and_verifies_inventory(self):
        dockerfile = (ROOT / "Dockerfile.invalid-token-failure").read_text()
        copied = {
            line.split()[1]
            for line in dockerfile.splitlines()
            if line.startswith("COPY runtime")
        }
        self.assertEqual(
            copied,
            {
                "runtime.invalid-token-failure/python/sglang/srt/entrypoints/openai/serving_chat.py",
                "runtime.invalid-token-failure/python/sglang/srt/entrypoints/openai/serving_completions.py",
                "runtime/python/sglang/srt/entrypoints/openai/protocol.py",
                "runtime.invalid-token-failure/python/sglang/srt/entrypoints/openai/serving_responses.py",
                "runtime/python/sglang/srt/entrypoints/openai/responses_compat.py",
                "runtime/python/sglang/srt/function_call/qwen3_coder_detector.py",
                "runtime/python/sglang/srt/multimodal/processors/qwen_vl.py",
                "runtime.invalid-token-failure/python/sglang/srt/managers/schedule_batch.py",
            },
        )
        self.assertIn("invalid-token-failure-runtime-files.json", dockerfile)
        self.assertIn("assert actual == set(expected)", dockerfile)
        self.assertIn("assert not bad", dockerfile)

    def test_runtime_drift_fails_closed(self):
        original = verifier.digest

        def changed(path):
            if path.name == "schedule_batch.py":
                return "0" * 64
            return original(path)

        with patch.object(verifier, "digest", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "Packaged runtime mismatch"):
                verifier.package_records()

    def test_patch_and_inventory_drift_fail_closed(self):
        original = verifier.digest
        cases = (
            ("0019-invalid-generated-token-failure.patch", "patch hash mismatch"),
            ("invalid-token-failure-runtime-files.json", "inventory digest mismatch"),
            ("pr8-final-responses-75.log", "Evidence hash mismatch"),
        )
        for target, message in cases:
            with self.subTest(target=target):
                def changed(path, target=target):
                    if path.name == target:
                        return "0" * 64
                    return original(path)

                with patch.object(verifier, "digest", side_effect=changed):
                    with self.assertRaisesRegex(ValueError, message):
                        verifier.package_records()
