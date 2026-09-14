"""Fail-closed packaging tests for the multimodal alias profile."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "qwen_multimodal_verifier", ROOT / "scripts/verify_qwen_multimodal_alias.py"
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class QwenMultimodalPackagingTest(unittest.TestCase):
    def test_manifest_chain_and_runtime_compile(self):
        manifest, inventory = verifier.package_records()
        self.assertEqual(len(inventory), 4392)
        self.assertEqual(manifest["source_files_before"], 4392)
        self.assertEqual(manifest["source_files_after"], 4392)
        self.assertEqual(
            list(manifest["files"]),
            ["python/sglang/srt/multimodal/processors/qwen_vl.py"],
        )
        for name in manifest["files"]:
            compile((ROOT / "runtime" / name).read_bytes(), name, "exec")

    def test_dockerfile_mounts_cumulative_overlay_and_verifies_inventory(self):
        dockerfile = (ROOT / "Dockerfile.qwen-multimodal-alias").read_text()
        copied = {
            line.split()[1].removeprefix("runtime/")
            for line in dockerfile.splitlines()
            if line.startswith("COPY runtime/")
        }
        self.assertEqual(
            copied,
            {
                "python/sglang/srt/entrypoints/openai/serving_chat.py",
                "python/sglang/srt/entrypoints/openai/protocol.py",
                "python/sglang/srt/entrypoints/openai/serving_responses.py",
                "python/sglang/srt/entrypoints/openai/responses_compat.py",
                "python/sglang/srt/function_call/qwen3_coder_detector.py",
                "python/sglang/srt/multimodal/processors/qwen_vl.py",
            },
        )
        self.assertIn("qwen-multimodal-alias-runtime-files.json", dockerfile)
        self.assertIn("assert actual == set(expected)", dockerfile)
        self.assertIn("assert not bad", dockerfile)

    def test_runtime_drift_fails_closed(self):
        original = verifier.digest

        def changed(path):
            if path.name == "qwen_vl.py":
                return "0" * 64
            return original(path)

        with patch.object(verifier, "digest", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "Packaged runtime mismatch"):
                verifier.package_records()

    def test_patch_drift_fails_closed(self):
        original = verifier.digest

        def changed(path):
            if path.name.startswith("0018-"):
                return "0" * 64
            return original(path)

        with patch.object(verifier, "digest", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "patch hash mismatch"):
                verifier.package_records()

    def test_inventory_drift_fails_closed(self):
        original = verifier.digest

        def changed(path):
            if path.name == "qwen-multimodal-alias-runtime-files.json":
                return "0" * 64
            return original(path)

        with patch.object(verifier, "digest", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "inventory digest mismatch"):
                verifier.package_records()
