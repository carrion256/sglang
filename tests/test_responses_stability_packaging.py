"""Standalone Responses overlay provenance and scope checks."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('responses_stability_verifier',ROOT/'scripts/verify_responses_stability.py')
VERIFIER=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


class PackagingTest(unittest.TestCase):
    def test_exact_changed_paths(self):
        manifest,inventory=VERIFIER.package_records()
        self.assertEqual(len(inventory),4392)
        self.assertEqual(set(manifest['files']),{
            'python/sglang/srt/entrypoints/openai/serving_responses.py',
            'python/sglang/srt/function_call/qwen3_coder_detector.py',
        })
        for record in manifest['patches']:
            paths={s.removeprefix('+++ b/') for s in (ROOT/'patches'/record['file']).read_text().splitlines() if s.startswith('+++ b/')}
            self.assertEqual(paths,set(record['files']))

    def test_patch_drift_is_rejected(self):
        manifest,_=VERIFIER.package_records()
        target=ROOT/'patches'/manifest['patches'][0]['file']
        original=VERIFIER.digest
        with patch.object(VERIFIER,'digest',side_effect=lambda p:'0'*64 if p==target else original(p)):
            with self.assertRaisesRegex(ValueError,'patch hash mismatch'):
                VERIFIER.package_records()

    def test_pinned_base(self):
        manifest,_=VERIFIER.package_records()
        self.assertIn('FROM '+manifest['base_image'],(ROOT/'Dockerfile.responses-stability').read_text())


if __name__=='__main__': unittest.main()
