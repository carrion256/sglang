"""CPU-only attestation checks for the separate Responses candidate profile."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('responses_verifier', ROOT / 'scripts/verify_responses_compat.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class ResponsesPackagingTest(unittest.TestCase):
    def test_full_manifest_chain_and_new_file_count(self):
        manifest, inventory = verifier.package_records()
        self.assertEqual(len(inventory), 4392)
        self.assertEqual([name for name, hashes in manifest['files'].items() if hashes['before'] is None],
                         ['python/sglang/srt/entrypoints/openai/responses_compat.py'])
        base = json.loads((ROOT / 'provenance/production/runtime-files.json').read_text())
        self.assertEqual(len(base), 4391)
        for name in manifest['files']:
            compile((ROOT / 'runtime' / name).read_bytes(), name, 'exec')

    def test_packaged_source_drift_fails_closed(self):
        original = verifier.digest

        def changed(path):
            return '0' * 64 if path.name == 'responses_compat.py' else original(path)

        with patch.object(verifier, 'digest', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'Packaged runtime mismatch'):
                verifier.package_records()

    def test_all_five_mounted_sources_fail_on_drift(self):
        original = verifier.digest
        for filename in ('protocol.py', 'serving_chat.py', 'serving_responses.py',
                         'responses_compat.py', 'qwen3_coder_detector.py'):
            with self.subTest(filename=filename):
                def changed(path):
                    return '0' * 64 if path.name == filename else original(path)
                with patch.object(verifier, 'digest', side_effect=changed):
                    with self.assertRaisesRegex(ValueError, 'Packaged runtime mismatch'):
                        verifier.package_records()

    def test_patch_drift_fails_closed(self):
        original = verifier.digest

        def changed(path):
            return '0' * 64 if path.name.startswith('0016-') else original(path)

        with patch.object(verifier, 'digest', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'patch hash mismatch'):
                verifier.package_records()
