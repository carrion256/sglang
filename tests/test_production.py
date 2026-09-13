"""CPU regression coverage for exact-production packaging and draft ownership.

The MTP seam executes its actual AST method in an isolated class, not the full
SGLang dependency graph. GPU quantization numerics are a separate historical gate.
"""
import ast
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('production_verify', ROOT / 'scripts/verify_source.py')
assert spec is not None and spec.loader is not None
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)
spec = importlib.util.spec_from_file_location('private_head', ROOT / 'runtime/python/sglang/srt/layers/quantization/private_draft_head.py')
assert spec is not None and spec.loader is not None
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

class ProductionTests(unittest.TestCase):
    def test_production_clean_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / 'tree'
            shutil.copytree(ROOT / 'upstream-preimages', tree)
            self.assertEqual(verify.apply(tree, 'production'), 35)

    def test_only_hf_path_correction_differs_between_profiles(self):
        combined = json.loads((ROOT / 'provenance/runtime-files.json').read_text())
        production = json.loads((ROOT / 'provenance/production/runtime-files.json').read_text())
        self.assertEqual(set(combined), set(production))
        self.assertEqual(len(production), 4391)
        self.assertEqual({p for p in production if production[p]['sha256'] != combined[p]['sha256']}, {
            'python/sglang/srt/models/qwen4_exp.py', 'python/sglang/srt/model_loader/loader.py'})
        prod_patches = json.loads((ROOT / 'provenance/production/patches.json').read_text())
        self.assertEqual({p for row in prod_patches for p in row['paths']},
                         {p for p, row in production.items() if row['changed_from_day0']})

    def test_external_command_exact_tokens_and_environment(self):
        tokens = shlex.split((ROOT / 'docs/production-command.sh').read_text().replace('\\\n', ''), comments=True)
        args = json.loads((ROOT / 'deploy/production/args.json').read_text())
        self.assertEqual(tokens[tokens.index('sglang.launch_server') + 1:], args)
        env = dict(tokens[i + 1].split('=', 1) for i, value in enumerate(tokens) if value == '-e')
        self.assertEqual(env, json.loads((ROOT / 'deploy/production/environment.json').read_text()))
        self.assertEqual(env['SGLANG_PRIVATE_DRAFT_NVFP4_A16'], '1')
        for value in ['--speculative-num-steps=3', '--speculative-eagle-topk=1',
                      '--speculative-num-draft-tokens=4', '--max-mamba-cache-size=512',
                      '--mamba-track-interval=128']:
            self.assertIn(value, args)
        dockerfile = (ROOT / 'Dockerfile.production').read_text()
        self.assertIn('--profile production', dockerfile)
        self.assertNotIn('COPY deploy', dockerfile)
        self.assertNotIn('ENV SGLANG_PRIVATE', dockerfile)

    def test_invalid_gate_rejected_and_default_shared(self):
        target = object()
        with patch.dict('os.environ', {'SGLANG_PRIVATE_DRAFT_NVFP4_A16': 'invalid'}):
            with self.assertRaises(ValueError):
                helper.select_draft_head(target)
        with patch.dict('os.environ', {'SGLANG_PRIVATE_DRAFT_NVFP4_A16': '0'}):
            self.assertIs(helper.select_draft_head(target), target)
        with patch.dict('os.environ', {'SGLANG_PRIVATE_DRAFT_NVFP4_A16': '1'}):
            private = object()
            with patch.object(helper, 'build_private_head', return_value=private) as build:
                self.assertIs(helper.select_draft_head(target), private)
                build.assert_called_once_with(target)

    def test_actual_mtp_method_preserves_tied_path_and_uses_private_for_untied(self):
        source = ast.parse((ROOT / 'runtime/python/sglang/srt/models/qwen4_exp_mtp.py').read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen4ExpForCausalLMMTP')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'set_lm_head_from_target')
        class Base:
            config = SimpleNamespace(tie_word_embeddings=True)
            embed_tokens: object = None
            def set_lm_head_from_target(self, target):
                self.lm_head = target
                return 'super-path'
        node = ast.ClassDef(name='Draft', bases=[ast.Name(id='Base', ctx=ast.Load())], keywords=[], body=[method], decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        scope = {'Base': Base}
        # Compile only the tracked trusted method; no user input is executed.
        exec(compile(module, '<tracked-mtp-method>', 'exec'), scope)
        draft = scope['Draft']()
        target, private, embedding = object(), object(), object()
        draft.embed_tokens = embedding
        with patch.dict('sys.modules', {'sglang.srt.layers.quantization.private_draft_head': helper}):
            with patch.object(helper, 'select_draft_head', return_value=private) as select:
                draft.config = SimpleNamespace(tie_word_embeddings=True)
                self.assertEqual(draft.set_lm_head_from_target(target), 'super-path')
                self.assertIs(draft.lm_head, target)
                select.assert_not_called()
                draft.config.tie_word_embeddings = False
                draft.set_lm_head_from_target(target)
                self.assertIs(draft.lm_head, private)
                self.assertIs(draft.embed_tokens, embedding)
                select.assert_called_once_with(target)
