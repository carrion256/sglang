#!/usr/bin/env python3
"""Full source attestation of the separate CPU-only Responses candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from verify_qwen_effort_alias import verify as verify_alias

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_records():
    manifest = json.loads((ROOT / 'provenance/responses-compat.json').read_text())
    records = json.loads((ROOT / 'provenance/production/runtime-files.json').read_text())
    for profile in ('chat-effort', 'qwen-effort-alias'):
        delta = json.loads((ROOT / f'provenance/{profile}.json').read_text())
        for name, hashes in delta['files'].items():
            if records[name]['sha256'] != hashes['before']:
                raise ValueError('Predecessor hash mismatch: ' + name)
            records[name]['sha256'] = hashes['after']
    for name, hashes in manifest['files'].items():
        if records.get(name, {}).get('sha256') != hashes['before']:
            raise ValueError('Responses preimage mismatch: ' + name)
        if digest(ROOT / 'runtime' / name) != hashes['after']:
            raise ValueError('Packaged runtime mismatch: ' + name)
        records[name] = {**records.get(name, {}), 'sha256': hashes['after']}
    patch = ROOT / 'patches' / manifest['patch']
    if digest(patch) != manifest['patch_sha256']:
        raise ValueError('Responses patch hash mismatch')
    inventory_path = ROOT / 'provenance/responses-compat-runtime-files.json'
    inventory = json.loads(inventory_path.read_text())
    if inventory != {name: row['sha256'] for name, row in sorted(records.items())}:
        raise ValueError('Full candidate inventory differs from predecessor chain')
    if digest(inventory_path) != manifest['inventory_sha256']:
        raise ValueError('Candidate inventory digest mismatch')
    for name in (
        'python/sglang/srt/entrypoints/openai/protocol.py',
        'python/sglang/srt/entrypoints/openai/serving_chat.py',
        'python/sglang/srt/entrypoints/openai/serving_responses.py',
        'python/sglang/srt/entrypoints/openai/responses_compat.py',
        'python/sglang/srt/function_call/qwen3_coder_detector.py',
    ):
        if digest(ROOT / 'runtime' / name) != inventory[name]:
            raise ValueError('Packaged runtime mismatch: ' + name)
    expected_series = ['0015-qwen-flash-next-effort-alias.patch', manifest['patch']]
    if (ROOT / 'patches/series.responses-compat').read_text().splitlines() != expected_series:
        raise ValueError('Candidate patch order differs')
    return manifest, inventory


def verify(tree, apply=False, from_image=False):
    manifest, inventory = package_records()
    if apply or from_image:
        verify_alias(tree, apply=from_image)
        patch = ROOT / 'patches' / manifest['patch']
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=tree, check=True)
        subprocess.run(['git', 'apply', str(patch)], cwd=tree, check=True)
    actual = {str(path.relative_to(tree)) for path in (tree / 'python/sglang').rglob('*')
              if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
    if actual != set(inventory):
        raise ValueError('Full source inventory differs')
    for name, expected in inventory.items():
        if digest(tree / name) != expected:
            raise ValueError('Source hash mismatch: ' + name)
    return len(inventory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', type=Path)
    parser.add_argument('--tokenizer', type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--apply', action='store_true', help='Apply 0016 to verified alias source')
    actions.add_argument('--from-image', action='store_true', help='Apply 0015 then 0016 to exact image source')
    args = parser.parse_args()
    if args.tokenizer is not None:
        tokenizer = json.loads((ROOT / 'provenance/qwen-effort-alias.json').read_text())['tokenizer']
        for name, expected in tokenizer['files'].items():
            if digest(args.tokenizer / name) != expected:
                raise ValueError('Tokenizer metadata mismatch: ' + name)
    if (args.apply or args.from_image) and args.tree is None:
        parser.error('Application requires --tree')
    count = verify(args.tree.resolve(), args.apply, args.from_image) if args.tree else len(package_records()[1])
    print(json.dumps({'profile': 'responses-compat-candidate', 'source_files': count,
                      'full_tree_verified': args.tree is not None}))
