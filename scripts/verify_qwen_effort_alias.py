#!/usr/bin/env python3
"""Apply/verify the candidate over the full attested Chat-effort source tree."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from verify_chat_effort import verify as verify_base

ROOT = Path(__file__).resolve().parents[1]


def verify(tree, apply=False):
    manifest = json.loads((ROOT / 'provenance/qwen-effort-alias.json').read_text())
    patch = ROOT / 'patches' / manifest['patch']
    if hashlib.sha256(patch.read_bytes()).hexdigest() != manifest['patch_sha256']:
        raise ValueError('Alias patch hash mismatch')
    records = json.loads((ROOT / 'provenance/production/runtime-files.json').read_text())
    for rel, hashes in json.loads((ROOT / 'provenance/chat-effort.json').read_text())['files'].items():
        if records[rel]['sha256'] != hashes['before']:
            raise ValueError('Chat preimage mismatch: ' + rel)
        records[rel]['sha256'] = hashes['after']
    for rel, hashes in manifest['files'].items():
        if records[rel]['sha256'] != hashes['before']:
            raise ValueError('Alias preimage mismatch: ' + rel)
        if hashlib.sha256((ROOT / 'runtime' / rel).read_bytes()).hexdigest() != hashes['after']:
            raise ValueError('Packaged runtime mismatch: ' + rel)
        records[rel]['sha256'] = hashes['after']
    if apply:
        verify_base(tree)
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=tree, check=True)
        subprocess.run(['git', 'apply', str(patch)], cwd=tree, check=True)
    actual = {str(p.relative_to(tree)) for p in (tree / 'python/sglang').rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if actual != set(records):
        raise ValueError('Full source inventory differs')
    for rel, row in records.items():
        if hashlib.sha256((tree / rel).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Source mismatch: ' + rel)
    return len(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps({'profile': 'qwen-effort-alias-candidate',
                      'verified_source_files': verify(args.tree.resolve(), args.apply)}))
