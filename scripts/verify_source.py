#!/usr/bin/env python3
"""Offline clean-apply and full source hash verification (no imports/GPU)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(tree, complete=False, profile='combined'):
    prefix = 'provenance/production' if profile == 'production' else 'provenance'
    records = json.loads((ROOT / prefix / 'runtime-files.json').read_text())
    checked = 0
    for rel, row in records.items():
        if complete or row['changed_from_day0']:
            if sha(tree / rel) != row['sha256']:
                raise ValueError('Runtime source hash mismatch: ' + rel)
            checked += 1
    if complete:
        actual = {str(p.relative_to(tree)) for p in (tree / 'python/sglang').rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
        if actual != set(records):
            raise ValueError('Full package path inventory differs')
    return checked

def apply(tree, profile='combined'):
    prefix = 'provenance/production' if profile == 'production' else 'provenance'
    series = 'series.production' if profile == 'production' else 'series'
    manifest = json.loads((ROOT / prefix / 'patches.json').read_text())
    names = (ROOT / 'patches' / series).read_text().splitlines()
    if names != [p['file'] for p in manifest]:
        raise ValueError('Patch order differs from manifest')
    for p in (ROOT / 'upstream-preimages').rglob('*'):
        if p.is_file() and sha(tree / p.relative_to(ROOT / 'upstream-preimages')) != sha(p):
            raise ValueError('Preimage mismatch: ' + p.name)
    for patch in manifest:
        path = ROOT / 'patches' / patch['file']
        if sha(path) != patch['sha256']:
            raise ValueError('Patch hash mismatch')
        subprocess.run(['git', 'apply', '--check', str(path)], cwd=tree, check=True)
        subprocess.run(['git', 'apply', str(path)], cwd=tree, check=True)
    return verify(tree, profile=profile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['combined', 'production'], default='combined')
    parser.add_argument('--tree', type=Path, help='SGLang checkout root')
    parser.add_argument('--apply', action='store_true', help='Modify the supplied tree by applying patches')
    parser.add_argument('--complete', action='store_true', help='Require exact full package inventory')
    args = parser.parse_args()
    if args.tree:
        if args.apply:
            apply(args.tree.resolve(), args.profile)
        print(json.dumps({'verified_source_files': verify(args.tree.resolve(), args.complete, args.profile)}))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / 'tree'
            shutil.copytree(ROOT / 'upstream-preimages', tree)
            count = apply(tree, args.profile)
            compiled = 0
            for p in tree.rglob('*.py'):
                compile(p.read_bytes(), str(p), 'exec')
                compiled += 1
            print(json.dumps({'clean_patch_apply': True, 'verified_changed_paths': count,
                              'syntax_checked_python_files': compiled}))
