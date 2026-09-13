#!/usr/bin/env python3
"""Verify the effort profile against the full attested production source tree."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def verify(tree):
    records = json.loads((ROOT / 'provenance/production/runtime-files.json').read_text())
    delta = json.loads((ROOT / 'provenance/chat-effort.json').read_text())['files']
    for path, hashes in delta.items():
        if records[path]['sha256'] != hashes['before']:
            raise ValueError('Production preimage mismatch: ' + path)
        records[path]['sha256'] = hashes['after']
    actual = {str(p.relative_to(tree)) for p in (tree / 'python/sglang').rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if actual != set(records):
        raise ValueError('Full source inventory differs')
    for path, row in records.items():
        if hashlib.sha256((tree / path).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Source mismatch: ' + path)
    return len(records)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps({'profile': 'production-chat-effort', 'verified_source_files': verify(args.tree)}))
