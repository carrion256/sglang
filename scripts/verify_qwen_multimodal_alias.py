#!/usr/bin/env python3
"""Verify the Qwen Flash-Next multimodal alias profile."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from verify_responses_compat import verify as verify_responses
from verify_responses_compat import package_records as responses_package_records

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_records():
    manifest = json.loads((ROOT / "provenance/qwen-multimodal-alias.json").read_text())
    _, _, predecessor = responses_package_records()
    base_inventory = ROOT / "provenance" / manifest["base_inventory"]
    if digest(base_inventory) != manifest["base_inventory_sha256"]:
        raise ValueError("Base inventory digest mismatch")
    inventory = json.loads(base_inventory.read_text())
    if inventory != predecessor:
        raise ValueError("Base inventory differs from Responses verifier")
    for name, hashes in manifest["files"].items():
        if inventory.get(name) != hashes["before"]:
            raise ValueError("Multimodal preimage mismatch: " + name)
        if digest(ROOT / "runtime" / name) != hashes["after"]:
            raise ValueError("Packaged runtime mismatch: " + name)
        inventory[name] = hashes["after"]
    patch = ROOT / "patches" / manifest["patch"]
    if digest(patch) != manifest["patch_sha256"]:
        raise ValueError("Multimodal patch hash mismatch")
    series = (ROOT / "patches/series.qwen-multimodal-alias").read_text().splitlines()
    if series != [
        "0015-qwen-flash-next-effort-alias.patch",
        "0016-responses-namespace-custom-boundary.patch",
        "0017-responses-phase-order.patch",
        manifest["patch"],
    ]:
        raise ValueError("Multimodal patch order differs")
    inventory_path = ROOT / "provenance" / manifest["inventory"]
    recorded_inventory = json.loads(inventory_path.read_text())
    if recorded_inventory != dict(sorted(inventory.items())):
        raise ValueError("Full candidate inventory differs from predecessor chain")
    if digest(inventory_path) != manifest["inventory_sha256"]:
        raise ValueError("Candidate inventory digest mismatch")
    return manifest, inventory


def verify(tree, apply=False, from_image=False):
    manifest, inventory = package_records()
    if apply or from_image:
        verify_responses(tree, from_image=from_image)
        patch = ROOT / "patches" / manifest["patch"]
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=tree, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=tree, check=True)
    actual = {
        str(path.relative_to(tree))
        for path in (tree / "python/sglang").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    if actual != set(inventory):
        raise ValueError("Full source inventory differs")
    for name, expected in inventory.items():
        if digest(tree / name) != expected:
            raise ValueError("Source hash mismatch: " + name)
    return len(inventory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--apply", action="store_true")
    actions.add_argument("--from-image", action="store_true")
    args = parser.parse_args()
    if (args.apply or args.from_image) and args.tree is None:
        parser.error("Application requires --tree")
    count = (
        verify(args.tree.resolve(), args.apply, args.from_image)
        if args.tree
        else len(package_records()[1])
    )
    print(json.dumps({"profile": "qwen-multimodal-alias", "source_files": count,
                      "full_tree_verified": args.tree is not None}))
