#!/usr/bin/env python3
"""Verify the invalid generated-token failure profile."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from verify_chat_effort import verify as verify_chat_effort

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_tree_inventory(tree, inventory):
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


def apply_patch(tree, patch):
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=tree, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=tree, check=True)


def package_records():
    manifest = json.loads((ROOT / "provenance/invalid-token-failure.json").read_text())
    base_inventory = ROOT / "provenance/responses-compat-runtime-files.json"
    if digest(base_inventory) != manifest["base_inventory_sha256"]:
        raise ValueError("Base inventory digest mismatch")
    inventory = json.loads(base_inventory.read_text())
    responses = json.loads((ROOT / "provenance/responses-compat.json").read_text())
    alias = json.loads((ROOT / "provenance/qwen-effort-alias.json").read_text())
    if responses["inventory_sha256"] != manifest["base_inventory_sha256"]:
        raise ValueError("Responses inventory identity mismatch")
    for predecessor in (alias, responses):
        predecessor_patch = ROOT / "patches" / predecessor["patch"]
        if digest(predecessor_patch) != predecessor["patch_sha256"]:
            raise ValueError("Predecessor patch hash mismatch")
    predecessor_series = (ROOT / "patches/series.responses-compat").read_text().splitlines()
    if predecessor_series != [alias["patch"], responses["patch"]]:
        raise ValueError("Predecessor patch order differs")
    for name in (
        "python/sglang/srt/entrypoints/openai/protocol.py",
        "python/sglang/srt/entrypoints/openai/responses_compat.py",
        "python/sglang/srt/function_call/qwen3_coder_detector.py",
    ):
        if digest(ROOT / "runtime" / name) != inventory[name]:
            raise ValueError("Packaged predecessor runtime mismatch: " + name)
    for name, hashes in manifest["files"].items():
        if inventory.get(name) != hashes["before"]:
            raise ValueError("Invalid-token preimage mismatch: " + name)
        if digest(ROOT / "runtime.invalid-token-failure" / name) != hashes["after"]:
            raise ValueError("Packaged runtime mismatch: " + name)
        inventory[name] = hashes["after"]
    patch = ROOT / "patches" / manifest["patch"]
    if digest(patch) != manifest["patch_sha256"]:
        raise ValueError("Invalid-token patch hash mismatch")
    series = (ROOT / "patches/series.invalid-token-failure").read_text().splitlines()
    if series != [
        "0015-qwen-flash-next-effort-alias.patch",
        "0016-responses-namespace-custom-boundary.patch",
        manifest["patch"],
    ]:
        raise ValueError("Invalid-token patch order differs")
    encoded = (json.dumps(dict(sorted(inventory.items())), indent=2) + "\n").encode()
    if hashlib.sha256(encoded).hexdigest() != manifest["result_inventory_sha256"]:
        raise ValueError("Result inventory digest mismatch")
    return manifest, inventory


def verify(tree, apply=False, from_image=False):
    manifest, inventory = package_records()
    base_inventory = json.loads(
        (ROOT / "provenance/responses-compat-runtime-files.json").read_text()
    )
    if from_image:
        verify_chat_effort(tree)
        for name in (
            "0015-qwen-flash-next-effort-alias.patch",
            "0016-responses-namespace-custom-boundary.patch",
        ):
            apply_patch(tree, ROOT / "patches" / name)
        verify_tree_inventory(tree, base_inventory)
    elif apply:
        verify_tree_inventory(tree, base_inventory)
    if apply or from_image:
        apply_patch(tree, ROOT / "patches" / manifest["patch"])
    verify_tree_inventory(tree, inventory)
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
    print(
        json.dumps(
            {
                "profile": "invalid-token-failure",
                "source_files": count,
                "full_tree_verified": args.tree is not None,
            }
        )
    )
