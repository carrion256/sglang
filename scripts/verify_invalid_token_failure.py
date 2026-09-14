#!/usr/bin/env python3
"""Verify the cumulative invalid generated-token failure profile."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from verify_qwen_multimodal_alias import package_records as multimodal_package_records
from verify_qwen_multimodal_alias import verify as verify_multimodal

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_tree_inventory(tree: Path, inventory: dict[str, str]) -> None:
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


def apply_patch(tree: Path, patch: Path) -> None:
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=tree, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=tree, check=True)


def package_records():
    manifest_path = ROOT / "provenance/invalid-token-failure.json"
    manifest = json.loads(manifest_path.read_text())
    predecessor_manifest, predecessor = multimodal_package_records()

    base_inventory_path = ROOT / "provenance" / manifest["base_inventory"]
    if digest(base_inventory_path) != manifest["base_inventory_sha256"]:
        raise ValueError("Base inventory digest mismatch")
    inventory = json.loads(base_inventory_path.read_text())
    if inventory != predecessor:
        raise ValueError("Base inventory differs from multimodal verifier")
    if manifest["base_inventory_sha256"] != predecessor_manifest["inventory_sha256"]:
        raise ValueError("Predecessor inventory identity mismatch")

    expected_series = [
        "0015-qwen-flash-next-effort-alias.patch",
        "0016-responses-namespace-custom-boundary.patch",
        "0017-responses-phase-order.patch",
        "0018-qwen-flash-next-multimodal-alias.patch",
        manifest["patch"],
    ]
    if manifest["series"] != expected_series:
        raise ValueError("Manifest patch order differs")
    if (ROOT / "patches/series.invalid-token-failure").read_text().splitlines() != expected_series:
        raise ValueError("Invalid-token patch order differs")

    for name, hashes in manifest["files"].items():
        if inventory.get(name) != hashes["before"]:
            raise ValueError("Invalid-token preimage mismatch: " + name)
        runtime_path = ROOT / "runtime.invalid-token-failure" / name
        if digest(runtime_path) != hashes["after"]:
            raise ValueError("Packaged runtime mismatch: " + name)
        inventory[name] = hashes["after"]

    patch = ROOT / "patches" / manifest["patch"]
    if digest(patch) != manifest["patch_sha256"]:
        raise ValueError("Invalid-token patch hash mismatch")

    inventory_path = ROOT / "provenance" / manifest["inventory"]
    recorded_inventory = json.loads(inventory_path.read_text())
    if recorded_inventory != dict(sorted(inventory.items())):
        raise ValueError("Full candidate inventory differs from predecessor chain")
    if digest(inventory_path) != manifest["inventory_sha256"]:
        raise ValueError("Candidate inventory digest mismatch")
    if manifest["source_files_before"] != len(predecessor):
        raise ValueError("Predecessor source count mismatch")
    if manifest["source_files_after"] != len(inventory):
        raise ValueError("Result source count mismatch")

    qwen_path = "python/sglang/srt/multimodal/processors/qwen_vl.py"
    if inventory[qwen_path] != predecessor[qwen_path]:
        raise ValueError("PR7 qwen_vl changed")
    if digest(ROOT / "runtime" / qwen_path) != predecessor[qwen_path]:
        raise ValueError("Packaged PR7 qwen_vl mismatch")

    validation = manifest["validation"]
    expected_counts = {
        "invalid_token_runtime_tests": 15,
        "invalid_token_packaging_tests": 4,
        "responses_tests": 75,
        "effort_tests": 14,
        "multimodal_tests": 4,
        "full_package_tests": 90,
        "dedicated_packaging_tests": 17,
        "exact_image_reconstructions": 2,
    }
    if {name: validation.get(name) for name in expected_counts} != expected_counts:
        raise ValueError("Validation count contract differs")
    for name, expected in validation["logs"].items():
        if digest(ROOT / "provenance" / name) != expected:
            raise ValueError("Evidence hash mismatch: " + name)

    return manifest, inventory


def verify(tree: Path, apply: bool = False, from_image: bool = False) -> int:
    manifest, inventory = package_records()
    if from_image:
        verify_multimodal(tree, from_image=True)
        apply_patch(tree, ROOT / "patches" / manifest["patch"])
    elif apply:
        verify_multimodal(tree)
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
