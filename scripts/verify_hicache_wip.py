#!/usr/bin/env python3
"""Verify the isolated, opt-in HiCache patch profile."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory_hash(inventory: dict[str, str]) -> str:
    payload = (json.dumps(dict(sorted(inventory.items())), indent=2) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def base_inventory() -> dict[str, str]:
    manifest = json.loads((ROOT / "provenance/hicache-wip.json").read_text())
    inventory_path = ROOT / manifest["base_inventory"]
    if digest(inventory_path) != manifest["base_inventory_sha256"]:
        raise ValueError("Cumulative parent inventory digest mismatch")
    return json.loads(inventory_path.read_text())


def apply_patch(tree: Path, patch: Path) -> None:
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=tree, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=tree, check=True)


def package_records() -> tuple[dict, dict[str, str]]:
    manifest_path = ROOT / "provenance/hicache-wip.json"
    manifest = json.loads(manifest_path.read_text())
    base_path = ROOT / manifest["base_inventory"]
    if digest(base_path) != manifest["base_inventory_sha256"]:
        raise ValueError("Base inventory digest mismatch")

    inventory = base_inventory()
    if len(inventory) != manifest["source_files_before"]:
        raise ValueError("Base source count mismatch")

    series = (ROOT / "patches/series.hicache-wip").read_text().splitlines()
    if series != [row["file"] for row in manifest["patches"]]:
        raise ValueError("HiCache patch order differs")

    initial = dict(inventory)
    changed_paths: set[str] = set()
    for patch_record in manifest["patches"]:
        patch = ROOT / "patches" / patch_record["file"]
        if digest(patch) != patch_record["sha256"]:
            raise ValueError("HiCache patch hash mismatch: " + patch.name)
        for name, hashes in patch_record["files"].items():
            if inventory.get(name) != hashes["before"]:
                raise ValueError("HiCache patch transition mismatch: " + name)
            inventory[name] = hashes["after"]
            changed_paths.add(name)

    if changed_paths != set(manifest["files"]):
        raise ValueError("HiCache changed-path manifest differs")
    for name, hashes in manifest["files"].items():
        if initial.get(name) != hashes["before"] or inventory[name] != hashes["after"]:
            raise ValueError("HiCache cumulative file transition mismatch: " + name)
    if len(inventory) != manifest["source_files_after"]:
        raise ValueError("Result source count mismatch")
    if inventory_hash(inventory) != manifest["result_inventory_sha256"]:
        raise ValueError("Result inventory digest mismatch")
    return manifest, inventory


def verify_tree(tree: Path, inventory: dict[str, str]) -> None:
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


def verify(tree: Path, should_apply: bool) -> int:
    manifest, result = package_records()
    if should_apply:
        verify_tree(tree, base_inventory())
        for patch_record in manifest["patches"]:
            apply_patch(tree, ROOT / "patches" / patch_record["file"])
    verify_tree(tree, result)
    return len(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply and args.tree is None:
        parser.error("--apply requires --tree")
    if args.tree is None:
        manifest, _ = package_records()
        changed = len(manifest["files"])
        source_files = manifest["source_files_after"]
        full_tree = False
    else:
        source_files = verify(args.tree.resolve(), args.apply)
        changed = len(package_records()[0]["files"])
        full_tree = True
    print(
        json.dumps(
            {
                "profile": "hicache-wip",
                "status": manifest["status"],
                "clean_patch_apply": bool(args.tree is not None and args.apply),
                "patch_chain_verified": True,
                "changed_source_files": changed,
                "source_files": source_files,
                "full_tree_verified": full_tree,
            }
        )
    )
