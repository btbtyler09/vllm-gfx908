#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Materialize Qwen3.5/Qwen3.6 MTP layers in a local checkpoint.

The public Qwen3.6 GPTQ checkpoints used for MI100 testing include one trained
MTP layer. This utility creates a structurally valid local checkpoint with a
target ``mtp_num_hidden_layers`` value by cloning layer 0 tensors into missing
layer slots, or dropping source layers above the requested count. That lets vLLM
exercise multi-step MTP code paths, but it does not create newly trained MTP
weights.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file


MTP_LAYER_RE = re.compile(r"(^|\.)(mtp\.layers\.)(\d+)(\.)")


def mtp_layer_idx(tensor_name: str) -> int | None:
    match = MTP_LAYER_RE.search(tensor_name)
    if match is None:
        return None
    return int(match.group(3))


def replace_mtp_layer_idx(tensor_name: str, layer_idx: int) -> str:
    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{match.group(2)}{layer_idx}{match.group(4)}"

    return MTP_LAYER_RE.sub(repl, tensor_name, count=1)


def tensor_nbytes(tensor: Any) -> int:
    return int(tensor.numel() * tensor.element_size())


def update_config(config: dict[str, Any], num_mtp_layers: int) -> None:
    if isinstance(config.get("text_config"), dict):
        config["text_config"]["mtp_num_hidden_layers"] = num_mtp_layers
    if "mtp_num_hidden_layers" in config:
        config["mtp_num_hidden_layers"] = num_mtp_layers


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    if copy_files:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a local Qwen3.5/Qwen3.6 checkpoint with additional "
            "materialized MTP layers cloned from layer 0."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument(
        "--num-mtp-layers",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Create exactly N MTP layers. Missing layers are cloned from "
            "mtp.layers.0; source layers >= N are dropped."
        ),
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy unchanged files instead of symlinking them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove the destination directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    dest = args.dest.resolve()

    if args.num_mtp_layers < 1:
        raise SystemExit("--num-mtp-layers must be at least 1")
    if not source.is_dir():
        raise SystemExit(f"source checkpoint does not exist: {source}")
    if dest.exists():
        if not args.force:
            raise SystemExit(f"destination already exists: {dest}")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    index_path = source / "model.safetensors.index.json"
    config_path = source / "config.json"
    if not index_path.exists():
        raise SystemExit(f"missing safetensors index: {index_path}")
    if not config_path.exists():
        raise SystemExit(f"missing config: {config_path}")

    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = dict(index["weight_map"])
    template_keys = [name for name in weight_map if mtp_layer_idx(name) == 0]
    if not template_keys:
        raise SystemExit("checkpoint has no mtp.layers.0 tensors to clone")

    present_layers = sorted(
        idx for idx in {mtp_layer_idx(name) for name in weight_map} if idx is not None
    )
    new_weight_map: dict[str, str] = {}
    dropped_by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        layer_idx = mtp_layer_idx(name)
        if layer_idx is not None and layer_idx >= args.num_mtp_layers:
            dropped_by_shard[shard].append(name)
            continue
        new_weight_map[name] = shard

    additions_by_shard: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for layer_idx in range(1, args.num_mtp_layers):
        for template_key in template_keys:
            new_key = replace_mtp_layer_idx(template_key, layer_idx)
            if new_key in new_weight_map:
                continue
            shard = weight_map[template_key]
            new_weight_map[new_key] = shard
            additions_by_shard[shard].append((new_key, template_key))

    rewritten_shards = set(additions_by_shard) | set(dropped_by_shard)
    shard_files = set(weight_map.values())
    for src in source.iterdir():
        dst = dest / src.name
        if src.name in {"config.json", "model.safetensors.index.json"}:
            continue
        if src.name in rewritten_shards:
            continue
        if src.name in shard_files or src.is_file():
            link_or_copy(src, dst, args.copy_files)

    added_bytes = 0
    dropped_bytes = 0
    for shard in sorted(rewritten_shards):
        additions = additions_by_shard.get(shard, [])
        dropped = set(dropped_by_shard.get(shard, []))
        src_shard = source / shard
        dst_shard = dest / shard
        with safe_open(src_shard, framework="pt", device="cpu") as reader:
            tensors = {}
            for name in reader.keys():
                tensor = reader.get_tensor(name)
                if name in dropped:
                    dropped_bytes += tensor_nbytes(tensor)
                    continue
                tensors[name] = tensor
            metadata = reader.metadata()

        for new_key, template_key in additions:
            tensors[new_key] = tensors[template_key].clone()
            added_bytes += tensor_nbytes(tensors[new_key])

        save_file(tensors, dst_shard, metadata=metadata)

    config = json.loads(config_path.read_text())
    update_config(config, args.num_mtp_layers)
    (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    metadata = dict(index.get("metadata") or {})
    if "total_size" in metadata:
        metadata["total_size"] = (
            int(metadata["total_size"]) + added_bytes - dropped_bytes
        )
    new_index = {
        "metadata": metadata,
        "weight_map": dict(sorted(new_weight_map.items())),
    }
    (dest / "model.safetensors.index.json").write_text(
        json.dumps(new_index, indent=2) + "\n"
    )

    print(f"source: {source}")
    print(f"dest: {dest}")
    print(f"present MTP layers: {present_layers}")
    print(f"target MTP layers: {args.num_mtp_layers}")
    print(f"rewritten shards: {len(rewritten_shards)}")
    print(f"added tensor bytes: {added_bytes}")
    print(f"dropped tensor bytes: {dropped_bytes}")


if __name__ == "__main__":
    main()
