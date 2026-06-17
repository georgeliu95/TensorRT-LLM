#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Consolidate parallel ``convert_ckpt_to_4o6_nvfp4.py --skip-untargeted-copy``
worker outputs into a single TRT-LLM-ready NVFP4 checkpoint.

Steps:
  1. Copy sidecar files (READMEs, config.json, etc.) from source.
  2. Move each worker's prefixed shards into the final output dir (atomic rename
     when on the same filesystem).
  3. Provide non-target tensors from source, either by compact shard-wise
     rewrite or by linking/copying whole source shards.
  4. Build a unified ``model.safetensors.index.json`` from worker shard headers
     plus the selected non-target source keys.
  5. Emit ``config.json`` + ``hf_quant_config.json`` with NVFP4 quant metadata
     and proper exclude_modules.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path

DEFAULT_INCLUDE_RE = (
    r".*(?:^|\.)(?:experts|block_sparse_moe\.experts)\.\d+\."
    r"(?:gate_proj|up_proj|down_proj|w1|w2|w3)\.weight$"
)
DEFAULT_SKIP_COPY_RE = r".*(?:\.safetensors|\.index\.json)$"
FLOAT_DTYPES = {"F16", "BF16", "F32", "F64", "F8_E4M3", "F8_E5M2"}


def parse_size(value: str) -> int:
    text = value.strip().lower()
    units = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4}
    for unit, mult in sorted(units.items(), key=lambda kv: -len(kv[0])):
        if text.endswith(unit):
            return int(float(text[:-len(unit)]) * mult)
    return int(text)


def read_safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path} too small to be a safetensors shard")
        header_len = struct.unpack("<Q", prefix)[0]
        return json.loads(f.read(header_len))


def add_index_key(weight_map: dict[str, str], key: str, filename: str) -> None:
    previous = weight_map.get(key)
    if previous is not None:
        raise SystemExit(
            f"duplicate key {key!r} across shards ({previous} and {filename})")
    weight_map[key] = filename


def add_shard_to_index(shard: Path, weight_map: dict[str, str]) -> int:
    header = read_safetensors_header(shard)
    for key in header:
        if key == "__metadata__":
            continue
        add_index_key(weight_map, key, shard.name)
    return shard.stat().st_size


def scan_source_weight_map(src: Path) -> tuple[dict[str, str], dict[str, dict]]:
    """Return (weight_map: key->shard_name, metas: key->header-meta-dict)."""
    weight_map: dict[str, str] = {}
    metas: dict[str, dict] = {}
    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted({weight_map[k] for k in weight_map})
    else:
        shards = [p.name for p in sorted(src.glob("*.safetensors"))]
    for shard_name in shards:
        header = read_safetensors_header(src / shard_name)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            weight_map[key] = shard_name
            metas[key] = meta
    return weight_map, metas


def copy_sidecar_files(src: Path, dst: Path) -> None:
    skip = re.compile(DEFAULT_SKIP_COPY_RE)
    for path in sorted(src.iterdir()):
        if path.name == ".git" or skip.match(path.name):
            continue
        target = dst / path.name
        if target.exists():
            continue
        if path.is_dir():
            shutil.copytree(path, target, ignore=shutil.ignore_patterns(".git"))
        else:
            shutil.copy2(path, target)


def existing_source_excludes(src_config: dict) -> list[str]:
    excludes: list[str] = []
    # K2.5 nests quantization_config under text_config; top-level fallback.
    candidates = []
    if "text_config" in src_config:
        candidates.append(src_config["text_config"].get("quantization_config"))
    candidates.append(src_config.get("quantization_config"))
    for qcfg in candidates:
        if not qcfg:
            continue
        excludes.extend(qcfg.get("ignore") or [])
        excludes.extend(qcfg.get("modules_to_not_convert") or [])
    return excludes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path,
                    help="Source checkpoint dir (e.g. /llm-models/Kimi-K2.5).")
    ap.add_argument("--worker-dirs", required=True, type=Path, nargs="+",
                    help="Per-worker output dirs (each contains prefixed shards).")
    ap.add_argument("--output", required=True, type=Path,
                    help="Final consolidated output dir.")
    ap.add_argument("--max-shard-size", default="2GB",
                    help="Max bytes per non-target bf16 shard (default 2GB).")
    ap.add_argument("--non-target-shard-prefix", default="bf16",
                    help="Filename prefix for non-target tensor shards.")
    ap.add_argument("--non-target-mode",
                    choices=["rewrite", "symlink-source", "hardlink-source",
                             "copy-source"],
                    default="rewrite",
                    help="How to provide non-target source tensors. 'rewrite' "
                         "keeps the output compact and self-contained by "
                         "reading non-target tensors once per source shard. "
                         "'symlink-source' is fastest and smallest but the "
                         "output depends on the read-only source checkpoint. "
                         "'hardlink-source'/'copy-source' reuse whole source "
                         "shards, including unused tensors, so they are "
                         "self-contained only when hardlinks/copies remain "
                         "available and use more disk than rewrite.")
    ap.add_argument("--include-regex", default=DEFAULT_INCLUDE_RE,
                    help="Same regex used by the workers; selects which "
                         "source keys are 'expert' (consumed by workers).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import torch  # safetensors save_file needs torch tensors
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = args.input.resolve()
    out = args.output.resolve()
    start_time = time.monotonic()

    # 1. Set up output dir.
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # 2. Sidecar copy from source.
    print(f"[finalize] copying sidecar files from {src}")
    copy_sidecar_files(src, out)

    # 3. Move worker shards into final dir.
    print(f"[finalize] moving worker shards from {len(args.worker_dirs)} workers")
    moved = 0
    unified_map: dict[str, str] = {}
    total_size = 0
    for wd in args.worker_dirs:
        for shard in sorted(wd.glob("*.safetensors")):
            target = out / shard.name
            if target.exists():
                raise SystemExit(
                    f"shard name collision at {target}; ensure workers used "
                    f"distinct --shard-prefix")
            # Worker parts may live on node-local scratch while the final
            # checkpoint lives on NFS; shutil.move falls back to copy+unlink
            # when os.rename hits EXDEV.
            shutil.move(str(shard), str(target))
            total_size += add_shard_to_index(target, unified_map)
            moved += 1
    print(f"[finalize] moved {moved} worker shards")

    # 4. Identify non-target source keys (those NOT processed by any worker).
    print(f"[finalize] scanning source weight_map")
    src_weight_map, src_metas = scan_source_weight_map(src)
    include_re = re.compile(args.include_regex)

    # An expert INT4 aux key (.weight_packed / .weight_scale / .weight_shape) is
    # "consumed by a worker" if its base ${base}.weight matches the include regex.
    expert_aux_suffixes = (".weight_packed", ".weight_scale", ".weight_shape")
    consumed = set()
    for k in src_weight_map:
        for suffix in expert_aux_suffixes:
            if k.endswith(suffix):
                base_weight = k[:-len(suffix)] + ".weight"
                if include_re.match(base_weight):
                    consumed.add(k)
                break

    non_target_keys = [k for k in sorted(src_weight_map) if k not in consumed]
    print(f"[finalize] source has {len(src_weight_map)} keys; "
          f"consumed-by-workers={len(consumed)}; non-target={len(non_target_keys)}")

    # 5. Provide non-target tensors in the output checkpoint.
    max_bytes = parse_size(args.max_shard_size)
    buf: dict[str, "torch.Tensor"] = {}
    buf_bytes = 0
    bf16_shard_idx = 0

    def flush() -> None:
        nonlocal buf, buf_bytes, bf16_shard_idx, total_size
        if not buf:
            return
        bf16_shard_idx += 1
        fname = f"{args.non_target_shard_prefix}-{bf16_shard_idx:05d}.safetensors"
        save_file(buf, str(out / fname))
        total_size += (out / fname).stat().st_size
        for key in buf:
            add_index_key(unified_map, key, fname)
        buf.clear()
        buf_bytes = 0

    if args.non_target_mode == "rewrite":
        keys_by_shard: dict[str, list[str]] = {}
        for key in non_target_keys:
            keys_by_shard.setdefault(src_weight_map[key], []).append(key)

        print(f"[finalize] rewriting {len(non_target_keys)} non-target tensors "
              f"from {len(keys_by_shard)} source shards")
        copied = 0
        for shard_idx, shard_name in enumerate(sorted(keys_by_shard), start=1):
            keys = keys_by_shard[shard_name]
            with safe_open(str(src / shard_name), framework="pt", device="cpu") as f:
                for key in keys:
                    t = f.get_tensor(key).contiguous()
                    nbytes = int(t.numel() * t.element_size())
                    if buf and buf_bytes + nbytes > max_bytes:
                        flush()
                    buf[key] = t
                    buf_bytes += nbytes
                    copied += 1
            if shard_idx % 8 == 0 or shard_idx == len(keys_by_shard):
                elapsed = time.monotonic() - start_time
                print(
                    f"[finalize] non_target_shards={shard_idx}/{len(keys_by_shard)} "
                    f"non_target_copied={copied}/{len(non_target_keys)} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True)
        flush()
        print(f"[finalize] wrote {bf16_shard_idx} non-target bf16 shards")
    else:
        source_shard_map: dict[str, str] = {}
        source_shards = sorted({src_weight_map[key] for key in non_target_keys})
        print(f"[finalize] {args.non_target_mode}: linking/copying "
              f"{len(source_shards)} source shards for non-target tensors")
        for shard_name in source_shards:
            source_path = src / shard_name
            target_name = f"source-{shard_name}"
            target = out / target_name
            if args.non_target_mode == "symlink-source":
                target.symlink_to(source_path)
            elif args.non_target_mode == "hardlink-source":
                os.link(source_path, target)
            elif args.non_target_mode == "copy-source":
                shutil.copy2(source_path, target)
            else:
                raise AssertionError(args.non_target_mode)
            total_size += target.stat().st_size
            source_shard_map[shard_name] = target_name
        for key in non_target_keys:
            add_index_key(unified_map, key,
                          source_shard_map[src_weight_map[key]])

    index_payload = {
        "metadata": {"total_size": int(total_size)},
        "weight_map": dict(sorted(unified_map.items())),
    }
    with open(out / "model.safetensors.index.json", "w") as f:
        json.dump(index_payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[finalize] unified index: {len(unified_map)} tensors / "
          f"{len(set(unified_map.values()))} shards / total_size={total_size}")

    # 7. Build exclude_modules + emit config.json + hf_quant_config.json.
    src_config = {}
    src_cfg_path = src / "config.json"
    if src_cfg_path.exists():
        with open(src_cfg_path) as f:
            src_config = json.load(f)

    excludes = existing_source_excludes(src_config)
    for k in non_target_keys:
        if not k.endswith(".weight"):
            continue
        meta = src_metas.get(k)
        if (meta and len(meta["shape"]) == 2 and meta["dtype"] in FLOAT_DTYPES):
            excludes.append(k[:-len(".weight")])

    nvfp4_quant_config = {"quant_method": "nvfp4", "group_size": 16}
    config = dict(src_config)
    config["quantization_config"] = dict(nvfp4_quant_config)
    if isinstance(config.get("text_config"), dict):
        config["text_config"] = dict(config["text_config"])
        config["text_config"]["quantization_config"] = dict(nvfp4_quant_config)
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")

    payload = {
        "producer": {
            "name": "llm_4o6.finalize_parallel_4o6_nvfp4",
            "version": "0.1",
        },
        "quantization": {
            "quant_algo": "NVFP4",
            "group_size": 16,
            "exclude_modules": sorted(set(excludes)),
        },
    }
    with open(out / "hf_quant_config.json", "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"[finalize] done: {out}")
    print(f"[finalize] elapsed_s={time.monotonic() - start_time:.1f}")


if __name__ == "__main__":
    main()
