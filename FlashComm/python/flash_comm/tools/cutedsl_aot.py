################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""Minimal CuTeDSL AOT helper CLI.

Example:

```
torchrun --nproc_per_node=4 -m flash_comm.tools.cutedsl_aot prebuild-ep-overlap \
  --manifest workload.json --aot-dir /tmp/flash_comm_cutedsl_aot
```
"""

import argparse
import json
import os
from collections import Counter
from typing import Any, Iterable

import torch


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dtype(name: str) -> torch.dtype:
    normalised = name.lower().replace("torch.", "")
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[normalised]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype in AOT manifest: {name}") from exc


def _iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if value is None:
        return ()
    if isinstance(value, dict):
        return (value, )
    return tuple(value)


def _selected_ops(manifest: dict[str, Any], override_ops: Iterable[str] | None = None) -> set[str]:
    ops = set(override_ops or manifest.get("ops", ()))
    if ops:
        return ops
    return {
        "dispatch",
        "combine_push",
        "combine_tile_push",
        "topk_reduce",
        "group_gemm",
        "group_gemm_combine",
        "dispatch_group_gemm",
    }


def _op_configs(manifest: dict[str, Any], op_name: str, default: dict[str, Any]) -> Iterable[dict[str, Any]]:
    items = tuple(_iter_dicts(manifest.get(op_name)))
    return items if items else (default, )


def _init_dist_from_torchrun() -> tuple[int, int, int]:
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError("prebuild-ep-overlap must run under torchrun; missing " + ", ".join(missing))
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl")
    return rank, world_size, local_world_size


def _prebuild_ep_overlap(args: argparse.Namespace) -> None:
    manifest = _load_json(args.manifest)
    ep = manifest["ep_overlap"]
    rank, world_size, local_world_size = _init_dist_from_torchrun()

    os.environ["FLASH_COMM_CUTEDSL_AOT_DIR"] = args.aot_dir
    os.environ["FLASH_COMM_CUTEDSL_EXPORT_ON_JIT"] = "1"
    os.environ.setdefault("FLASH_COMM_CUTEDSL_CACHE_MODE", "auto")

    from flash_comm.ep_overlap import EPOverlapKernels

    kernels = EPOverlapKernels(
        max_m=int(ep["max_m"]),
        hidden=int(ep["hidden"]),
        topk=int(ep["topk"]),
        num_experts=int(ep["num_experts"]),
        local_world_size=int(ep.get("local_world_size", local_world_size)),
        ep_group=torch.distributed.group.WORLD,
        capacity=float(ep.get("capacity", 1.2)),
        num_worst_tokens=int(ep.get("num_worst_tokens", -1)),
        expert_alignment=int(ep.get("expert_alignment", 1)),
        check_num_worst_tokens=bool(ep.get("check_num_worst_tokens", False)),
    )
    kernels.cutedsl_cache.aot_dir = args.aot_dir
    kernels.cutedsl_cache.export_on_jit = True

    dtypes = [_dtype(name) for name in manifest.get("dtypes", ["bfloat16"])]
    ops = _selected_ops(manifest, args.ops)
    experts_per_rank = int(ep["num_experts"]) // int(world_size)
    default_num_sm = int(manifest.get("gemm_num_sm", kernels.device_sm_count))
    default_hidden_in = int(manifest.get("hidden_in", ep["hidden"]))
    default_n_out = int(manifest.get("n_out", ep.get("intermediate_size", ep["hidden"])))
    default_hidden = int(manifest.get("hidden", ep["hidden"]))
    default_topk = int(manifest.get("topk", ep["topk"]))
    default_expert_alignment = int(manifest.get("expert_alignment", ep.get("expert_alignment", 1)))
    ctx = kernels.overlap_context

    for dtype in dtypes:
        if "dispatch" in ops:
            ctx.ensure_dispatch_input()
            for item in _op_configs(manifest, "dispatch", {"enable_expert_signals": False}):
                kernels._dispatch_op._compile(
                    dtype,
                    int(item.get("hidden", default_hidden)),
                    int(item.get("experts_per_rank", experts_per_rank)),
                    int(item.get("expert_alignment", default_expert_alignment)),
                    bool(item.get("enable_expert_signals", False)),
                    ctx.dispatch_input_ptrs,
                    ctx.token_src_rank_topk_and_indices_buf,
                    ctx.expert_signals,
                    ctx.expert_signal_counters,
                )
        if "combine_push" in ops:
            ctx.ensure_combine_output()
            for item in _op_configs(manifest, "combine_push", {}):
                kernels._combine_push_op._compile(
                    dtype,
                    int(item.get("hidden", default_hidden)),
                    int(item.get("topk", default_topk)),
                    ctx.token_src_rank_topk_and_indices_buf,
                    ctx.combine_output_ptrs,
                )
        if "combine_tile_push" in ops:
            ctx.ensure_combine_output()
            for item in _op_configs(manifest, "combine_tile_push", {"tile_m": 128, "tile_n": 256}):
                kernels._combine_tile_push_op._compile(
                    dtype,
                    int(item.get("hidden", default_hidden)),
                    int(item.get("topk", default_topk)),
                    int(item.get("tile_m", 128)),
                    int(item.get("tile_n", 256)),
                    ctx.token_src_rank_topk_and_indices_buf,
                    ctx.combine_output_ptrs,
                )
        if "topk_reduce" in ops:
            hidden_sizes = manifest.get("topk_reduce_hidden_sizes", [int(ep["hidden"]), default_n_out])
            for hidden_size in hidden_sizes:
                kernels._topk_reduce_op._compile(dtype, int(hidden_size), int(ep["topk"]), int(ep["num_experts"]))
        if "group_gemm" in ops:
            for item in _op_configs(manifest, "group_gemm", {}):
                kernels._group_gemm_op._compile(
                    dtype,
                    int(item.get("experts_per_rank", experts_per_rank)),
                    int(item.get("n_out", default_n_out)),
                    int(item.get("hidden_in", default_hidden_in)),
                    int(item.get("num_sm", default_num_sm)),
                )
        if "group_gemm_combine" in ops:
            for item in _op_configs(manifest, "group_gemm_combine", {}):
                kernels._group_gemm_combine_op._compile(
                    dtype,
                    int(item.get("experts_per_rank", experts_per_rank)),
                    int(item.get("n_out", default_n_out)),
                    int(item.get("hidden_in", default_hidden_in)),
                    int(item.get("topk", ep["topk"])),
                    int(item.get("num_sm", default_num_sm)),
                    _dtype(item.get("weight_dtype", "float32")),
                    bool(item.get("has_weight", False)),
                )
        if "dispatch_group_gemm" in ops:
            for item in _op_configs(manifest, "dispatch_group_gemm", {}):
                kernels._dispatch_group_gemm_op._compile(
                    dtype,
                    int(item.get("experts_per_rank", experts_per_rank)),
                    int(item.get("n_out", default_n_out)),
                    int(item.get("hidden_in", default_hidden_in)),
                    int(item.get("dispatch_num_stages", 1)),
                    int(item.get("num_sm", default_num_sm)),
                    int(item.get("topk", ep["topk"])),
                    _dtype(item.get("weight_dtype", "float32")),
                    bool(item.get("has_weight", False)),
                )

    if rank == 0:
        print(json.dumps(kernels.cutedsl_cache.stats(), indent=2, sort_keys=True))
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def _record_summary(args: argparse.Namespace) -> None:
    counts: Counter[str] = Counter()
    hashes: set[str] = set()
    with open(args.record, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            key = item.get("key", {})
            counts[key.get("op_name", "unknown")] += 1
            hashes.add(item.get("key_hash", ""))
    print(json.dumps({"unique_keys": len(hashes), "ops": dict(sorted(counts.items()))}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prebuild = sub.add_parser("prebuild-ep-overlap",
                              help="Compile and export ep_overlap CuTeDSL kernels from a manifest")
    prebuild.add_argument("--manifest", required=True)
    prebuild.add_argument("--aot-dir", required=True)
    prebuild.add_argument("--op", dest="ops", action="append", help="Override manifest ops; repeat for multiple ops")
    prebuild.set_defaults(func=_prebuild_ep_overlap)

    summary = sub.add_parser("record-summary", help="Summarise FLASH_COMM_CUTEDSL_RECORD_KEYS JSONL output")
    summary.add_argument("--record", required=True)
    summary.set_defaults(func=_record_summary)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
