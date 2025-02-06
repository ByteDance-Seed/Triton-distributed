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

import datetime
import os
import random

import numpy as np
import torch
from typing import Callable, List, Tuple, Union, Sequence, Optional, Any, Dict
from contextlib import contextmanager, nullcontext
from pathlib import Path
import json
import logging
import gzip
import shutil
from multiprocessing import Pool, cpu_count
import re
import string


def is_cuda():
    if torch.cuda.is_available() and (torch.version.hip is None):
        return True


def is_hip():
    if torch.cuda.is_available() and (torch.version.hip is not None):
        return True


if is_cuda():
    from cuda import cuda, cudart

    import pynvshmem
elif is_hip():
    from hip import hip
else:
    pass

# Some code from python/flux/util.py in flux project

_TP_LOCAL_GROUP = None
_TP_GROUP = None


def init_seed(seed=0):
    os.environ["NCCL_DEBUG"] = os.getenv("NCCL_DEBUG", "ERROR")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_printoptions(precision=2)
    torch.manual_seed(3 + seed)
    torch.cuda.manual_seed_all(3 + seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    np.random.seed(3 + seed)
    random.seed(3 + seed)


def init_nvshmem_by_uniqueid(group: torch.distributed.ProcessGroup):
    rank, nranks = group.rank(), group.size()
    if rank == 0:
        unique_id: bytes = pynvshmem.nvshmemx_get_uniqueid()
        unique_id = torch.frombuffer(unique_id, dtype=torch.uint8).clone()
    else:
        unique_id = torch.empty(128, dtype=torch.uint8)

    if not unique_id.is_cuda:
        tensor_gpu = unique_id.cuda()
        torch.distributed.broadcast(tensor_gpu, src=0, group=group)
        unique_id.copy_(tensor_gpu)
    else:
        torch.distributed.broadcast(unique_id, src=0, group=group)
    torch.cuda.synchronize()

    unique_id = unique_id.numpy().tobytes()
    pynvshmem.nvshmemx_init_attr_with_uniqueid(rank, nranks, unique_id)


def initialize_distributed():
    global _TP_GROUP
    assert _TP_GROUP is None, "TP_GROUP has already been initialized"

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    # use all ranks as tp group
    _TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")

    init_seed(seed=RANK)
    init_nvshmem_by_uniqueid(_TP_GROUP)
    pynvshmem.nvshmem_barrier_all()
    torch.cuda.synchronize()
    return _TP_GROUP


def TP_GROUP() -> torch.distributed.ProcessGroup:
    global _TP_GROUP
    assert _TP_GROUP is not None, "TP_GROUP has not been initialized"
    return _TP_GROUP


@contextmanager
def with_torch_deterministic(mode: bool, warn_only: bool = True):
    old_mode = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(mode, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(old_mode, warn_only=warn_only)


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype.itemsize == 1 and dtype.is_floating_point


def _make_tensor(
    shape: List[Union[int, Callable[[], int]]],
    dtype: torch.dtype,
    init_args: Union[Tuple[float, float], Tuple[int, int]],
    device: str = "cuda",
):
    """
    rand() * scale + bias
    randint(-scale, scale) + bias
    """
    if isinstance(shape, Sequence):
        shape = tuple([x() if isinstance(x, Callable) else x for x in shape])
    elif isinstance(shape, int):
        shape = (shape, )
    elif isinstance(shape, Callable):
        shape = shape()
    else:
        raise ValueError(f"unsupported shape {shape}")

    scale, bias = init_args
    if dtype in [torch.float16, torch.bfloat16, torch.float32]:
        out = (torch.rand(shape, dtype=dtype, device=device) * 2 - 1) * scale + bias
    elif dtype == torch.int8:
        out = torch.randint(-scale, scale, shape, dtype=torch.int8, device=device)
        out = out + bias
    elif is_fp8_dtype(dtype):
        out = (torch.rand(shape, dtype=torch.float16, device=device) * 2 - 1) * scale + bias
        with with_torch_deterministic(False):
            out = out.to(dtype)
    else:
        raise ValueError(f"unsupported dtype {dtype}")

    return out


def generate_data(configs):
    while True:
        yield (_make_tensor(*args) if args else None for args in configs)


def get_torch_prof_ctx(do_prof: bool):
    ctx = (torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
    ) if do_prof else nullcontext())
    return ctx


def perf_func(func, iters, warmup_iters):
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    for n in range(iters + warmup_iters):
        if n == warmup_iters:
            start_event.record()
        output = func()
    stop_event.record()
    start_event.wait()
    stop_event.wait()
    torch.cuda.current_stream().synchronize()
    duration_ms = start_event.elapsed_time(stop_event)
    return output, duration_ms / iters


def dist_print(*args, **kwargs):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    prefix = False
    if "allowed_ranks" in kwargs:
        allowed_ranks = kwargs["allowed_ranks"]
        if isinstance(allowed_ranks, str) and allowed_ranks == "all":
            allowed_ranks = list(range(world_size))

        del kwargs["allowed_ranks"]
    else:
        allowed_ranks = [0]
    if "prefix" in kwargs:
        prefix = kwargs["prefix"]

        del kwargs["prefix"]

    need_sync = False
    if "need_sync" in kwargs:
        need_sync = kwargs["need_sync"]

        del kwargs["need_sync"]

    for allowed in allowed_ranks:
        if need_sync:
            torch.distributed.barrier()
        if rank == allowed:
            if prefix:
                print(f"[rank:{rank}]", end="")
            print(*args, **kwargs)


def CUDA_CHECK(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}: {cuda.cuGetErrorName(err)}")
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cuda Error: {err}: {cudart.cudaGetErrorString(err)}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


def HIP_CHECK(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


def load_json(json_file):
    with open(json_file, 'r', encoding='utf-8', errors='replace') as file:
        content = file.read()

        # torch 2.4+ profile with with_stack makes some invalid argument, which makes chrome/edge unhappy
        # use work around here: https://github.com/pytorch/pytorch/issues/121219
        # Decode Unicode escape sequences
        content = content.encode().decode('unicode_escape')

        # Regex to find "name": "<value>"
        def replace_non_ascii_and_quotes(match):
            name = match.group(1)
            visible_printable = ''.join(c for c in string.printable if c not in '\t\n\r\x0b\x0c}{')
            cleaned_name = ''.join(c if c in visible_printable else 'x' for c in name)
            cleaned_name = cleaned_name.replace('"', 'y')  # Replace internal quotes
            return f'"name": "{cleaned_name}"'

        # Apply regex to clean names
        cleaned_content = re.sub(r'"name": "([\s\S]*?)"(?=, |\}|\s*\})', replace_non_ascii_and_quotes, content,
                                 flags=re.DOTALL)

    return json.loads(cleaned_content, strict=False)


def process_trace_json(json_file):
    RANK_MAX_PID = 100000000

    def _mapping(x, delta):
        if isinstance(x, str):
            return f"{x}_{delta}"
        return x + delta

    def _process_item(item, rank, delta):
        # remapping tid and pid
        item["pid"] = _mapping(item["pid"], delta)
        item["tid"] = _mapping(item["tid"], delta)
        # rename metadata name
        if item["ph"] == "M":
            if item["name"] in ["process_name", "thread_name"]:
                name = item["args"]["name"]
                item["args"]["name"] = f"{name}_rank{rank}"
            elif item["name"] == "process_labels":
                labels = item["args"]["labels"]
                item["args"]["labels"] = f"{labels}_{rank}"

    logging.info(f"process {json_file}")
    trace = load_json(json_file)
    events = trace["traceEvents"]
    rank = trace["distributedInfo"]["rank"]
    delta = rank * RANK_MAX_PID
    [_process_item(x, rank, delta) for x in events]
    return trace


def _merge_json_v1(to_merge_files: List[Path], output_json: Path, compress: bool = True):
    events = []
    for json_file in to_merge_files:
        logging.info(f"process {json_file}")
        trace = process_trace_json(json_file)
        events.extend(trace["traceEvents"])

    logging.info("compress...")
    trace["traceEvents"] = events
    if compress:
        with gzip.open(output_json + ".tar.gz", mode="wt", compresslevel=3) as g:
            json.dump(trace, g)
    else:
        with open(output_json, "w") as f:
            json.dump(trace, f)

    logging.info("done.")


class ParallelJsonDumper:

    def __init__(self, parallel_field: str, chunk_size: int = 5000):
        self.chunk_size = chunk_size
        self.cpu_count = cpu_count()
        self.parallel_field = parallel_field

    def dump(self, data: Dict[str, Any], output_path: Path) -> None:
        """Dump JSON with parallel processing of large parallel_field field"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pvalue = data.pop(self.parallel_field)

        # Split the large list into manageable chunks
        chunks = self._chunkify_list(pvalue)

        # Create processing pool
        with Pool(processes=min(len(chunks), self.cpu_count)) as pool:
            # Process chunks in parallel but maintain order
            chunk_strings = pool.map(self._process_chunk, chunks)

            # Stream results to disk
            self._write_output(data, chunk_strings, output_path)

    def _chunkify_list(self, pvalue: List[Any]) -> List[List[Any]]:
        """Split list into chunks for parallel processing"""
        return [pvalue[i:i + self.chunk_size] for i in range(0, len(pvalue), self.chunk_size)]

    def _process_chunk(self, chunk: List[Any]) -> str:
        """Convert chunk to JSON and strip enclosing brackets"""
        chunk_json = json.dumps(chunk, separators=(",", ":"))
        return chunk_json[1:-1]  # Remove [ and ]

    def _write_output(self, base_data: Dict[str, Any], chunk_strings: List[str], output_path: Path) -> None:
        """Write JSON to disk with proper structure"""
        with open(output_path, "w") as f:
            # Write base data
            f.write(json.dumps(base_data, separators=(",", ":"))[:-1])

            # Append pvalue header
            f.write(f',"{self.parallel_field}":[')

            # Write chunks with proper commas
            for i, chunk_str in enumerate(chunk_strings):
                if i > 0:
                    f.write(",")
                f.write(chunk_str)

            # Close JSON structure
            f.write("]}")


def _merge_json_v2(
    to_merge_files: List[Path],
    output_json: Path,
    compress: bool = True,
):
    events = []
    with Pool(processes=min(len(to_merge_files), cpu_count())) as pool:
        for trace in pool.map(process_trace_json, to_merge_files):
            events.extend(trace["traceEvents"])

    trace["traceEvents"] = events
    logging.info("dump json")
    ParallelJsonDumper("traceEvents", 100000).dump(trace, Path(output_json))

    if compress:
        with gzip.open(output_json.with_suffix(".tar.gz"), mode="wb", compresslevel=3) as g, open(output_json,
                                                                                                  "rb") as f:
            logging.info("compress...")
            g.write(f.read())
        output_json.unlink()
    logging.info("done.")


def _merge_json(
    to_merge_files: List[Path],
    output_json: Path,
    compress: bool = True,
    version: int = 2,
):
    if version == 1:
        _merge_json_v1(to_merge_files, output_json, compress)
    elif version == 2:
        _merge_json_v2(to_merge_files, output_json, compress)


class group_profile:

    def __init__(
        self,
        name: str = None,
        do_prof: bool = True,
        merge_group: bool = True,
        keep_merged_only: bool = True,
        compress: bool = True,
        group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        self.name = name
        self.do_prof = do_prof
        self.profile = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        )
        self.group = group or torch.distributed.group.WORLD
        self.merge_group = merge_group
        self.keep_merged_only = keep_merged_only
        self.compress = compress
        self.trace_file = Path("prof") / f"{self.name}" / f"rank{self.group.rank()}.json"

    def __enter__(self):
        if self.do_prof:
            self.profile.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.do_prof:
            self.profile.__exit__(exc_type, exc_val, exc_tb)
            # export chrome trace
            self.trace_file.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"export chrome trace to {self.trace_file}")
            self.profile.export_chrome_trace(str(self.trace_file))
            if self.merge_group:
                self.merge_all()

    def _collect_all_to_rank0(self):
        # merge all
        if self.merge_group:
            torch.cuda.synchronize()  # wait for all ranks export
            with open(self.trace_file, "rb") as f:
                trace_content = f.read()
            trace_content_list = [None for _ in range(self.group.size())]
            torch.distributed.gather_object(trace_content, trace_content_list if self.group.rank() == 0 else None,
                                            dst=0, group=self.group)
            torch.cuda.synchronize()  # wait for all ranks export
            return trace_content_list if self.group.rank() == 0 else None

    def _merge_all_trace(self, trace_content_list):
        logging.info("merge profiles...")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir).mkdir(exist_ok=True)

            for n in range(self.group.size()):
                with open(Path(tmpdir) / f"trace_{n}.json", "wb") as f:
                    f.write(trace_content_list[n])

            # merge all json
            to_merge_files = [Path(tmpdir) / f"trace_{n}.json" for n in range(self.group.size())]
            merged_json = Path("prof") / f"{self.name}_merged.json"
            _merge_json(to_merge_files, merged_json, self.compress)

    def merge_all(self):
        trace_content_list = self._collect_all_to_rank0()
        if self.group.rank() == 0:
            self._merge_all_trace(trace_content_list)
        self.group.barrier()
        torch.cuda.synchronize()
        outdir = Path("prof") / f"{self.name}"
        if self.keep_merged_only:
            logging.info(f"remove profile directory: {outdir}")
            self.trace_file.unlink(missing_ok=True)
            if torch.cuda.current_device() == 0:  # run once for a device
                shutil.rmtree(self.trace_file.parent, ignore_errors=True)
