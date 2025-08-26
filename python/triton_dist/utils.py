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
import functools
import gzip
import json
import logging
import os
import random
import re
import shutil
import string
import subprocess
import sys
from contextlib import contextmanager, nullcontext, redirect_stdout
from multiprocessing import Pool, cpu_count
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
from functools import wraps

import numpy as np
import packaging.version
import torch


def is_cuda():
    if torch.cuda.is_available() and (torch.version.hip is None):
        return True


def is_hip():
    if torch.cuda.is_available() and (torch.version.hip is not None):
        return True


if is_cuda():
    from cuda import cuda, cudart

    import nvshmem
    import nvshmem.core
    from nvshmem.core.utils import _get_device
elif is_hip():
    from hip import hip
else:
    pass

# Some code from python/flux/util.py in flux project

_TP_GROUP = None


def init_seed(seed=0):
    os.environ["NCCL_DEBUG"] = os.getenv("NCCL_DEBUG", "ERROR")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    # zero empty takes more kernel launch and may hide uninitialized problem. always set to False
    # available since torch 2.2: https://docs.pytorch.org/docs/2.2/deterministic.html
    try:
        torch.utils.deterministic.fill_uninitialized_memory = False
    except Exception:
        logging.warning("torch.utils.fill_uninitialized_memory is available only for torch >=2.2")
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


def init_nvshmem_by_torch_process_group(pg: torch.distributed.ProcessGroup):
    # Extract rank, nranks from process group
    num_ranks = pg.size()
    rank_id = pg.rank()

    # Create an empty uniqueid for all ranks
    broadcast_objects = [nvshmem.core.get_unique_id(empty=rank_id != 0)]
    torch.distributed.broadcast_object_list(broadcast_objects, src=0, group=pg)
    torch.distributed.barrier(group=pg)
    from cuda.core.experimental import Device
    nvshmem.core.init(device=Device(torch.cuda.current_device()), uid=broadcast_objects[0], rank=rank_id,
                      nranks=num_ranks, initializer_method="uid")
    # nvshmem.core.utils._configure_logging("DEBUG")


def nvshmem_create_tensor(shape, dtype) -> torch.Tensor:
    torch.cuda.synchronize()
    tensor = nvshmem.core.tensor(shape, dtype=dtype)
    torch.cuda.synchronize()
    return tensor


def nvshmem_create_tensors(shape, dtype, rank, local_world_size) -> List[torch.Tensor]:

    def _get_peer_tensor(t, peer) -> torch.Tensor:
        # avoid create tensor on the same buf again. nvshmem4py can't handle multiple reference with grace. so we handle it here.
        # https://forums.developer.nvidia.com/t/nvshmem4py-nvshmem-core-finalize-does-not-handle-everything/337979
        if peer == rank:
            return t
        return nvshmem.core.get_peer_tensor(t, peer)

    local_rank = rank % local_world_size
    rank_on_same_node_start = rank - local_rank
    rank_on_same_node_end = rank_on_same_node_start + local_world_size
    torch.cuda.synchronize()
    tensor = nvshmem_create_tensor(shape, dtype=dtype)
    torch.cuda.synchronize()
    return [_get_peer_tensor(tensor, peer) for peer in range(rank_on_same_node_start, rank_on_same_node_end)]


def nvshmem_free_tensor_sync(tensor):
    torch.cuda.synchronize()
    nvshmem.core.free_tensor(tensor)
    torch.cuda.synchronize()


def finalize_distributed():
    if is_cuda():
        nvshmem.core.finalize()
    torch.distributed.destroy_process_group()


class TorchStreamWrapper:

    def __init__(self, pt_stream: torch.cuda.Stream):
        self.pt_stream = pt_stream
        self.handle = pt_stream.cuda_stream

    def __cuda_stream__(self):
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)  # Return format required by CUDA Python


def nvshmem_barrier_all_on_stream(stream: Optional[torch.cuda.Stream] = None):
    stream = stream or torch.cuda.current_stream()
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=TorchStreamWrapper(stream))


# nvshmem4py version 0.1.0 bug:
# wrong parameter order of signal_value and signal_op
# so we reimplement this signal_wait
def nvshmem_signal_wait(signal: torch.Tensor, pe: int, signal_val: int, signal_op: int,
                        stream: torch.cuda.Stream) -> None:
    signal_buf = nvshmem.core.tensor_get_buffer(nvshmem.core.get_peer_tensor(signal, pe))[0]
    # signal_buf = nvshmem.core.tensor_get_buffer(signal)[0]
    user_nvshmem_dev, other_dev = _get_device()

    nvshmem.bindings.signal_wait_until_on_stream(signal_buf._mnff.ptr, signal_op, signal_val, stream.cuda_stream)

    if other_dev is not None:
        other_dev.set_current()


def initialize_distributed(seed=None) -> torch.distributed.ProcessGroup:
    global _TP_GROUP
    assert _TP_GROUP is None, "TP_GROUP has already been initialized"

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    # use all ranks as tp group
    _TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    torch.distributed.barrier(_TP_GROUP)
    _TP_GROUP_GLOO = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="gloo")
    torch.distributed.barrier(_TP_GROUP_GLOO)

    init_seed(seed=seed if seed is not None else RANK)
    init_nvshmem_by_torch_process_group(_TP_GROUP_GLOO)
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
    with open(json_file, "r", encoding="utf-8", errors="replace") as file:
        content = file.read()

        # torch 2.4+ profile with with_stack makes some invalid argument, which makes chrome/edge unhappy
        # use work around here: https://github.com/pytorch/pytorch/issues/121219
        # Decode Unicode escape sequences
        content = content.encode().decode("unicode_escape")

        # Regex to find "name": "<value>"
        def replace_non_ascii_and_quotes(match):
            name = match.group(1)
            visible_printable = "".join(c for c in string.printable if c not in "\t\n\r\x0b\x0c}{")
            cleaned_name = "".join(c if c in visible_printable else "x" for c in name)
            cleaned_name = cleaned_name.replace('"', "y")  # Replace internal quotes
            return f'"name": "{cleaned_name}"'

        # Apply regex to clean names
        cleaned_content = re.sub(
            r'"name": "([\s\S]*?)"(?=, |\}|\s*\})',
            replace_non_ascii_and_quotes,
            content,
            flags=re.DOTALL,
        )

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
        with gzip.open(str(output_json) + ".tar.gz", mode="wt", compresslevel=3) as g:
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
        self.trace_file = (Path("prof") / f"{self.name}" / f"rank{self.group.rank()}.json")

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
            torch.distributed.gather_object(
                trace_content,
                trace_content_list if self.group.rank() == 0 else None,
                dst=0,
                group=self.group,
            )
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


class NvidiaSmiUtil:

    @staticmethod
    def get_nvlink_adjacency_matrix():
        output = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
        lines = [line.strip() for line in output.split("\n") if line.startswith("GPU")]

        device_count = len(lines)
        matrix = [[-1 for _ in range(device_count)] for _ in range(device_count)]

        # 解析每行数据
        for i, line in enumerate(lines):
            parts = line.split()
            for j in range(1, len(parts)):
                if "NV" in parts[j]:
                    matrix[i][j - 1] = 1  # 标记 NVLink 连接

        return matrix

    @staticmethod
    def get_gpu_numa_node(gpu_index=0):
        try:
            # 获取 GPU 的 PCI 总线 ID
            cmd = f"nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader,nounits -i {gpu_index}"
            pci_id = subprocess.check_output(cmd, shell=True).decode().strip()
            pci_address = pci_id.replace("00000000:", "").lower()  # 示例输入 "00000000:17:00.0" → "17:00.0"
            # print(f"gpu_index: {gpu_index} => {pci_id} => {pci_address}")

            # 通过 sysfs 查询 NUMA 节点
            numa_node_path = f"/sys/bus/pci/devices/0000:{pci_address}/numa_node"
            with open(numa_node_path, "r") as f:
                numa_node = int(f.read().strip())

            assert numa_node >= 0
            return numa_node if numa_node >= 0 else 0

        except Exception as e:
            print(f"Error: {e}")
            return -1


_pynvml_initialized = False
_lock = Lock()


def ensure_nvml_initialized():
    global _pynvml_initialized
    if not _pynvml_initialized:
        with _lock:
            if not _pynvml_initialized:
                import pynvml

                pynvml.nvmlInit()
                _pynvml_initialized = True


@functools.lru_cache(maxsize=16)
def get_active_nvlinks_pynvml(gpu_index):
    ensure_nvml_initialized()
    import pynvml

    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    values = pynvml.nvmlDeviceGetFieldValues(handle, [pynvml.NVML_FI_DEV_NVLINK_LINK_COUNT])
    return values[0].value.siVal


def parse_nvml_field_value(fv):
    import pynvml
    if fv.valueType == pynvml.NVML_VALUE_TYPE_DOUBLE:
        return fv.value.dVal
    if fv.valueType == pynvml.NVML_VALUE_TYPE_UNSIGNED_INT:
        return fv.value.uiVal
    if fv.valueType == pynvml.NVML_VALUE_TYPE_UNSIGNED_LONG:
        return fv.value.ulVal
    if fv.valueType == pynvml.NVML_VALUE_TYPE_SIGNED_LONG_LONG:
        return fv.value.llVal
    if fv.valueType == pynvml.NVML_VALUE_TYPE_SIGNED_INT:
        return fv.value.siVal
    if fv.valueType == pynvml.NVML_VALUE_TYPE_UNSIGNED_SHORT:
        return fv.value.usVal

    return "Unsupported type"


@functools.lru_cache(maxsize=16)
def get_nvlink_max_speed_pynvml(gpu_index=0):
    ensure_nvml_initialized()
    import pynvml

    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    values = pynvml.nvmlDeviceGetFieldValues(handle, [pynvml.NVML_FI_DEV_NVLINK_GET_SPEED])
    speed = parse_nvml_field_value(values[0])
    # speed in Mbps but in 1e6, not MB (1024 * 1024)
    speed = speed * 1e6 / 1024 / 1024
    return get_active_nvlinks_pynvml(gpu_index) * speed


@functools.lru_cache(maxsize=16)
def get_nvlink_max_speed_nvsmi(gpu_index=0):
    """Returns total NVLink bandwidth in GB/s for specified GPU"""
    # Run nvidia-smi command
    result = subprocess.run(['nvidia-smi', 'nvlink', '-s', '-i', str(gpu_index)], capture_output=True, text=True,
                            check=True)

    total_speed = 0.0

    # Parse output lines
    for line in result.stdout.split('\n'):
        if 'Link' in line and 'GB/s' in line:
            # Example line: " Link 0: 26.562 GB/s"
            parts = line.split(':')
            speed_str = parts[1].strip().split()[0]
            total_speed += float(speed_str)

    return total_speed


def get_nvlink_max_speed(gpu_index=0):
    try:
        return get_nvlink_max_speed_nvsmi(gpu_index)
    except Exception:
        return get_nvlink_max_speed_pynvml(gpu_index)


@functools.lru_cache()
def has_fullmesh_nvlink_pynvml():
    num_devices = torch.cuda.device_count()

    ensure_nvml_initialized()
    import pynvml

    try:
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(num_devices)]
        for cur_device in range(num_devices):
            cur_handle = handles[cur_device]
            for remote_device in range(num_devices):
                if remote_device == cur_device:
                    continue
                remote_handle = handles[remote_device]
                p2p_status = pynvml.nvmlDeviceGetP2PStatus(cur_handle, remote_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK)
                if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                    return False
        return True
    except pynvml.NVMLError_NotSupported:
        return False


@functools.lru_cache()
def get_numa_node_pynvml(gpu_index):
    ensure_nvml_initialized()
    import pynvml

    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    return pynvml.nvmlDeviceGetNumaNodeId(handle)  # no such symbol for CUDA driver 535.161.08


def calculate_pcie_bandwidth(generation: int, lanes: int) -> tuple:
    """
    Calculate PCIe bandwidth for a given generation and number of lanes.
    Returns (per_direction_gbs, bidirectional_gbs)

    Args:
        generation: PCIe generation (1-6)
        lanes: Number of lanes (x1, x4, x8, x16, etc.)

    Returns:
        Tuple with per-direction and bidirectional bandwidth in GB/s
    """
    # PCIe specifications (transfer rates in GT/s and encoding efficiency)
    pcie_specs = {
        1: {'transfer_rate': 2.5, 'encoding': 0.8},  # 8b/10b encoding
        2: {'transfer_rate': 5.0, 'encoding': 0.8},  # 8b/10b
        3: {'transfer_rate': 8.0, 'encoding': 128 / 130},  # 128b/130b
        4: {'transfer_rate': 16.0, 'encoding': 128 / 130}, 5: {'transfer_rate': 32.0, 'encoding': 128 / 130}, 6:
        {'transfer_rate': 64.0, 'encoding': 242 / 256}  # FLIT encoding
    }

    if generation not in pcie_specs:
        raise ValueError(f"Invalid PCIe generation: {generation}. Supported: 1-6")

    if not isinstance(lanes, int) or lanes <= 0:
        raise ValueError("Lanes must be a positive integer")

    # Get specs for requested generation
    spec = pcie_specs[generation]
    transfer_rate = spec['transfer_rate']  # GT/s per lane
    encoding = spec['encoding']  # Encoding efficiency

    # Calculate bandwidth
    per_direction_gbs = (transfer_rate * encoding * lanes) / 8

    return per_direction_gbs


@functools.lru_cache(maxsize=16)
def get_pcie_link_info_nvsmi(gpu_index=0):
    """Returns (pcie_generation, pcie_width) as integers or (None, None) on error"""
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=pcie.link.gen.gpucurrent,pcie.link.width.current", "--format=csv,noheader", "-i",
        str(gpu_index)
    ], capture_output=True, text=True, check=True)

    # Parse output like "4, 16"
    gen_str, width_str = result.stdout.strip().split(',')
    return int(gen_str.strip()), int(width_str.strip())


@functools.lru_cache()
def get_pcie_link_max_speed_nvsmi(gpu_index=0):
    """Returns the maximum PCIe link speed in GB/s for specified GPU"""
    pcie_gen, lanes = get_pcie_link_info_nvsmi(gpu_index)
    return calculate_pcie_bandwidth(pcie_gen, lanes)


@functools.lru_cache()
def get_pcie_link_max_speed_pynvml(gpu_index=0):
    ensure_nvml_initialized()
    import pynvml
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
    lanes = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
    return calculate_pcie_bandwidth(pcie_gen, lanes)


def get_pcie_link_max_speed(gpu_index):
    try:
        return get_pcie_link_max_speed_nvsmi(gpu_index)
    except Exception:
        return get_pcie_link_max_speed_pynvml(gpu_index)


def get_intranode_max_speed(gpu_index=0, with_scale: bool = False):
    if has_fullmesh_nvlink():
        # 200GB/s => 160GB/s
        _factor = 1.0 if not with_scale else 0.8
        return get_nvlink_max_speed(gpu_index) * _factor
    else:
        # 32GB/s => 22.4GB/s
        _factor = 1.0 if not with_scale else 0.7
        return get_pcie_link_max_speed(gpu_index) * _factor


@functools.lru_cache()
def get_numa_node(gpu_index):
    try:
        return get_numa_node_pynvml(gpu_index)
    except Exception:
        return NvidiaSmiUtil.get_gpu_numa_node(gpu_index)


@functools.lru_cache()
def has_fullmesh_nvlink():
    try:
        return has_fullmesh_nvlink_pynvml()
    except Exception:
        nvlink_matrix = NvidiaSmiUtil.get_nvlink_adjacency_matrix()
        has_nvlink = any([any(x == 1 for x in row) for row in nvlink_matrix])
        _has_fullmesh_nvlink = all([i == j or v == 1 for i, row in enumerate(nvlink_matrix) for j, v in enumerate(row)])
        if has_nvlink and not _has_fullmesh_nvlink:
            warnings.warn(
                "⚠️ found NVLink but not fullmesh NVLink, this may cause undefined behavior, please check your GPU topology"
            )
        return _has_fullmesh_nvlink


@functools.lru_cache()
def get_numa_world_size():
    numa_node = [get_numa_node(n) for n in range(torch.cuda.device_count())]
    numa_node_set = set(numa_node)
    assert len(numa_node_set) <= 2  # TODO(houqi.1993) only 2 NUMA node supported now.
    if len(numa_node_set) == 1:
        return torch.cuda.device_count()

    gpu_count_per_numa = [numa_node.count(x) for x in numa_node_set]
    assert gpu_count_per_numa[0] == gpu_count_per_numa[1]
    return torch.cuda.device_count() // 2


def assert_allclose(x: torch.Tensor, y: torch.Tensor, rtol, atol, verbose=True):
    if not torch.allclose(x, y, rtol=rtol, atol=atol):
        print(f"shape of x: {x.shape}")
        print(f"shape of y: {y.shape}")

        with redirect_stdout(sys.stderr):
            print("x:")
            print(x)
            print("y:")
            print(y)
            print("x-y", x - y)

            diff_loc = torch.isclose(x, y, rtol=rtol, atol=atol) == False  # noqa: E712
            print("x@diff:")
            print(x[diff_loc])
            print("y@diff:")
            print(y[diff_loc])
            num_diff = torch.sum(diff_loc)
            diff_rate = num_diff / y.shape.numel()
            print(f"diff count: {num_diff} ({diff_rate*100:.3f}%), {list(y.shape)}")
            max_diff = torch.max(torch.abs(x - y))
            rtol_abs = rtol * torch.min(torch.abs(y))
            print(f"diff max: {max_diff}, atol: {atol}, rtol_abs: {rtol_abs}")
            diff_indices = (diff_loc == True).nonzero(as_tuple=False)  # noqa: E712
            print(f"diff locations:\n{diff_indices}")
            print("--------------------------------------------------------------\n")
        raise RuntimeError

    if verbose:
        print("✅ all close!")


def bitwise_equal(x: torch.Tensor, y: torch.Tensor):
    return (torch.bitwise_xor(x.view(torch.int8), y.view(torch.int8)) == 0).all().item()


def assert_bitwise_equal(x: torch.Tensor, y: torch.Tensor, verbose=True):
    if not bitwise_equal(x, y):
        print(f"shape of x: {x.shape}")
        print(f"shape of y: {y.shape}")

        with redirect_stdout(sys.stderr):
            print("x:")
            print(x)
            print("y:")
            print(y)
            print("x-y", x - y)

            x = x.view(torch.int8)
            y = y.view(torch.int8)
            print("x as bytes:")
            print(x)
            print("y as bytes:")
            print(y)
            print("x as bytes - y as bytes", x - y)

            diff_loc = torch.bitwise_xor(x.view(torch.int8), y.view(torch.int8)) != 0  # noqa: E712
            print("x as bytes@diff:")
            print(x[diff_loc])
            print("y as bytes@diff:")
            print(y[diff_loc])
            num_diff = torch.sum(diff_loc)
            diff_rate = num_diff / y.shape.numel()
            print(f"diff count: {num_diff} ({diff_rate*100:.3f}%), {list(y.shape)}")
            diff_indices = (diff_loc == True).nonzero(as_tuple=False)  # noqa: E712
            print(f"diff locations:\n{diff_indices}")
            print("--------------------------------------------------------------\n")
        raise RuntimeError

    if verbose:
        print("✅ bitwise equal!")


@functools.lru_cache()
def supports_p2p_native_atomic():
    assert torch.cuda.is_available()
    count = torch.cuda.device_count()
    if count <= 1:
        return True

    # force create CUDA context
    (err, ) = cudart.cudaFree(0)
    CUDA_CHECK(err)

    (err, support) = cudart.cudaDeviceGetP2PAttribute(cudart.cudaDeviceP2PAttr.cudaDevP2PAttrNativeAtomicSupported, 0,
                                                      1)
    CUDA_CHECK(err)
    return support == 1


def requires_p2p_native_atomic(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not supports_p2p_native_atomic():
            warnings.warn(
                f"⚠️ function {fn.__name__} requires P2P native atomic support but you are running on a platform that does not support it. this may cause undefined behavior"
            )
        return fn(*args, **kwargs)

    return wrapper


@functools.lru_cache()
def get_device_max_shared_memory_size(device):
    err, prop = cudart.cudaGetDeviceProperties(device)
    CUDA_CHECK(err)
    return prop.sharedMemPerBlockOptin


# TODO(houqi.1993) nvshmem4py does not support torch.uint64, use torch.int64 instead
# https://forums.developer.nvidia.com/t/nvshmem4py-nvshmem-core-tensor-does-not-support-dtype-torch-uint64-which-is-wired/337929/2
NVSHMEM_SIGNAL_DTYPE = torch.int64


@functools.lru_cache()
def get_nvshmem_home():
    try:
        import nvidia.nvshmem
        return Path(nvidia.nvshmem.__file__).parent
    except Exception:
        return Path(os.getenv("NVSHMEM_HOME"))


@functools.lru_cache()
def has_nvshmemi_bc_built():
    try:
        nvshmem_home = get_nvshmem_home()
        return Path(nvshmem_home / "lib" / "libnvshmemi_device.bc").exists()
    except Exception:
        return False


@functools.lru_cache()
def is_nvshmem_multimem_supported():
    if not is_cuda():
        return False
    # this is a python version of nvshmem nvshmemi_detect_nvls_support
    err, cuda_driver_version = cuda.cuDriverGetVersion()
    CUDA_CHECK(err)

    err, is_multicast_supported = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, 0)
    CUDA_CHECK(err)

    # nvshmem configure support
    if os.getenv("NVSHMEM_DISABLE_CUDA_VMM", "0") == "1" or os.getenv("NVSHMEM_DISABLE_NVLS", "0") == "1":
        return False

    # hardware support
    if torch.cuda.get_device_capability()[0] < 9 or not has_fullmesh_nvlink():
        return False

    return all([
        hasattr(cuda, x) for x in [
            "cuMulticastCreate",
            "cuMulticastBindMem",
            "cuMulticastUnbind",
            "cuMulticastGetGranularity",
            "cuMulticastAddDevice",
        ]
    ])


@functools.lru_cache()
def has_tma():
    cap_major = torch.cuda.get_device_capability()[0]
    return is_cuda() and cap_major >= 9


def requires(condition_func):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            assert condition_func(), f"{condition_func.__name__} is needed for {func.__name__}, please check..."
            return func(*args, **kwargs)

        return wrapper

    return decorator


@functools.lru_cache()
def get_device_property(device_id=0):
    return torch.cuda.get_device_properties(device_id)


def sleep_async(duration_ms: int):
    clock_rate_hz = torch.cuda.clock_rate() * 1e6
    torch.cuda._sleep(int(clock_rate_hz * duration_ms / 1000))


def triton_packed_version():
    import triton
    return packaging.version.Version(triton.__version__)


@functools.lru_cache()
def support_launch_cooperative_grid():
    return triton_packed_version() >= packaging.version.Version("3.3.0")


def launch_cooperative_grid_options():
    # launch_cooperative_grid is enabled since 3.3.0
    if support_launch_cooperative_grid():
        return {"launch_cooperative_grid": True}

    return {}


def cuda_occupancy_max_activate_blocks_per_multiprocessor(triton_func, num_warps, *func_args, **func_kwargs):

    compiled = triton_func.run(*func_args, grid=(1, ), warmup=True, **func_kwargs)
    compiled._init_handles()
    ret = cudart.cudaOccupancyMaxActiveBlocksPerMultiprocessor(compiled.function, num_warps * 32,
                                                               compiled.metadata.shared)
    CUDA_CHECK(ret[0])
    return ret[1]


def cuda_stream_max_priority():
    ret = cudart.cudaDeviceGetStreamPriorityRange()
    CUDA_CHECK(ret[0])
    return ret[2]  # (leastPriority, greatestPriority) -> greatestPriority is max priority
