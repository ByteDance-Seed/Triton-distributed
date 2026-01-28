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
import hashlib
import logging
import os
import glob
import random
import re
import time
import sysconfig
import warnings
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import packaging.version
import torch
import triton
import triton_dist
import dataclasses
import shutil


def is_cuda():
    """Checks if 'nvidia-smi' is available on the system's PATH."""
    if shutil.which("nvidia-smi"):
        return True
    else:
        return False


def is_hip():
    if shutil.which("rocm-smi"):
        return True
    else:
        return False


def is_maca():
    if shutil.which("mx-smi"):
        return True
    else:
        return False


def get_shmem_backend():
    if is_cuda():
        return 'nvshmem'
    elif is_hip():
        backend = os.getenv('TRITON_DIST_SHMEM_BACKEND', 'rocshmem').lower()
        if backend not in ['rocshmem', 'mori_shmem']:
            raise ValueError(f"Invalid SHMEM backend: '{backend}'. "
                             f"Must be 'rocshmem' or 'mori_shmem'. "
                             f"Set via: export TRITON_DIST_SHMEM_BACKEND=<backend>")
        return backend
    else:
        raise Exception("either CUDA or HIP platform is supported")


def is_rocshmem():
    """Check if current backend is ROCSHMEM"""
    return bool(is_hip() and get_shmem_backend() == 'rocshmem')


def is_mori_shmem():
    """Check if current backend is MORI SHMEM"""
    return bool(is_hip() and get_shmem_backend() == 'mori_shmem')


if is_cuda():
    import nvshmem
    import nvshmem.core
    from cuda import cuda, cudart
    from .nv_utils import (
        get_numa_node,
        _get_pynvml_device_id,
        get_max_gpu_clock_rate_in_khz,
        get_current_gpu_clock_rate_in_khz,
        has_fullmesh_nvlink,
    )
elif is_hip():
    from hip import hip
    from .amd_utils import (
        get_numa_node,
        _get_amdsmi_device_index,
        get_max_gpu_clock_rate_in_khz,
        get_current_gpu_clock_rate_in_khz,
    )
    # Dynamically import SHMEM library based on backend selection
    _shmem_backend = get_shmem_backend()
    if _shmem_backend == 'rocshmem':
        import pyrocshmem
    elif _shmem_backend == 'mori_shmem':
        try:
            import mori.shmem as mori_shmem
        except ImportError:
            raise ImportError("mori_shmem Python package not found. "
                              "Please install mori_shmem or use rocshmem backend: "
                              "export TRITON_DIST_SHMEM_BACKEND=rocshmem")
elif is_maca():
    from maca import macart
else:
    raise Exception("either CUDA or HIP platform is supported")

# Some code from python/flux/util.py in flux project

_TRITON_DIST_WORLD: torch.distributed.ProcessGroup = None
_TRITON_DIST_LOCAL_WORLD_SIZE: int = None


def CUDA_CHECK(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}: {cuda.cuGetErrorName(err)}")
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cuda Error: {err}: {cudart.cudaGetErrorString(err)}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


def init_seed(seed=0):
    os.environ["NCCL_DEBUG"] = os.getenv("NCCL_DEBUG", "ERROR")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = os.getenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
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


def init_rocshmem_by_torch_process_group(pg: torch.distributed.ProcessGroup):
    global _TRITON_DIST_WORLD
    assert _TRITON_DIST_WORLD is None, "TRITON_DIST_WORLD has already been initialized"
    _TRITON_DIST_WORLD = pg

    pyrocshmem.init_rocshmem_by_uniqueid(pg)


def init_mori_by_torch_process_group(pg: torch.distributed.ProcessGroup):
    # TODO:: It will be re-implemented later
    global _TRITON_DIST_WORLD
    assert _TRITON_DIST_WORLD is None, "TRITON_DIST_WORLD has already been initialized"
    _TRITON_DIST_WORLD = pg

    rank, nranks = pg.rank(), pg.size()
    if rank == 0:
        buffer: bytes = bytearray(mori_shmem.shmem_get_unique_id())
        unique_id: torch.Tensor = torch.frombuffer(buffer, dtype=torch.uint8).cpu().clone()
    else:
        unique_id: torch.Tensor = torch.empty(128, dtype=torch.uint8, device="cpu")
    # Broadcast unique_id from rank 0 to all ranks
    if not unique_id.is_cuda:
        tensor_gpu = unique_id.cuda()
        torch.distributed.broadcast(tensor_gpu, src=0, group=pg)
        unique_id.copy_(tensor_gpu)
    else:
        torch.distributed.broadcast(unique_id, src=0, group=pg)
    torch.cuda.synchronize()

    # Initialize mori_shmem with the unique_id
    unique_id = unique_id.numpy().tobytes()
    mori_shmem.shmem_init_attr(mori_shmem.MORI_SHMEM_INIT_WITH_UNIQUEID, rank, nranks, unique_id)
    torch.distributed.barrier(group=pg)
    torch.cuda.synchronize()


def init_nvshmem_by_torch_process_group(pg: torch.distributed.ProcessGroup):
    global _TRITON_DIST_WORLD
    assert _TRITON_DIST_WORLD is None, "TRITON_DIST_WORLD has already been initialized"

    _TRITON_DIST_WORLD = pg
    torch.cuda.synchronize()
    # Extract rank, nranks from process group
    num_ranks = pg.size()
    rank_id = pg.rank()

    # Create an empty uniqueid for all ranks
    broadcast_objects = [nvshmem.core.get_unique_id(empty=rank_id != 0)]
    torch.distributed.broadcast_object_list(broadcast_objects, src=torch.distributed.get_global_rank(pg, 0), group=pg)
    torch.distributed.barrier(group=pg)
    from cuda.core.experimental import Device
    nvshmem.core.init(device=Device(torch.cuda.current_device()), uid=broadcast_objects[0], rank=rank_id,
                      nranks=num_ranks, initializer_method="uid")
    # nvshmem.core.utils._configure_logging("DEBUG")


def is_shmem_initialized() -> bool:
    return _TRITON_DIST_WORLD is not None


def nvshmem_create_tensor(shape, dtype) -> torch.Tensor:
    torch.cuda.synchronize()
    # NVSHMEM doesn't support fp8 dtypes, use int8 as storage
    if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        tensor = nvshmem.core.tensor(shape, dtype=torch.int8)
        # View as fp8 type
        tensor = tensor.view(dtype)
    else:
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
    elif is_hip():
        backend = get_shmem_backend()
        if backend == 'rocshmem':
            pyrocshmem.rocshmem_finalize()
        elif backend == 'mori_shmem':
            mori_shmem.shmem_finalize()
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


def rocshmem_barrier_all_on_stream(stream: Optional[torch.cuda.Stream] = None):
    stream = stream.cuda_stream or torch.cuda.current_stream().cuda_stream
    pyrocshmem.rocshmem_barrier_all_on_stream(stream)


def initialize_distributed(seed=None, initialize_shmem: bool = True) -> torch.distributed.ProcessGroup:
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 8))
    global _TRITON_DIST_LOCAL_WORLD_SIZE
    _TRITON_DIST_LOCAL_WORLD_SIZE = LOCAL_WORLD_SIZE
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        device_id=torch.device(LOCAL_RANK),
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    # use all ranks as tp group
    pg: torch.distributed.ProcessGroup = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    torch.distributed.barrier(pg)

    init_seed(seed=seed if seed is not None else RANK)
    if initialize_shmem:
        if is_cuda():
            init_nvshmem_by_torch_process_group(pg)
        elif is_hip():
            backend = get_shmem_backend()
            if backend == 'rocshmem':
                init_rocshmem_by_torch_process_group(pg)
            elif backend == 'mori_shmem':
                init_mori_by_torch_process_group(pg)
            else:
                raise ValueError(f"Invalid SHMEM backend: '{backend}'")
    return pg


def get_triton_dist_world():
    global _TRITON_DIST_WORLD
    if not _TRITON_DIST_WORLD:
        warnings.warn("Using triton_dist but it has not been initialized. "
                      "This will result in Undefined Behavior.")
    return _TRITON_DIST_WORLD


def get_triton_dist_local_world_size():
    global _TRITON_DIST_LOCAL_WORLD_SIZE
    if not _TRITON_DIST_LOCAL_WORLD_SIZE:
        warnings.warn("Using triton_dist but it has not been initialized. "
                      "This will result in Undefined Behavior.")
    return _TRITON_DIST_LOCAL_WORLD_SIZE


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


def HIP_CHECK(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t):
        if err != hip.hipError_t.hipSuccess:
            raise RuntimeError(f"HIP Error: {str(err)}")
    return result


def MACA_CHECK(err):
    if isinstance(err, macart.mcError_t):
        if err != macart.mcError_t.mcSuccess:
            raise RuntimeError(f"MACA Error: {err}: {macart.mcGetErrorString(err)}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


def get_cpu_info_linux():
    vendor = None
    model_name = None
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("vendor_id"):
                vendor = line.split(":", 1)[1].strip()
            elif line.startswith("model name"):
                model_name = line.split(":", 1)[1].strip()
            # stop after we’ve found both
            if vendor and model_name:
                break
    return model_name


@functools.lru_cache()
def get_numa_node_count_in_group(pg: torch.distributed.ProcessGroup):
    nranks = pg.size()
    numa_node = get_numa_node(torch.cuda.current_device())
    numa_nodes = [-1 for _ in range(nranks)]
    torch.distributed.all_gather_object(numa_nodes, numa_node, group=pg)
    # assert GPU NUMA node is symmetric
    nnodes = len(set(numa_nodes))
    # only optimize for NUMA nodes = 2 with ranks evenly distributed
    if nnodes != 2:
        return 1

    numa_nodes_low = numa_nodes[:nranks // 2]
    numa_nodes_high = numa_nodes[nranks // 2:]
    if len(set(numa_nodes_low)) == 1 and len(set(numa_nodes_high)) == 1:
        return 2
    return 1


@functools.lru_cache()
def get_group_numa_world_size(pg: torch.distributed.ProcessGroup):
    """
    allgather all ranks in the process group and get the NUMA world size
    """
    return pg.size() // get_numa_node_count_in_group(pg)


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
def get_device_max_shared_memory_size(device_id):
    err, prop = cudart.cudaGetDeviceProperties(device_id)
    CUDA_CHECK(err)
    return prop.sharedMemPerBlockOptin


# TODO(houqi.1993) nvshmem4py does not support torch.uint64, use torch.int64 instead
# https://forums.developer.nvidia.com/t/nvshmem4py-nvshmem-core-tensor-does-not-support-dtype-torch-uint64-which-is-wired/337929/2
NVSHMEM_SIGNAL_DTYPE = torch.int64


@functools.lru_cache()
def get_nvshmem_home() -> Path:
    if (nvshmem_home := os.getenv("NVSHMEM_HOME")) is not None:
        return Path(nvshmem_home)

    try:
        import nvidia.nvshmem

        return Path(nvidia.nvshmem.__path__[0])
    except Exception:
        pass


@functools.lru_cache()
def get_nvshmem_version():
    header_path = get_nvshmem_home() / "include" / "non_abi" / "nvshmem_version.h"
    version_macros = {
        "NVSHMEM_VENDOR_MAJOR_VERSION": None,
        "NVSHMEM_VENDOR_MINOR_VERSION": None,
        "NVSHMEM_VENDOR_PATCH_VERSION": None,
        "NVSHMEM_VENDOR_PACKAGE_VERSION": None,
    }
    pattern = re.compile(r"#define\s+(\w+)\s+(\d+)")

    header_file = Path(header_path)
    if not header_file.exists():
        raise FileNotFoundError(f"{header_path} not found")

    with open(header_file, "r") as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                name, value = m.groups()
                if name in version_macros:
                    version_macros[name] = int(value)

    if None in version_macros.values():
        raise RuntimeError("Failed to parse all NVSHMEM version components")

    return "{major}.{minor}.{patch}-{pkg}".format(
        major=version_macros["NVSHMEM_VENDOR_MAJOR_VERSION"],
        minor=version_macros["NVSHMEM_VENDOR_MINOR_VERSION"],
        patch=version_macros["NVSHMEM_VENDOR_PATCH_VERSION"],
        pkg=version_macros["NVSHMEM_VENDOR_PACKAGE_VERSION"],
    )


def get_nvshmem_hash():
    nvshmem_home = get_nvshmem_home()
    nvshmem_lib = nvshmem_home / "lib" / "libnvshmem_device.a"
    with open(nvshmem_lib, "rb") as f:
        nvshmem_hash = hashlib.sha256(f.read(1024 * 1024)).hexdigest()
    return nvshmem_hash


def get_rocshmem_home():
    return os.getenv("ROCSHMEM_HOME",
                     Path(__file__).parent.parent.parent / "shmem" / "rocshmem_bind" / "rocshmem_build" / "install")


@functools.lru_cache
def get_rocshmem_version():
    return "unknown"


def _get_rocshmem_libdevice():
    if os.getenv("ROCSHMEM_HOME") is not None:
        rocshmem_lib_dir = Path(os.getenv("ROCSHMEM_HOME")) / "lib"
    else:
        rocshmem_lib_dir = Path(triton_dist.__path__[0]) / "tools" / "compile"
    return rocshmem_lib_dir / "librocshmem_device.bc"


def get_rocshmem_hash():
    rocshmem_libdevice = _get_rocshmem_libdevice()
    with open(rocshmem_libdevice, "rb") as f:
        rocshmem_hash = hashlib.sha256(f.read(1024 * 1024)).hexdigest()
    return rocshmem_hash


def _get_mxshmem_libdevice():
    mxshmem_device_bc_path_user_specify = os.getenv("TRITON_MXSHMEM_LIBDEVICE_PATH", None)
    if mxshmem_device_bc_path_user_specify is not None:
        mxshmem_lib_dir = Path(mxshmem_device_bc_path_user_specify)
    else:
        try:
            import triton.backends.metax
            mxshmem_lib_dir = Path(triton.backends.metax.__path__[0]) / "lib"
        except Exception:
            pass
    return mxshmem_lib_dir / "libmxshmem_device.bc"


def get_mxshmem_hash():
    mxshmem_libdevice = _get_mxshmem_libdevice()
    with open(mxshmem_libdevice, "rb") as f:
        mxshmem_hash = hashlib.sha256(f.read(1024 * 1024)).hexdigest()
    return mxshmem_hash


# Note: MORI SHMEM currently only requires a single device BC file (_get_mori_shmem_libdevice()).
# get_mori_home() is kept for future compatibility but not currently used.
@functools.lru_cache()
def get_mori_home() -> Path:
    if (mori_home := os.getenv("MORI_HOME")) is not None:
        return Path(mori_home)

    # Note: This path does not exist yet. MORI is installed via pip and only produces BC file.
    return Path(__file__).parent.parent.parent / "shmem" / "mori_bind" / "mori_build" / "install"


@functools.lru_cache
def get_mori_version():
    return "unknown"


def _get_mori_shmem_libdevice():
    if os.getenv("MORI_HOME") is not None:
        mori_lib_dir = Path(os.getenv("MORI_HOME")) / "lib"
    else:
        mori_lib_dir = Path(triton_dist.__path__[0]) / "tools" / "compile"
    return mori_lib_dir / "libmori_shmem_device.bc"


def get_mori_shmem_hash():
    mori_libdevice = _get_mori_shmem_libdevice()
    with open(mori_libdevice, "rb") as f:
        mori_hash = hashlib.sha256(f.read(1024 * 1024)).hexdigest()
    return mori_hash


@functools.lru_cache()
def get_shmem_version():
    if is_cuda():
        return get_nvshmem_version()
    elif is_hip():
        backend = get_shmem_backend()
        if backend == 'rocshmem':
            return get_rocshmem_version()
        elif backend == 'mori_shmem':
            return get_mori_version()
    return "unknown"


@functools.lru_cache()
def get_shmem_hash():
    if is_cuda():
        return get_nvshmem_hash()
    elif is_hip():
        backend = get_shmem_backend()
        if backend == 'rocshmem':
            return get_rocshmem_hash()
        elif backend == 'mori_shmem':
            return get_mori_shmem_hash()
    elif is_maca():
        return get_mxshmem_hash()
    return "unknown"


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
    if is_multicast_supported == 0:
        return False

    # nvshmem configure support
    if get_bool_env("NVSHMEM_DISABLE_CUDA_VMM", False) or get_bool_env("NVSHMEM_DISABLE_NVLS", False):
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


def sleep_async(duration_ms: float):
    """  sleep for duration_ms in CUDA kernel """
    clock_rate_hz = get_max_gpu_clock_rate_in_khz(0) * 1e3
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


@functools.lru_cache()
def torch_stream_max_priority():
    try:
        _, high = torch.cuda.current_stream().priority_range()
    except Exception:
        high = -1
    return high


@functools.lru_cache()
def triton_dist_key():

    TRITON_DIST_PATH = triton_dist.__path__[0]
    contents = []
    # compiler
    subdirs = ["kernels", "mega_triton_kernel", "language"]
    for subdir in subdirs:
        path = os.path.join(TRITON_DIST_PATH, subdir)
        # use pkgutil.walk_package is more accurate but requires that all submodules can be loaded.
        # which may not be satisfied when torch version is low and may conflict with transformers or so
        for pyfile in glob.glob(os.path.join(path, "*.py")):
            with open(pyfile, "rb") as f:
                contents += [hashlib.sha256(f.read()).hexdigest()]

    # backend
    libtriton_hash = hashlib.sha256()
    ext = sysconfig.get_config_var("EXT_SUFFIX").split(".")[-1]
    libs = ["libtriton", "libtriton_distributed"]
    for lib in libs:
        with open(os.path.join(triton.__path__[0], "_C", f"{lib}.{ext}"), "rb") as f:
            while True:
                chunk = f.read(1024**2)
                if not chunk:
                    break
                libtriton_hash.update(chunk)
    contents.append(libtriton_hash.hexdigest())

    # TODO(houqi.1993)
    __version__ = "0.2.0"
    return f"{__version__}" + "-".join(contents)


def get_bool_env(env, default_value):
    env_value = os.getenv(env)
    if env_value is None:
        return default_value
    env_value = env_value.lower()
    try:
        assert env_value in ["on", "off", "1", "0", "true", "false"]
    except Exception:
        print(f"env {env} is not bool, use default value {default_value}")
        return default_value
    return env_value in ["on", "1", "true"]


def get_int_env(env, default_value):
    env_value = os.getenv(env)
    if env_value is None:
        return default_value
    try:
        return int(env_value)
    except Exception:
        print(f"env {env} is not int, use default value {default_value}")
        return default_value


@functools.lru_cache()
def _is_cuda_launch_blocking():
    if is_cuda():
        return get_bool_env("CUDA_LAUNCH_BLOCKING", False)

    # https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/how_to_guides/debugging.html#summary-of-environment-variables-in-hip
    for env in ["AMD_SERIALIZE_COPY", "AMD_SERIALIZE_KERNEL"]:
        if get_int_env(env, None) is not None:
            return False
    return get_bool_env("HIP_LAUNCH_BLOCKING", False)


def warn_if_cuda_launch_blocking():
    if _is_cuda_launch_blocking():
        launch_blocking_env = "" if is_cuda() else "HIP_LAUNCH_BLOCKING/AMD_SERIALIZE_COPY/AMD_SERIALIZE_KERNEL"
        warnings.warn(f"{launch_blocking_env} is set, which may cause performance issue. "
                      f"Please set {launch_blocking_env} to default value to disable this warning.")


def barrier_async(pg: torch.distributed.ProcessGroup):
    x = torch.empty((1, ), device="cuda", dtype=torch.int32, requires_grad=False)
    pg.allreduce(x)


# in case sometimes GPU is in trouble and always drop frequency
_LAST_MAX_CLOCK_RATE_KHz = None


def get_smi_device_index(device_id):
    if is_cuda():
        return _get_pynvml_device_id(device_id)
    else:
        return _get_amdsmi_device_index(device_id)


def wait_until_max_gpu_clock_or_warning(device_id=None, timeout_sec=10):
    # TODO(houqi.1993) if GPU is not in performance mode, when no workload on GPU, clock may get even lower after waiting.
    # so by default don't wait until GPU to max clock. Make sure you set the GPU to performance mode and then export TRITON_DIST_WAIT_GPU_CLOCK=True.
    if not get_bool_env("TRITON_DIST_WAIT_GPU_CLOCK", False):
        return True

    device_id = get_smi_device_index(device_id)
    end_time = time.time() + timeout_sec
    max_clock_rate = get_max_gpu_clock_rate_in_khz(device_id)
    interval = 0.1
    global _LAST_MAX_CLOCK_RATE_KHz
    if _LAST_MAX_CLOCK_RATE_KHz is not None:
        max_clock_rate = _LAST_MAX_CLOCK_RATE_KHz

    while time.time() < end_time:
        current_clock_rate = get_current_gpu_clock_rate_in_khz(device_id)
        if current_clock_rate >= max_clock_rate:
            _LAST_MAX_CLOCK_RATE_KHz = current_clock_rate
            return True
        time.sleep(interval)
        if interval < 1:
            interval *= 2
    print(
        f"warning: clock rate {current_clock_rate} not reached max clock rate {max_clock_rate} in {timeout_sec} seconds"
    )
    _LAST_MAX_CLOCK_RATE_KHz = current_clock_rate
    return False


def _torch_has_fp8():
    return getattr(torch, "float8_e4m3fn", None) and getattr(torch, "float8_e5m2", None)


def rand_tensor(shape, dtype: torch.dtype, device: torch.device | int | str = "cuda"):
    """
    for float types, return uniform distribution [-1, 1]
    for int types, return uniform distribution in [int_type_min, int_type_max]
    """
    if dtype in [torch.float16, torch.bfloat16, torch.float]:
        return torch.rand(shape, dtype=dtype, device=device) * 2 - 1

    if _torch_has_fp8():
        if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            return (torch.rand(shape, dtype=torch.bfloat16, device=device) * 2 - 1).to(dtype)

    if dtype == torch.int8:
        return torch.randint(-2**7, 2**7, shape, dtype=dtype, device=device)
    if dtype == torch.int16:
        return torch.randint(-2**15, 2**15, shape, dtype=dtype, device=device)
    if dtype == torch.int32:
        return torch.randint(-2**31, 2**31, shape, dtype=dtype, device=device)
    if dtype == torch.uint8:
        return torch.randint(0, 2**8, shape, dtype=dtype, device=device)
    if dtype == torch.uint16:
        return torch.randint(0, 2**16, shape, dtype=dtype, device=device)
    if dtype == torch.uint32:
        return torch.randint(0, 2**32, shape, dtype=dtype, device=device)

    raise Exception(f"rand for {dtype} not implemented")


################################################################################
"""
Lazy tensor allocation utilities.

This module provides a lazy allocation mechanism for tensors, allowing users to:
1. Create tensor specifications without actually allocating memory
2. Query the total memory requirement before allocation
3. Materialize all tensors at once when ready

This is particularly useful for nvshmem tensors where knowing the total memory
requirement upfront is important.

Usage:
    from ditron_kernel.utils.lazy_allocator import LazyAllocator, LazyTensor
    
    # Create allocator with custom tensor creation function
    allocator = LazyAllocator(
        create_tensor_fn=nvshmem_create_tensor,
        lazy=True
    )
    
    # Create lazy tensors (no actual allocation yet)
    buf1 = allocator.create_tensor("buffer1", [1024, 256], torch.bfloat16)
    buf2 = allocator.create_tensor("buffer2", [512], torch.int32, fill_value=0)
    
    # Query total size before allocation
    print(f"Total memory needed: {allocator.get_total_size_gb():.2f} GB")
    
    # Actually allocate all tensors
    allocator.sync()
    
    # Now tensors can be used normally
    buf1[0] = 1.0
"""


def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size in bytes for a given dtype."""
    dtype_sizes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.uint64: 8,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1,
        torch.float64: 8,
    }
    return dtype_sizes.get(dtype, torch.tensor([], dtype=dtype).element_size())


@dataclasses.dataclass
class LazyTensorSpec:
    """Specification for a lazy tensor."""
    name: str
    shape: List[int]
    dtype: torch.dtype
    fill_value: Optional[float] = None  # Value to fill after allocation, None means no fill

    @property
    def numel(self) -> int:
        """Total number of elements."""
        result = 1
        for s in self.shape:
            result *= s
        return result

    @property
    def nbytes(self) -> int:
        """Total size in bytes."""
        return self.numel * get_dtype_size(self.dtype)


class LazyTensor:
    """
    A lazy tensor wrapper that delays allocation until sync() is called.
    
    Before sync(): records the shape/dtype, allows querying size
    After sync(): behaves like a normal tensor
    
    This class implements __torch_function__ to be compatible with PyTorch
    operations like torch.empty_like(), torch.zeros_like(), etc.
    """

    def __init__(self, spec: LazyTensorSpec, allocator: 'LazyAllocator'):
        self._spec = spec
        self._allocator = allocator
        self._tensor: Optional[torch.Tensor] = None

    @property
    def is_materialized(self) -> bool:
        """Check if the tensor has been allocated."""
        return self._tensor is not None

    @property
    def spec(self) -> LazyTensorSpec:
        """Get the tensor specification."""
        return self._spec

    @property
    def shape(self) -> torch.Size:
        """Get the shape (available before materialization)."""
        if self._tensor is not None:
            return self._tensor.shape
        return torch.Size(self._spec.shape)

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype (available before materialization)."""
        if self._tensor is not None:
            return self._tensor.dtype
        return self._spec.dtype

    @property
    def nbytes(self) -> int:
        """Get the size in bytes (available before materialization)."""
        return self._spec.nbytes

    def size(self, dim: Optional[int] = None) -> Union[torch.Size, int]:
        """Get size like torch.Tensor.size()."""
        if dim is None:
            return self.shape
        return self.shape[dim]

    def fill_(self, value: float) -> 'LazyTensor':
        """
        Fill the tensor with a value.
        If not materialized, the fill is deferred until sync().
        """
        if self._tensor is not None:
            self._tensor.fill_(value)
        else:
            self._spec.fill_value = value
        return self

    def zero_(self) -> 'LazyTensor':
        """Zero the tensor."""
        return self.fill_(0)

    def _materialize(self, tensor: torch.Tensor) -> None:
        """Called by allocator to set the actual tensor."""
        self._tensor = tensor
        # Apply pending fill if any
        if self._spec.fill_value is not None:
            self._tensor.fill_(self._spec.fill_value)

    def _ensure_materialized(self) -> None:
        """Ensure the tensor is materialized before access."""
        if self._tensor is None:
            raise RuntimeError(f"LazyTensor '{self._spec.name}' has not been materialized. "
                               f"Call allocator.sync() or allocator.materialize() first.")

    @property
    def tensor(self) -> torch.Tensor:
        """Get the underlying tensor (must be materialized)."""
        self._ensure_materialized()
        return self._tensor

    def get_underlying_tensor(self) -> Optional[torch.Tensor]:
        """Get the underlying tensor, or None if not materialized."""
        return self._tensor

    # ==================== Tensor-like operations ====================

    def __getitem__(self, key):
        self._ensure_materialized()
        return self._tensor[key]

    def __setitem__(self, key, value):
        self._ensure_materialized()
        self._tensor[key] = value

    def copy_(self, src) -> 'LazyTensor':
        self._ensure_materialized()
        self._tensor.copy_(src)
        return self

    def data_ptr(self) -> int:
        self._ensure_materialized()
        return self._tensor.data_ptr()

    def is_contiguous(self) -> bool:
        if self._tensor is not None:
            return self._tensor.is_contiguous()
        return True  # Assume contiguous before materialization

    def view(self, *args):
        self._ensure_materialized()
        return self._tensor.view(*args)

    def reshape(self, *args):
        self._ensure_materialized()
        return self._tensor.reshape(*args)

    def contiguous(self):
        self._ensure_materialized()
        return self._tensor.contiguous()

    def numel(self) -> int:
        if self._tensor is not None:
            return self._tensor.numel()
        return self._spec.numel

    def dim(self) -> int:
        if self._tensor is not None:
            return self._tensor.dim()
        return len(self._spec.shape)

    def stride(self, dim: Optional[int] = None):
        self._ensure_materialized()
        if dim is None:
            return self._tensor.stride()
        return self._tensor.stride(dim)

    @property
    def device(self) -> torch.device:
        if self._tensor is not None:
            return self._tensor.device
        return torch.device("cuda")  # Default to CUDA

    @property
    def data(self):
        self._ensure_materialized()
        return self._tensor.data

    def __getattr__(self, name: str):
        """Forward attribute access to underlying tensor for compatibility."""
        # Avoid infinite recursion for internal attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Check if materialized
        if self._tensor is None:
            raise RuntimeError(f"LazyTensor '{self._spec.name}' has not been materialized. "
                               f"Cannot access attribute '{name}'. Call allocator.sync() first.")

        # Forward to underlying tensor
        return getattr(self._tensor, name)

    def __repr__(self) -> str:
        if self._tensor is not None:
            return f"LazyTensor(materialized, {self._spec.name}, shape={list(self._tensor.shape)}, dtype={self._tensor.dtype})"
        return f"LazyTensor(pending, {self._spec.name}, shape={self._spec.shape}, dtype={self._spec.dtype}, nbytes={self._spec.nbytes})"

    # ==================== PyTorch compatibility ====================

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Handle PyTorch functions that receive LazyTensor as input.
        Automatically unwrap LazyTensor to underlying tensor.
        
        This makes LazyTensor compatible with functions like:
        - torch.empty_like()
        - torch.zeros_like()
        - torch.ones_like()
        - etc.
        """
        if kwargs is None:
            kwargs = {}

        # Unwrap LazyTensor arguments to underlying tensors
        def unwrap(arg):
            if isinstance(arg, LazyTensor):
                arg._ensure_materialized()
                return arg._tensor
            elif isinstance(arg, (list, tuple)):
                return type(arg)(unwrap(a) for a in arg)
            elif isinstance(arg, dict):
                return {k: unwrap(v) for k, v in arg.items()}
            return arg

        args = unwrap(args)
        kwargs = unwrap(kwargs)

        return func(*args, **kwargs)


class LazyAllocator:
    """
    Lazy allocator for tensors.
    
    This allocator can delay tensor allocation until sync() is called,
    allowing users to query the total memory requirement before allocation.
    
    Args:
        create_tensor_fn: Function to create a tensor, signature: (shape, dtype) -> Tensor
        free_tensor_fn: Optional function to free a tensor, signature: (tensor) -> None
        lazy: If True, delay allocation until sync() is called.
              If False, allocate immediately (default behavior).
    
    Usage:
        allocator = LazyAllocator(
            create_tensor_fn=nvshmem_create_tensor,
            free_tensor_fn=nvshmem_free_tensor_sync,
            lazy=True
        )
        
        # Create lazy tensors (no actual allocation)
        tensor1 = allocator.create_tensor("buf1", [1024, 256], torch.bfloat16)
        tensor2 = allocator.create_tensor("buf2", [512], torch.int32)
        
        # Query total size before allocation
        total_bytes = allocator.get_total_size()
        print(f"Total memory needed: {total_bytes / 1e9:.2f} GB")
        
        # Actually allocate all tensors
        allocator.sync()
        
        # Now tensors can be used normally
        tensor1.fill_(0)
    """

    def __init__(self, create_tensor_fn: Callable[[List[int], torch.dtype], torch.Tensor],
                 free_tensor_fn: Optional[Callable[[torch.Tensor], None]] = None, lazy: bool = False):
        """
        Initialize the allocator.
        
        Args:
            create_tensor_fn: Function to create a tensor, signature: (shape, dtype) -> Tensor
            free_tensor_fn: Optional function to free a tensor, signature: (tensor) -> None
            lazy: If True, delay allocation until sync() is called.
                  If False, allocate immediately (default behavior).
        """
        self._create_tensor_fn = create_tensor_fn
        self._free_tensor_fn = free_tensor_fn
        self._lazy = lazy
        self._lazy_tensors: List[LazyTensor] = []
        self._materialized = False
        self._total_bytes = 0

    @property
    def lazy(self) -> bool:
        """Check if allocator is in lazy mode."""
        return self._lazy

    @property
    def is_materialized(self) -> bool:
        """Check if all tensors have been materialized."""
        return self._materialized or not self._lazy

    def create_tensor(self, name: str, shape: List[int], dtype: torch.dtype,
                      fill_value: Optional[float] = None) -> LazyTensor:
        """
        Create a (potentially lazy) tensor.
        
        Args:
            name: Name for debugging/tracking
            shape: Tensor shape
            dtype: Tensor dtype
            fill_value: Optional value to fill after allocation
        
        Returns:
            LazyTensor that wraps the allocation
        """
        spec = LazyTensorSpec(name=name, shape=list(shape), dtype=dtype, fill_value=fill_value)
        lazy_tensor = LazyTensor(spec, self)

        self._total_bytes += spec.nbytes
        self._lazy_tensors.append(lazy_tensor)

        if not self._lazy or self._materialized:
            # Allocate immediately
            tensor = self._create_tensor_fn(shape, dtype)
            lazy_tensor._materialize(tensor)

        return lazy_tensor

    def get_total_size(self) -> int:
        """
        Get the total size in bytes.
        
        This can be called before sync() to query the total memory needed.
        """
        return self._total_bytes

    def get_total_size_gb(self) -> float:
        """Get the total size in GB."""
        return self._total_bytes / (1024**3)

    def get_total_size_mb(self) -> float:
        """Get the total size in MB."""
        return self._total_bytes / (1024**2)

    def get_tensor_breakdown(self) -> Dict[str, int]:
        """
        Get a breakdown of memory usage by tensor.
        
        Returns:
            Dict mapping tensor name to size in bytes
        """
        return {lt._spec.name: lt._spec.nbytes for lt in self._lazy_tensors}

    def print_memory_breakdown(self) -> None:
        """Print a human-readable breakdown of memory usage."""
        breakdown = self.get_tensor_breakdown()
        print(f"{'Tensor Name':<60} {'Size (Bytes)':>12} {'Size (GB)':>12}")
        print("-" * 86)
        for name, nbytes in sorted(breakdown.items(), key=lambda x: -x[1]):
            print(f"{name:<60} {nbytes:>12} {nbytes / 1e9:>12.4f}")
        print("-" * 86)
        print(f"{'TOTAL':<60} {self._total_bytes / 1e6:>12.2f} {self._total_bytes / 1e9:>12.4f}")

    def sync(self) -> None:
        """
        Materialize all lazy tensors.
        
        This actually allocates the memory for all pending tensors.
        """
        if self._materialized:
            return

        if not self._lazy:
            self._materialized = True
            return

        # Allocate all pending tensors
        for lazy_tensor in self._lazy_tensors:
            if not lazy_tensor.is_materialized:
                tensor = self._create_tensor_fn(lazy_tensor._spec.shape, lazy_tensor._spec.dtype)
                lazy_tensor._materialize(tensor)

        self._materialized = True

    def materialize(self) -> None:
        """Alias for sync()."""
        self.sync()

    def free_tensor(self, tensor_or_lazy: Union[LazyTensor, torch.Tensor, None]) -> None:
        """
        Free a tensor using the configured free function.
        
        Handles both LazyTensor and regular torch.Tensor.
        """
        if tensor_or_lazy is None:
            return

        if self._free_tensor_fn is None:
            return

        underlying = tensor_or_lazy
        if isinstance(tensor_or_lazy, LazyTensor):
            underlying = tensor_or_lazy.get_underlying_tensor()

        if underlying is not None:
            self._free_tensor_fn(underlying)

    def __len__(self) -> int:
        """Number of tensors managed by this allocator."""
        return len(self._lazy_tensors)


# Convenience function for getting underlying tensor
def get_underlying_tensor(tensor_or_lazy: Union[LazyTensor, torch.Tensor, None]) -> Optional[torch.Tensor]:
    """
    Get the underlying tensor from a LazyTensor or return the tensor as-is.
    
    Args:
        tensor_or_lazy: LazyTensor, torch.Tensor, or None
    
    Returns:
        The underlying torch.Tensor, or None
    """
    if tensor_or_lazy is None:
        return None
    if isinstance(tensor_or_lazy, LazyTensor):
        return tensor_or_lazy.get_underlying_tensor()
    return tensor_or_lazy


def nvshmem_free_lazy_tensor(tensor_or_lazy):
    """
    Free a nvshmem tensor, handling both LazyTensor and regular torch.Tensor.
    """
    underlying = get_underlying_tensor(tensor_or_lazy)
    if underlying is not None:
        nvshmem_free_tensor_sync(underlying)


class NVSHMEMLazyAllocator(LazyAllocator):
    """
    Lazy allocator specifically for nvshmem tensors.
    
    This is a convenience wrapper around LazyAllocator that uses
    nvshmem_create_tensor and nvshmem_free_tensor_sync by default.
    
    Usage:
        allocator = NVSHMEMLazyAllocator(lazy=True)
        
        # Create lazy tensors (no actual allocation)
        tensor1 = allocator.create_tensor("buf1", [1024, 256], torch.bfloat16)
        tensor2 = allocator.create_tensor("buf2", [512], torch.int32)
        
        # Query total size before allocation
        total_bytes = allocator.get_total_nvshmem_size()
        print(f"Total nvshmem needed: {total_bytes / 1e9:.2f} GB")
        
        # Actually allocate all tensors
        allocator.sync()
        
        # Now tensors can be used normally
        tensor1.fill_(0)
    """

    def __init__(self, lazy: bool = False):
        """
        Initialize the nvshmem allocator.
        
        Args:
            lazy: If True, delay allocation until sync() is called.
                  If False, allocate immediately (default behavior).
        """
        super().__init__(create_tensor_fn=nvshmem_create_tensor, free_tensor_fn=nvshmem_free_tensor_sync, lazy=lazy)

    # Convenience aliases for backward compatibility
    def get_total_nvshmem_size(self) -> int:
        """Get the total nvshmem size in bytes."""
        return self.get_total_size()

    def get_total_nvshmem_size_gb(self) -> float:
        """Get the total nvshmem size in GB."""
        return self.get_total_size_gb()

    def get_total_nvshmem_size_mb(self) -> float:
        """Get the total nvshmem size in MB."""
        return self.get_total_size_mb()
