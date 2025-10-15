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
import subprocess
import sys
import functools
import warnings
import torch

from threading import Lock
from cuda import cuda

_HAS_PYNVML = False
try:
    import pynvml
    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False
    pynvml = None

_PYNVML_INITIALIZED = False

_LOCK = Lock()


def ensure_nvml_initialized():
    global _PYNVML_INITIALIZED
    if not _PYNVML_INITIALIZED:
        with _LOCK:
            if not _PYNVML_INITIALIZED:
                import pynvml

                pynvml.nvmlInit()
                _PYNVML_INITIALIZED = True


def with_pynvml():
    global _HAS_PYNVML
    if _HAS_PYNVML:
        ensure_nvml_initialized()
    return _HAS_PYNVML


def nvsmi(attrs, device_id=0, dtype: type = int):
    attrs = ','.join(attrs)
    cmd = ['nvidia-smi', '-i', str(device_id), '--query-gpu=' + attrs, '--format=csv,noheader,nounits']
    out = subprocess.check_output(cmd)
    ret = [x.strip() for x in out.decode(sys.stdout.encoding).split(',')]
    return [dtype(x) for x in ret]


# this is much a copy of https://github.com/triton-lang/kernels/blob/main/kernels/matmul_perf_model.py
@functools.lru_cache()
def get_max_gpu_clock_rate_in_khz(device_id=0):
    if with_pynvml():
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM) * 1e3

    return nvsmi(["clocks.max.sm"])[0] * 1e3


def get_nvlink_adjacency_matrix():
    output = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    lines = [line.strip() for line in output.split("\n") if line.startswith("GPU")]

    device_count = len(lines)
    matrix = [[-1 for _ in range(device_count)] for _ in range(device_count)]

    for i, line in enumerate(lines):
        parts = line.split()
        for j in range(1, len(parts)):
            if "NV" in parts[j]:
                matrix[i][j - 1] = 1

    return matrix


def _get_gpu_numa_node(gpu_index=0):
    try:
        pci_id = nvsmi(["pci.bus_id"], gpu_index, dtype=str)[0]
        pci_address = pci_id.replace("00000000:", "").lower()  # "00000000:17:00.0" → "17:00.0"
        # print(f"gpu_index: {gpu_index} => {pci_id} => {pci_address}")

        numa_node_path = f"/sys/bus/pci/devices/0000:{pci_address}/numa_node"
        with open(numa_node_path, "r") as f:
            numa_node = int(f.read().strip())

        assert numa_node >= 0
        return numa_node
    except Exception as e:
        print(f"Error: {e}")
        return 0


@functools.lru_cache(maxsize=16)
def _get_active_nvlinks_pynvml(gpu_index: int):
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
def _get_nvlink_max_speed_gbps_pynvml(gpu_index=0):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    values = pynvml.nvmlDeviceGetFieldValues(handle, [pynvml.NVML_FI_DEV_NVLINK_GET_SPEED])
    speed = parse_nvml_field_value(values[0])
    # speed in Mbps but in 1e6, not MB (1024 * 1024)
    speed = speed * 1e6 / 2**30
    return _get_active_nvlinks_pynvml(gpu_index) * speed


@functools.lru_cache()
def to_physical_device_id(gpu_index: int | None = None):
    uuid = _get_nvml_gpu_uuid(gpu_index)

    uuid_map = {get_physical_gpu_uuid(i): i for i in range(get_physical_device_count())}
    return uuid_map[uuid]


@functools.lru_cache(maxsize=16)
def _get_nvlink_max_speed_gbps_nvsmi(gpu_index=0):
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


def get_nvlink_max_speed_gbps(gpu_index=0):
    try:
        if with_pynvml():
            return _get_nvlink_max_speed_gbps_pynvml(gpu_index)
    except Exception:
        return _get_nvlink_max_speed_gbps_nvsmi(gpu_index)


@functools.lru_cache()
def has_fullmesh_nvlink_pynvml():
    num_devices = torch.cuda.device_count()

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
def _get_numa_node_pynvml(gpu_index):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    return pynvml.nvmlDeviceGetNumaNodeId(handle)  # no such symbol for CUDA driver 535.161.08


def calculate_pcie_bandwidth_gbps(generation: int, lanes: int) -> tuple:
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
    per_direction_gbps = (transfer_rate * encoding * lanes) / 8

    return per_direction_gbps


@functools.lru_cache()
def _get_pcie_link_max_speed_gbps_nvsmi(gpu_index=0):
    """Returns the maximum PCIe link speed in GB/s for specified GPU"""
    pcie_gen, lanes = nvsmi(["pcie.link.gen.current", "pcie.link.width.current"], gpu_index)
    return calculate_pcie_bandwidth_gbps(pcie_gen, lanes)


@functools.lru_cache()
def _get_pcie_link_max_speed_gbps_pynvml(gpu_index=0):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
    lanes = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
    return calculate_pcie_bandwidth_gbps(pcie_gen, lanes)


@functools.lru_cache()
def get_pcie_link_max_speed_gbps(gpu_index):
    if with_pynvml():
        return _get_pcie_link_max_speed_gbps_pynvml(gpu_index)

    return _get_pcie_link_max_speed_gbps_nvsmi(gpu_index)


@functools.lru_cache()
def get_numa_node(gpu_index):
    try:
        if with_pynvml():
            _get_numa_node_pynvml(gpu_index)
    except Exception:
        return _get_gpu_numa_node(gpu_index)


@functools.lru_cache()
def has_fullmesh_nvlink():
    if with_pynvml():
        return has_fullmesh_nvlink_pynvml()

    nvlink_matrix = get_nvlink_adjacency_matrix()
    has_nvlink = any([any(x == 1 for x in row) for row in nvlink_matrix])
    _has_fullmesh_nvlink = all([i == j or v == 1 for i, row in enumerate(nvlink_matrix) for j, v in enumerate(row)])
    if has_nvlink and not _has_fullmesh_nvlink:
        warnings.warn(
            "⚠️ found NVLink but not fullmesh NVLink, this may cause undefined behavior, please check your GPU topology"
        )
    return _has_fullmesh_nvlink


def get_intranode_max_speed_gbps(gpu_index=0, with_scale: bool = False):
    if has_fullmesh_nvlink():
        # 200GB/s => 160GB/s
        _factor = 1.0 if not with_scale else 0.8
        return get_nvlink_max_speed_gbps(gpu_index) * _factor
    else:
        # 32GB/s => 22.4GB/s
        _factor = 1.0 if not with_scale else 0.7
        return get_pcie_link_max_speed_gbps(gpu_index) * _factor


@functools.lru_cache()
def get_device_name(device_id):
    return nvsmi(["name"], device_id, dtype=str)[0]


def get_current_gpu_clock_rate_in_khz(device_id=0):
    if with_pynvml():
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM) * 1e3

    return nvsmi(["clocks.current.sm"])[0] * 1e3


def gpu_uuid_string(uuid_bytes: bytes) -> str:
    """Format 16-byte CUuuid as NVML-style 'GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'."""
    h = uuid_bytes.hex()
    return f"GPU-{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


@functools.lru_cache()
def _get_nvml_gpu_uuid(device_id: int | None):
    if device_id is None:
        device_id = torch.cuda.current_device()
    try:
        uuid = torch.cuda.get_device_properties(device_id).uuid
        return "GPU-" + uuid
    except Exception:
        err, dev = cuda.cuDeviceGet(device_id)
        assert err == cuda.CUresult.CUDA_SUCCESS, f"cuDeviceGet({device_id}) failed: {err}"

        # UUID (16 bytes)
        err, cuuuid = cuda.cuDeviceGetUuid(dev)
        assert err == cuda.CUresult.CUDA_SUCCESS, f"cuDeviceGetUuid({device_id}) failed: {err}"

        # cuuuid has a .bytes field (length 16)
        uuid_bytes = bytes(cuuuid.bytes)
        return gpu_uuid_string(uuid_bytes)


def _get_physical_gpu_uuid_pynvml(device_id: int | None):
    device_id = device_id or 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    return pynvml.nvmlDeviceGetUUID(handle)


def _get_physical_gpu_uuid_nvsmi(device_id: int | None):
    device_id = device_id or 0
    return nvsmi(["uuid"], device_id, dtype=str)[0]


def get_physical_gpu_uuid(gpu_index: int | None):
    if with_pynvml():
        return _get_physical_gpu_uuid_pynvml(gpu_index)
    return _get_physical_gpu_uuid_nvsmi(gpu_index)


def get_physical_device_count():
    if with_pynvml():
        return pynvml.nvmlDeviceGetCount()
    return nvsmi(["count"], dtype=int)[0]


def _get_gpu_performance_mode_nvsmi(device_id: int) -> str:
    return nvsmi(["pstate"], device_id, dtype=str)


def _get_gpu_performance_mode_pynvml(device_id: int) -> int:
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    performance_state = pynvml.nvmlDeviceGetPerformanceState(handle)
    return performance_state


@functools.lru_cache()
def is_gpu_max_performance_mode(device_id: int):
    if with_pynvml():
        return _get_gpu_performance_mode_pynvml(device_id) == 0

    return _get_gpu_performance_mode_nvsmi(device_id) == "P0"


__all__ = [
    "get_numa_node",
    "to_physical_device_id",
    "get_max_gpu_clock_rate_in_khz",
    "get_current_gpu_clock_rate_in_khz",
    "get_intranode_max_speed_gbps",
    "is_gpu_max_performance_mode",
]
