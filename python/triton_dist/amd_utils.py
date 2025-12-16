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
import torch
import functools
import subprocess
import json
import warnings
import re
from threading import Lock
from hip import hip

_HAS_AMDSMI = False
_AMDSMI_INITIALIZED = False
_LOCK = Lock()
try:
    import amdsmi

    _HAS_AMDSMI = True
except Exception:
    amdsmi = None


def has_amdsmi():
    global _HAS_AMDSMI
    if _HAS_AMDSMI:
        _ensure_amdsmi_initialized()
    return _HAS_AMDSMI


def _ensure_amdsmi_initialized():
    global _AMDSMI_INITIALIZED
    with _LOCK:
        if not _AMDSMI_INITIALIZED:
            amdsmi.amdsmi_init()
            _AMDSMI_INITIALIZED = True


def _check_rocm_smi_json(cmd):
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _get_current_gpu_clock_rate_in_khz_rocm(device_id=None):
    device_id = device_id if device_id is not None else torch.cuda.current_device()
    try:
        data = _check_rocm_smi_json(["rocm-smi", "--showclock", "--json", "-d", str(device_id)])
        sclk_entries = data[f"card{device_id}"]["sclk clock speed:"]
        match = re.search(r"(\d+)Mhz", sclk_entries)
        if not match:
            raise ValueError(f"Could not parse sclk from rocm-smi output: {data}")

        sclk_mhz = int(match.group(1))
        return sclk_mhz * 1000  # convert MHz to kHz
    except (
            subprocess.CalledProcessError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
    ) as e:
        warnings.warn(f"Could not get max GPU clock from rocm-smi: {e}. Using fallback.")
        # A reasonable default for MI200/MI300 is around 1.5-1.7 GHz
        return 1700 * 1000  # in kHz


def _get_current_gpu_clock_rate_in_khz_amdsmi(device_id: int):
    handle = amdsmi.amdsmi_get_processor_handles()[device_id]
    clock_info = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.GFX)
    return clock_info["clk"] * 1000


def get_current_gpu_clock_rate_in_khz(device_id=None):
    device_id = _get_amdsmi_device_index(device_id)
    try:
        if has_amdsmi():
            return _get_current_gpu_clock_rate_in_khz_amdsmi(device_id)
    except Exception:
        warnings.warn("get_current_gpu_clock_rate_in_khz failed with amdsmi, try using rocm-smi")

    return _get_current_gpu_clock_rate_in_khz_rocm(device_id)


@functools.lru_cache()
def _get_max_gpu_clock_rate_in_khz_amdsmi(device_id):
    _ensure_amdsmi_initialized()
    devices = amdsmi.amdsmi_get_processor_handles()
    handle = devices[device_id]
    # clock info is something like {'clk': 88, 'min_clk': 500, 'max_clk': 1420, 'clk_locked': 0, 'clk_deep_sleep': 88}
    clock_info = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.GFX)
    return clock_info["max_clk"] * 1000


@functools.lru_cache()
def _get_max_gpu_clock_rate_in_khz_rocm(device_id):
    # {"card0": {"Valid sclk range": "500Mhz - 1420Mhz"}}
    data = _check_rocm_smi_json(["rocm-smi", "--showsclkrange", "--json", "-d", str(device_id)])
    sclk_entries = data[f"card{device_id}"]["Valid sclk range"]
    match = re.search(r"(\d+)Mhz - (\d+)Mhz", sclk_entries)
    if not match:
        raise ValueError(f"Could not parse sclk from rocm-smi output: {data}")
    _, sclk_mhz_max = int(match.group(1)), int(match.group(2))
    return sclk_mhz_max * 1000  # convert MHz to kHz


@functools.lru_cache()
def get_max_gpu_clock_rate_in_khz(device_id: int | None = None):
    device_id = _get_amdsmi_device_index(device_id)
    try:
        if has_amdsmi():
            return _get_max_gpu_clock_rate_in_khz_amdsmi(device_id)
    except Exception:
        warnings.warn("get_max_gpu_clock_rate_in_khz failed with amdsmi, try using rocm-smi")

    return _get_max_gpu_clock_rate_in_khz_rocm(device_id)


@functools.lru_cache()
def _get_numa_node_amdsmi(device_id: int):
    devices = amdsmi.amdsmi_get_processor_handles()
    handle = devices[device_id]
    return amdsmi.amdsmi_topo_get_numa_node_number(handle)


@functools.lru_cache()
def _get_numa_node_rocm(device_id: int):
    """
    Uses `rocm-smi --showtoponuma --json` and returns {"card0": 0, ...}
    """
    data = _check_rocm_smi_json(["rocm-smi", "--showtoponuma", "--json"])
    return int(data[f"card{device_id}"]["(Topology) Numa Node"])


def get_numa_node(device_id=None):
    device_id = _get_amdsmi_device_index(device_id)
    try:
        if has_amdsmi():
            return _get_numa_node_amdsmi(device_id)
    except Exception:
        warnings.warn("get_numa_node failed with amdsmi, try using rocm-smi")

    return _get_numa_node_rocm(device_id)


@functools.lru_cache()
def _has_fullmesh_xgmi_amdsmi():
    devices = amdsmi.amdsmi_get_processor_handles()
    for i, hi in enumerate(devices):
        for j, hj in enumerate(devices):
            if i == j:
                continue
            link_type = amdsmi.amdsmi_topo_get_link_type(hi, hj)["type"]
            # use this with care: https://rocm.docs.amd.com/projects/amdsmi/en/latest/reference/changelog.html#id3: AMDSMI_LINK_TYPE_XGMI and AMDSMI_LINK_TYPE_PCIE is reordered
            if link_type != amdsmi.AmdSmiIoLinkType.XGMI:
                return False
    return True


@functools.lru_cache()
def _has_fullmesh_xgmi_rocm():
    out = _check_rocm_smi_json(["rocm-smi", "--showtopotype", "--json"])
    for _gpus, link_type in out["system"].items():
        if link_type != "XGMI":
            return False
    return True


@functools.lru_cache()
def has_fullmesh_xgmi():
    try:
        if has_amdsmi():
            return _has_fullmesh_xgmi_amdsmi()
    except Exception:
        warnings.warn("has_fullmesh_xgmi failed with amdsmi, try use rocm-smi")

    return _has_fullmesh_xgmi_rocm()


@functools.lru_cache()
def _get_gpu_uuid_by_physical_device_id(device_id: int):
    _ensure_amdsmi_initialized()
    devices = amdsmi.amdsmi_get_processor_handles()
    handle = devices[device_id]
    return amdsmi.amdsmi_get_gpu_device_uuid(handle)


def torch_uuid_to_unique_id(torch_uuid: str) -> str:
    """
    Convert torch UUID to unique ID.

    torch.cuda.get_device_properties().uuid => 30333235-3434-3765-3339-313439656261
    rocm-smi --showuniqueid => 0x325447e39149eba
    amdsmi_get_gpu_device_uuid => 03ff74a2-0000-1000-8025-447e39149eba
    """
    """'30333235-...' -> '0325447e39149eba'"""
    h = torch_uuid.replace("-", "")
    return bytes.fromhex(h).decode("ascii")


def _get_physical_gpu_uuid_rocm(device_id: int):
    ret = _check_rocm_smi_json(["rocm-smi", "--showuniqueid", "--json", "-d", str(device_id)])
    uuid = ret[f"card{device_id}"]["Unique ID"]
    return uuid


def get_uuid_by_physical_device_id(device_id: int | None = None):
    try:
        if has_amdsmi():
            return _get_gpu_uuid_by_physical_device_id(device_id)
    except Exception:
        warnings.warn("get_uuid_by_physical_device_id failed with amdsmi, try using rocm-smi")

    return _get_physical_gpu_uuid_rocm(device_id)


@functools.lru_cache()
def _get_gpu_uuid(device_id: int):
    try:
        return torch_uuid_to_unique_id(str(torch.cuda.get_device_properties(device_id).uuid))
    except AttributeError:
        prop = hip.hipDeviceProp_t()
        err, = hip.hipGetDeviceProperties(prop, device_id)
        assert err == hip.hipError_t.hipSuccess
        return prop.uuid.bytes.decode("ascii")


@functools.lru_cache()
def _get_amdsmi_device_index(device_id: int | None):
    if device_id is None:
        device_id = torch.cuda.current_device()

    uuid = _get_gpu_uuid(device_id)

    uuid_map = {get_uuid_by_physical_device_id(i)[-12:]: i for i in range(get_physical_device_count())}
    return uuid_map[uuid[-12:]]


def get_physical_device_count():
    try:
        if has_amdsmi():
            return len(amdsmi.amdsmi_get_processor_handles())
    except Exception:
        warnings.warn("get_physical_device_count failed with amdsmi, try using rocm-smi")

    return len(_check_rocm_smi_json(["rocm-smi", "--showid", "--json"]))


@functools.lru_cache()
def _get_bus_bw_gbps_between_amdsmi(device_id_i, device_id_j):
    devices = amdsmi.amdsmi_get_processor_handles()
    handle_i = devices[device_id_i]
    handle_j = devices[device_id_j]
    return amdsmi.amdsmi_get_minmax_bandwidth_between_processors(handle_i, handle_j)["max_bandwidth"] * 1e6 / 2**30


def _parse_rocm_shownodesbw_output_in_gbps(output):
    """
        $ rocm-smi --shownodesbw


    ============================ ROCm System Management Interface ============================
    ======================================= Bandwidth ========================================
           GPU0         GPU1         GPU2         GPU3         GPU4         GPU5         GPU6         GPU7
    GPU0   N/A          50000-50000  50000-50000  50000-50000  50000-50000  50000-50000  50000-50000  50000-50000
    GPU1   50000-50000  N/A          50000-50000  50000-50000  50000-50000  50000-50000  50000-50000  50000-50000
    GPU2   50000-50000  50000-50000  N/A          50000-50000  50000-50000  50000-50000  50000-50000  50000-50000
    GPU3   50000-50000  50000-50000  50000-50000  N/A          50000-50000  50000-50000  50000-50000  50000-50000
    GPU4   50000-50000  50000-50000  50000-50000  50000-50000  N/A          50000-50000  50000-50000  50000-50000
    GPU5   50000-50000  50000-50000  50000-50000  50000-50000  50000-50000  N/A          50000-50000  50000-50000
    GPU6   50000-50000  50000-50000  50000-50000  50000-50000  50000-50000  50000-50000  N/A          50000-50000
    GPU7   50000-50000  50000-50000  50000-50000  50000-50000  50000-50000  50000-50000  50000-50000  N/A
    Format: min-max; Units: mps
    "0-0" min-max bandwidth indicates devices are not connected directly
    ================================== End of ROCm SMI Log ===================================
    """
    lines = output.strip().split("\n")
    bandwidth_matrix = {}
    header = []

    # Regex to find the data lines, which start with 'GPU' and a digit.
    data_line_regex = re.compile(r"^GPU\d+\s+")

    for line in lines:
        # Find the header row which contains the column titles
        if not header and all(f"GPU{i}" in line for i in range(2)):
            # Split by 2 or more spaces to handle alignment
            header = re.split(r"\s{2,}", line.strip())
            continue

        # Process lines that contain the actual bandwidth data
        if data_line_regex.match(line):
            parts = re.split(r"\s{2,}", line.strip())
            row_gpu = parts[0]
            row_gpu = int(row_gpu[3:])
            values = parts[1:]

            if not header:
                print("Error: Could not parse header row before data.")
                return None

            for i, value in enumerate(values):
                col_gpu = int(header[i][3:])
                if value == "N/A":
                    bandwidth_matrix[(row_gpu, col_gpu)] = None
                else:
                    try:
                        _, max_bw = map(int, value.split("-"))
                        bandwidth_matrix[(row_gpu, col_gpu)] = max_bw * 1e6 / 2**30
                    except ValueError:
                        # Handle potential parsing errors if format is unexpected
                        bandwidth_matrix[(row_gpu, col_gpu)] = float("nan")

    return bandwidth_matrix


@functools.lru_cache()
def _get_bus_bw_gbps_a2a_rocm():
    try:
        result = subprocess.run(["rocm-smi", "--shownodesbw"], capture_output=True, text=True, check=True)
        output = result.stdout
        return _parse_rocm_shownodesbw_output_in_gbps(output)
    except FileNotFoundError:
        print("Error: 'rocm-smi' command not found.")
        print("Please ensure the ROCm stack is installed and 'rocm-smi' is in your system's PATH.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


@functools.lru_cache()
def _get_bus_bw_gbps_between_rocm(device_id_i: int, device_id_j: int):
    bw_matrix_gbps = _get_bus_bw_gbps_a2a_rocm()
    return bw_matrix_gbps[(device_id_i, device_id_j)]


@functools.lru_cache()
def _get_bus_bw_gbps_between(device_id_i: int, device_id_j: int):
    try:
        if has_amdsmi():
            return _get_bus_bw_gbps_between_amdsmi(device_id_i, device_id_j)
    except Exception:
        warnings.warn("get_bus_bw_gbps_between fails with amdsmi, try using rocm-smi")

    return _get_bus_bw_gbps_between_rocm(device_id_i, device_id_j)


@functools.lru_cache()
def _get_xgmi_max_speed_gbps_amdsmi():
    devices = amdsmi.amdsmi_get_processor_handles()
    for i, hi in enumerate(devices):
        for j, hj in enumerate(devices):
            if i == j:
                continue
            # No unit from the doc: https://rocm.docs.amd.com/projects/amdsmi/en/latest/reference/amdsmi-py-api.html#amdsmi-get-minmax-bandwidth-between-processors
            # it approximately matches the number here https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html#theoretical-performance-claims-vs-achievable
            # so here we take it as Mbps
            speed = amdsmi.amdsmi_get_minmax_bandwidth_between_processors(hi, hj)["max_bandwidth"]
            return speed * 1e6 / 2**30


@functools.lru_cache()
def _get_xgmi_max_speed_gbps_rocm():
    return _get_bus_bw_gbps_between(0, 1)


@functools.lru_cache()
def get_xgmi_max_speed_gbps():
    try:
        if has_amdsmi():
            return _get_xgmi_max_speed_gbps_amdsmi()
    except Exception:
        warnings.warn("get_xgmi_max_speed_gbps failed with amdsmi, try use rocm-smi")

    return _get_xgmi_max_speed_gbps_rocm()


@functools.lru_cache()
def get_pcie_link_max_speed_gbps():
    pass


@functools.lru_cache()
def get_intranode_max_speed_gbps(device_id, with_scale: bool = False):
    """ suppose the node is symmetric """
    if has_fullmesh_xgmi():
        return _get_bus_bw_gbps_between(0, 1) * (torch.cuda.device_count() - 1)

    return _get_bus_bw_gbps_between(0, 1)


def _get_gpu_performance_mode_rocm(device_id: int):
    data = _check_rocm_smi_json(["rocm-smi", "--json", "-d", str(device_id), "--showperflevel"])
    return data[f"card{device_id}"]["Performance Level"]


def _get_gpu_performance_mode_amdsmi(device_id: int) -> str:
    handle = amdsmi.amdsmi_get_processor_handles()[device_id]
    return amdsmi.amdsmi_get_gpu_perf_level(handle)


def is_gpu_max_performance_mode(device_id: int):
    device_id = _get_amdsmi_device_index(device_id)
    try:
        if has_amdsmi():
            return _get_gpu_performance_mode_amdsmi(device_id) == "AMDSMI_DEV_PERF_LEVEL_HIGH"
    except Exception:
        warnings.warn("is_gpu_max_performance_mode failed with amdsmi, try using rocm-smi")

    return _get_gpu_performance_mode_rocm(device_id) == "high"


def get_num_xcds_by_amdsmi(device_id: int = 0):
    if has_amdsmi():
        try:
            devices = amdsmi.amdsmi_get_processor_handles()
            handle = devices[device_id]
            return amdsmi.amdsmi_get_gpu_xcd_counter(handle)
        except Exception:
            return -1
    return -1


__all__ = [
    "_get_amdsmi_device_index", "get_max_gpu_clock_rate_in_khz", "get_current_gpu_clock_rate_in_khz",
    "get_intranode_max_speed_gbps", "get_numa_node", "is_gpu_max_performance_mode", "get_num_xcds_by_amdsmi",
    "has_amdsmi"
]
