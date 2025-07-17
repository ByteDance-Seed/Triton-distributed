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
import functools
from typing import Optional

import torch
import triton
from triton.testing import get_max_simd_tflops, nvsmi
import logging


def is_fp8_dtype(dtype: torch.dtype):
    return dtype.itemsize == 1 and dtype.is_floating_point


# this is much a copy of https://github.com/triton-lang/kernels/blob/main/kernels/matmul_perf_model.py
@functools.lru_cache()
def get_clock_rate_in_khz():
    try:
        return nvsmi(["clocks.max.sm"])[0] * 1e3
    except FileNotFoundError:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM) * 1e3


@functools.lru_cache()
def get_device_property(device):
    return torch.cuda.get_device_properties(device=device)


@functools.lru_cache()
def get_device_multi_processor_count(device):
    return get_device_property(device).multi_processor_count


def get_max_tensorcore_tflops(dtype, clock_rate_mhz, device=None):
    """
    clock_rate in Mhz

    NOTE: don't use triton.testing.get_max_tensorcore_tflops. not correct for Hopper and later archs
    """
    device = device or torch.cuda.current_device()
    num_subcores = get_device_multi_processor_count(device) * 4  # on recent GPUs
    cap_major, _ = torch.cuda.get_device_capability(device)
    if cap_major < 8:  # Volta arch. TensorCore V2
        assert dtype == torch.float16
        ops_per_sub_core = 256  # 2 4x4x4 Tensor Cores
    elif cap_major < 9:  # Ampere
        assert dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int8,
        ], f"dtype not supported: {dtype}"
        ops_per_sub_core = 256 * (4 / dtype.itemsize)
    elif cap_major < 10:  # Hopper
        assert dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int8,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ], f"dtype not supported: {dtype}"
        ops_per_sub_core = 512 * (4 / dtype.itemsize)
    else:  # TensorCore gen05
        # TODO(houqi.1993)
        assert dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int8,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ], f"dtype not supported: {dtype}"
        ops_per_sub_core = 1024 * (4 / dtype.itemsize)

    tflops = num_subcores * clock_rate_mhz * ops_per_sub_core * 1e-9
    logging.info(f"num_subcores: {num_subcores} clock_rate: {clock_rate_mhz} ops_per_sub_core: {ops_per_sub_core}")
    return tflops


def get_tensorcore_tflops_by_calc(device, num_ctas, num_warps, dtype):
    """return compute throughput in TOPS"""
    total_warps = num_ctas * min(num_warps, 4)
    num_subcores = get_device_multi_processor_count(device) * 4  # on recent GPUs
    tflops = (min(num_subcores, total_warps) / num_subcores *
              get_max_tensorcore_tflops(dtype, get_clock_rate_in_khz(), device))
    return tflops


def get_simd_tflops(device, num_ctas, num_warps, dtype):
    """return compute throughput in TOPS"""
    total_warps = num_ctas * min(num_warps, 4)
    num_subcores = get_device_multi_processor_count(device) * 4  # on recent GPUs
    tflops = (min(num_subcores, total_warps) / num_subcores *
              get_max_simd_tflops(dtype, get_clock_rate_in_khz(), device))
    return tflops


def get_tflops_approx(device: torch.dtype, num_ctas: int, num_warps: int, dtype: torch.dtype):
    """You may not achieve"""
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8 and dtype == torch.float32:
        return get_simd_tflops(device, num_ctas, num_warps, dtype)
    return get_tensorcore_tflops_by_calc(device, num_ctas, num_warps, dtype)


def get_full_tflops_approx(dtype: torch.dtype, device: Optional[torch.device] = None):
    prop = torch.cuda.get_device_properties(device)
    return get_tflops_approx(device, prop.multi_processor_count, 4, dtype)


def get_tensorcore_dtype_support(device_id=0):
    """ INT4/MXFP4/MXFP6 is not listed here
    """
    cap = torch.cuda.get_device_capability(device_id)
    DTYPE_MAP = {
        # Ampere Family
        (8, 0): [torch.float16, torch.bfloat16, torch.float32, torch.int8],  # A100
        (8, 6): [torch.float16, torch.bfloat16, torch.float32, torch.int8],  # RTX 30xx
        (8, 9): [torch.float16, torch.bfloat16, torch.float32, torch.int8, torch.float8_e4m3fn,
                 torch.float8_e5m2],  # Ada L40S/RTX 40xx
        # Hopper
        (9, 0): [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2, torch.int8]
    }
    return DTYPE_MAP.get(cap, [torch.float16, torch.float32])


def get_tensorcore_tflops_by_device_name(dtype, device_id=0):
    """TFLOPS with no sparse."""
    assert dtype in get_tensorcore_dtype_support(device_id)

    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    # Ampere
    # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf
    # "NVIDIA A100 80GB PCIe" or "NVIDIA A100-SXM4-80GB" or "A100-SXM4-40GB" or so
    if device_name.find("A100") >= 0 or device_name.find("A800") >= 0:
        return 312 * (2 / dtype.itemsize)
    # https://www.nvidia.com/en-us/data-center/products/a10-gpu/
    if device_name.find("A10") >= 0:
        return 125 * (2 / dtype.itemsize)
    # https://www.nvidia.com/en-us/data-center/products/a30-gpu/
    if device_name.find("A30") >= 0:
        return 165 * (2 / dtype.itemsize)
    # https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf
    if device_name.find("A40") >= 0:
        return 149.7 * (2 / dtype.itemsize)

    if device_name == "NVIDIA L20":  # No doc from NVIDIA
        return 119 * (2 / dtype.itemsize)
    # https://www.nvidia.com/en-us/data-center/l4/
    if device_name == "NVIDIA L4":
        return 121 * (2 / dtype.itemsize)
    # https://images.nvidia.com/content/Solutions/data-center/vgpu-L40-datasheet.pdf
    if device_name == "NVIDIA L40":
        return 181 * (2 / dtype.itemsize)
    # https://www.nvidia.com/en-us/data-center/l40s/
    if device_name == "NVIDIA L40S":
        return 366 * (2 / dtype.itemsize)
    # https://www.nvidia.com/en-us/data-center/h100/
    # https://chaoqing-i.com/upload/20231128/NVIDIA%20H800%20GPU%20Datasheet.pdf
    if device_name == "NVIDIA H100" or device_name == "NVIDIA H800":
        return 989 * (2 / dtype.itemsize)
    if device_name == "NVIDIA H20":
        return 148 * (2 / dtype.itemsize)

    logging.warning(
        f"device {device_name} not listed here. calculate tflops by estimation, or you can report it to developers.")
    return None


def get_tensorcore_tflops(dtype: torch.dtype):
    tflops = get_tensorcore_tflops_by_device_name(dtype)
    if tflops is not None:
        return tflops
    return get_full_tflops_approx(dtype=dtype)


def get_dram_gbps_by_device_name(device_name: str):
    _DRAM_GBPS = {
        "NVIDIA L20": 864,
        "NVIDIA L4": 300,
        "NVIDIA L40": 864,
        "NVIDIA L40S": 864,
        "NVIDIA H20": 4000,
        "NVIDIA A100 80GB PCIe": 1935,
        "NVIDIA A100-SXM4-80GB": 2039,
        "NVIDIA A100-SXM4-40GB": 1555,
        "NVIDIA A10": 600,
        "NVIDIA A30": 933,
        "NVIDIA A40": 696,
        "NVIDIA H100 SXM": 3958,
        "NVIDIA H100 NVL": 3341,
        "NVIDIA H800": 3350,
    }
    return _DRAM_GBPS[device_name]


def get_dram_gbps(device=None):
    try:
        return triton.testing.get_dram_gbps(device)
    except Exception:
        return get_dram_gbps_by_device_name(torch.cuda.get_device_name(device))


def estimate_gemm_sol_time_ms(M: int, N: int, K: int, dtype=torch.bfloat16):
    """refer to this: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/"""

    flops = M * N * K * 2
    return flops / get_tensorcore_tflops(dtype=dtype) / 1e9


if __name__ == "__main__":
    print(f"DRAM: {get_dram_gbps():0.2f} GB/s")
    print(f"DRAM by approx: {triton.testing.get_dram_gbps():0.2f} GB/s")
    print(f"DRAM by device name: {get_dram_gbps_by_device_name(torch.cuda.get_device_name(0)):0.2f} GB/s")
    print(f"TFLOPS: {get_tensorcore_tflops(torch.float16):0.2f} TFLOPS")
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"TFLOPS by approx: {get_tensorcore_tflops_by_calc(0, num_sms, 4, torch.float16):0.2f} TFLOPS")
    print(f"TFLOPS by device name: {get_tensorcore_tflops_by_device_name(torch.float16):0.2f} TFLOPS")
    print("")
