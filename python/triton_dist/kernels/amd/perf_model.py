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
from triton_dist.amd_utils import has_amdsmi, get_num_xcds_by_amdsmi, get_max_gpu_clock_rate_in_khz
import torch
from triton.runtime.driver import driver

NUM_XCDS_BY_DEVICE_NAME = {"AMD Instinct MI308X": 4, "AMD Instinct MI300X": 8}


def get_num_xcds(device_id: int = 0):
    try:
        if has_amdsmi():
            return get_num_xcds_by_amdsmi(device_id)
    except Exception:
        return NUM_XCDS_BY_DEVICE_NAME[torch.cuda.get_device_name(device_id)]


def get_max_shared_memory_size(device_id: int = 0):
    return driver.active.utils.get_device_properties(device_id)["max_shared_mem"]


def get_cu_count(device_id: int):
    return torch.cuda.get_device_properties(device_id).multi_processor_count


def get_max_tensorcore_tflops_by_hw_spec(dtype: torch.dtype, device_id: int):
    # Assume MI300X-like CDNA3: 4x4 matrix cores per SIMD, 4 SIMD per CU, 304 CUs, 2.1 GHz
    # Each core does 4x4x4 = 64 FMAs/cycle -> 128 FP16 TFLOPS per core
    # Total TFLOPS = 304 CU * 4 SIMD/CU * 4 cores/SIMD * 128 ops/cycle * 2.1 GHz / 1e12
    # Simplified: 304 * 4 * 4 * 128 * 2.1 / 1e12 ≈ 1.3 TFLOPS (FP16)
    # For FP32 accumulate, divide by 2 -> ~0.65 TFLOPS
    cus = get_cu_count(device_id)
    smid_per_cu = 4
    cores_per_simd = 4
    ops_per_core_per_cycle = 128
    freq_ghz = get_max_gpu_clock_rate_in_khz() / 1e6
    tflops_fp16 = cus * smid_per_cu * cores_per_simd * ops_per_core_per_cycle * freq_ghz / 1e3
    return tflops_fp16 * 2 / dtype.itemsize


def get_max_tensorcore_tflops_by_name(dtype: torch.dtype, device_id: int):
    device_name = torch.cuda.get_device_name(device_id)
    factor = 2 / dtype.itemsize
    if device_name in ["AMD Instinct MI300X", "AMD Instinct MI325X"]:
        # MI300X: 304 CU, 4x4x4 cores, 2.1 GHz -> ~1.3 TFLOPS (FP16)
        # https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
        # https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
        # https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html
        return 1307 * factor
    elif device_name == "AMD Instinct MI308X":
        return 232 * factor
    elif device_name == "AMD Instinct MI350X":
        # https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html
        # https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi350x-gpu-brochure.pdf
        return 2309.6 * factor
    else:
        return -1


def get_max_tensorcore_tflops(dtype: torch.dtype, device_id: int):
    if tflops := get_max_tensorcore_tflops_by_name(dtype, device_id) > 0:
        return tflops
    return get_max_tensorcore_tflops_by_hw_spec(dtype, device_id)


def get_dram_gbps_by_device_name(device_id: int):
    device_name = torch.cuda.get_device_name(device_id)
    return {
        "AMD Instinct MI308X": 5.3 * 1e3,  # TODO(houqi.1993)
        "AMD Instinct MI300X": 5.3 * 1e3, "AMD Instinct MI325X": 6 * 1e3, "AMD Instinct MI350X": 8 * 1e3
    }.get(device_name, -1)


def get_dram_gbps(device_id: int):
    gbps = get_dram_gbps_by_device_name(device_id)
    if gbps > 0:
        return gbps
    # fallback: query via driver if available
    return -1
