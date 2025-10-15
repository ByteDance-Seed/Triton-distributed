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

from pathlib import Path
import argparse
import glob

import pandas as pd
import torch
import torch.multiprocessing as mp
import logging
import triton
from triton_dist.kernels.nvidia.gemm import (matmul, matmul_descriptor_persistent, matmul_persistent, matmul_tma,
                                             matmul_tma_persistent)
from triton_dist.profiler_utils import perf_func
from triton_dist.utils import wait_until_max_gpu_clock_or_warning
from triton_dist.tune import load_autotune_data, pretty_triton_config_repr

MATMUL_FUNC = {
    "matmul_torch": torch.matmul,
    "matmul": matmul,
    "matmul_tma": matmul_tma,
    "matmul_persistent": matmul_persistent,
    "matmul_tma_persistent": matmul_tma_persistent,
    "matmul_descriptor_persistent": matmul_descriptor_persistent,
}

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", type=str, default="matmul", choices=MATMUL_FUNC.keys())
    parser.add_argument("--dtype", type=str, default="float16")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--M", "-M", type=int)
    group.add_argument("--M_range", "--M-range", type=str, help="M range, format: start-end-step, such as 128-1024-128")
    parser.add_argument("--N", "-N", type=int, default=2048)
    parser.add_argument("--K", "-K", type=int, default=2048)
    parser.add_argument("--warp_specialize", type=bool, default=False)
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--inspect", default=False, action="store_true")
    parser.add_argument("--list", default=False, action="store_true")
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def parse_range(range_str):
    start, end, step = range_str.split("-")
    start, end, step = int(start), int(end), int(step)
    assert start < end
    assert step > 0
    return start, end, step


def pretty_time(duration_ms):
    if duration_ms < 0.1:
        return f"{duration_ms * 1000:.2f} us"
    if duration_ms < 1000:
        return f"{duration_ms:.3f} ms"
    return f"{duration_ms / 1000:.2f} s"


def perf_matmul(M, N, K, trans_b, dtype, func, iters=10, warmup_iters=5, verbose=False):

    wait_until_max_gpu_clock_or_warning()
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    if trans_b:
        B = torch.randn(N, K, dtype=dtype, device="cuda").T
    else:
        B = torch.randn(K, N, dtype=dtype, device="cuda")

    try:
        _ = func(A, B, autotune_verbose=verbose)
    except TypeError:
        func(A, B)

    fn = lambda: func(A, B)
    _, duration_ms = perf_func(fn, iters, warmup_iters)

    memory_read_gb = (dtype.itemsize * M * K + dtype.itemsize * K * N) / 2**30
    memory_write_gb = dtype.itemsize * M * N / 2**30
    tflops = 2 * M * N * K / 1e12
    duration = duration_ms / 1000
    logging.info(
        f"[{M}, {N}, {K}, {dtype}] {pretty_time(duration_ms)}, read {memory_read_gb / duration:.2f} GB/s write {memory_write_gb / duration:.2f} GB/s, {tflops / duration:.2f} TFLOPS"
    )

    return duration_ms


def perf_matmuls(device_id, M_list, N, K, trans_b, dtype, impl, global_results, iters=10, warmup_iters=5,
                 verbose=False):
    func = MATMUL_FUNC[impl]
    local_results = {}
    import pynvml
    pynvml.nvmlInit()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    with torch.cuda.device(device_id):
        for M in M_list:
            local_results[M] = perf_matmul(M, N, K, trans_b, dtype, func, iters=iters, warmup_iters=warmup_iters,
                                           verbose=verbose)

    global_results[device_id] = local_results
    pynvml.nvmlShutdown()
    return local_results


class IntFilter:
    """
    A class to filter integers based on a defined rule.
    The rule can be an integer, a [min, max] list for a range, or None for a wildcard.
    """

    def __init__(self, rule):
        """
        Initializes the filter with a specific rule.

        Args:
            rule: Can be None (matches all), an int (exact match),
                  or a list of two ints [min, max] (range match).

        Raises:
            TypeError: If the rule is not an int, list, or None.
            ValueError: If the list rule is not properly formatted.
        """
        if rule is None or isinstance(rule, int):
            self.rule = rule
        elif isinstance(rule, list):
            if len(rule) != 2 or not all(isinstance(i, int) for i in rule):
                raise ValueError("List rule must be a [min, max] pair of two integers.")
            if rule[0] > rule[1]:
                raise ValueError("In a range [min, max], min cannot be greater than max.")
            self.rule = rule
        else:
            raise TypeError("Rule must be an int, a list of two ints, or None.")

    def match(self, val: int) -> bool:
        """
        Checks if a given integer value matches the filter's rule.

        Args:
            val: The integer value to check.

        Returns:
            True if the value matches, False otherwise.
        """
        # Rule is None: This is a wildcard and matches any integer.
        if self.rule is None:
            return True

        # Rule is an int: Check for an exact match.
        if isinstance(self.rule, int):
            return val == self.rule

        # Rule is a list: Check if the value is within the range [min, max].
        if isinstance(self.rule, list):
            min_val, max_val = self.rule
            return min_val <= val <= max_val

        return False  # Should not be reached due to __init__ validation

    def __repr__(self):
        """Provides a clear string representation of the filter object."""
        return f"IntFilter(rule={self.rule})"

    def is_int(self):
        return isinstance(self.rule, int)


def transpose_2d(mat: list[list[int]]) -> list[list[int]]:
    return [[mat[j][i] for j in range(len(mat))] for i in range(len(mat[0]))]


def get_matched_autotune_files(M: IntFilter, N: IntFilter, K: IntFilter, dtype, func, verbose):
    func_dir = func.func_dir()
    autotune_files = glob.glob(f"{func_dir}/*.json")
    autotune_configs = {}

    matched_autotune_files = []
    for autotune_file in autotune_files:
        autotune_data = load_autotune_data(autotune_file)
        (A_shape, A_stride, A_dtype), (B_shape, B_stride, B_dtype) = autotune_data["key"]
        config = autotune_data["config"]
        M_, K_ = A_shape
        _, N_ = B_shape
        if M.match(M_) and K.match(K_) and N.match(N_) and A_dtype == B_dtype == dtype:
            autotune_configs[(M_, N_, K_, dtype)] = config
            matched_autotune_files.append(autotune_file)

    return matched_autotune_files


def list_matched_autotune_files(M: IntFilter, N: IntFilter, K: IntFilter, dtype, func, verbose):
    matched_autotune_files = get_matched_autotune_files(M, N, K, dtype, func, verbose)
    for autotune_file in matched_autotune_files:
        print(autotune_file)


def inspect_autotune(M: IntFilter, N: IntFilter, K: IntFilter, dtype, func, verbose):
    func_dir = func.func_dir()
    autotune_files = glob.glob(f"{func_dir}/*.json")
    autotune_configs = {}

    for autotune_file in autotune_files:
        autotune_data = load_autotune_data(autotune_file)
        (A_shape, A_stride, A_dtype), (B_shape, B_stride, B_dtype) = autotune_data["key"]
        B_stride_k, B_stride_n = B_stride
        trans_b = B_stride_k == 1
        config = autotune_data["config"]
        M_, K_ = A_shape
        _, N_ = B_shape
        if M.match(M_) and K.match(K_) and N.match(N_) and A_dtype == B_dtype == dtype and trans_b == args.trans_b:
            autotune_configs[(M_, N_, K_, dtype, trans_b)] = config

    if verbose:
        for (M_, N_, K_, dtype_, trans_b_), config in autotune_configs.items():
            best_duration = min([x[0] for x in config])
            config = [x for x in config if x[0] < best_duration * 1.15]
            print(f"key: {M_}, {N_}, {K_}, {dtype_}, {trans_b_}")
            with pretty_triton_config_repr():
                for c in config:
                    print(c)

    all_configs = []
    for _, configs in autotune_configs.items():
        all_configs.extend([x["config"] for _, x in configs])
    all_configs = set(all_configs)

    t_relative = []
    t_absolute = []
    for _, configs in autotune_configs.items():
        best_duration = min([t for t, _ in configs])
        d = {**{c: float("inf") for c in all_configs}, **{c["config"]: t / best_duration for t, c in configs}}
        t_relative.append([d[c] for c in all_configs])
        d = {**{c: float("inf") for c in all_configs}, **{c["config"]: t for t, c in configs}}
        t_absolute.append([d[c] for c in all_configs])

    if not t_relative:
        print("No autotune configs found, exit...")
        return
    t_relative = transpose_2d(t_relative)
    t_absolute = transpose_2d(t_absolute)

    def _as_index(config: triton.Config):
        BLOCK_SIZE_M = config.kwargs["BLOCK_SIZE_M"]
        BLOCK_SIZE_N = config.kwargs["BLOCK_SIZE_N"]
        BLOCK_SIZE_K = config.kwargs["BLOCK_SIZE_K"]
        GROUP_SIZE_M = config.kwargs["GROUP_SIZE_M"]
        nwarps = config.num_warps
        nstages = config.num_stages
        ep_suffix = ""
        if "EPILOGUE_SUBTILE" in config.kwargs:
            EPILOGUE_SUBTILE = config.kwargs["EPILOGUE_SUBTILE"]
            if EPILOGUE_SUBTILE:
                ep_suffix = "_epilogue_subtile"
            else:
                ep_suffix = "_no_epilogue_subtile"

        return f"M_{BLOCK_SIZE_M}_N_{BLOCK_SIZE_N}_K_{BLOCK_SIZE_K}_GM_{GROUP_SIZE_M}_nwarps_{nwarps}_nstages_{nstages}{ep_suffix}"

    df = pd.DataFrame(t_relative, columns=list(autotune_configs.keys()), index=[_as_index(c) for c in all_configs])
    df = df[sorted(df.columns)]
    df["mean"] = df[list(autotune_configs.keys())].mean(axis=1)
    df = df.sort_values("mean")
    print(df)
    func_name = func.func.__name__
    if N.is_int():
        func_name += f"_N_{N.rule}"
    if K.is_int():
        func_name += f"_K_{K.rule}"
    df.to_csv(f"autotune_{func_name}_rel.csv")

    df = pd.DataFrame(t_absolute, columns=list(autotune_configs.keys()), index=[_as_index(c) for c in all_configs])
    df = df[sorted(df.columns)]
    df["mean"] = df[list(autotune_configs.keys())].mean(axis=1)
    df = df.sort_values("mean")
    print(df)
    # use this to compare w/o TMA or w/o persistent GEMM implementation
    df.to_csv(f"autotune_{func_name}_abs.csv")


def do_list(args):
    dtype = DTYPE_MAP[args.dtype]
    K, N = args.K, args.N
    if args.M:
        M_filter = IntFilter(args.M)
    else:
        M_start, M_end, M_step = parse_range(args.M_range)
        M_filter = IntFilter([M_start, M_end])
    list_matched_autotune_files(M_filter, IntFilter(N), IntFilter(K), dtype, MATMUL_FUNC[args.impl], args.verbose)


def do_inspect(args):
    dtype = DTYPE_MAP[args.dtype]
    K, N = args.K, args.N
    if args.M:
        M_filter = IntFilter(args.M)
    else:
        M_start, M_end, M_step = parse_range(args.M_range)
        M_filter = IntFilter([M_start, M_end])
    inspect_autotune(M_filter, IntFilter(N), IntFilter(K), dtype, MATMUL_FUNC[args.impl], args.verbose)


def do_tune(args):
    dtype = DTYPE_MAP[args.dtype]
    K, N = args.K, args.N
    Ms = [args.M] if args.M else range(*parse_range(args.M_range))
    perf_results = {}

    if args.ngpus > 1:
        mp.set_start_method("spawn", force=True)
        num_gpus = args.ngpus
        if num_gpus < args.ngpus:
            logging.warning(f"Warning: --ngpus set to {args.ngpus}, but only {num_gpus} GPUs are available.")
        work_chunks = [Ms[i::num_gpus] for i in range(num_gpus)]
        with mp.Manager() as manager:
            results_dict = manager.dict()
            processes = []
            for i, M_list in enumerate(work_chunks):
                p = mp.Process(
                    target=perf_matmuls, args=(i, M_list, N, K, args.trans_b, dtype, args.impl, results_dict,
                                               args.iters, args.warmup_iters, args.verbose))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()

            for i in range(num_gpus):
                local_results = results_dict[i]
                perf_results.update(local_results)

    else:
        global_results = {}
        perf_matmuls(0, Ms, N, K, args.trans_b, dtype, args.impl, global_results, iters=args.iters,
                     warmup_iters=args.warmup_iters, verbose=args.verbose)
        perf_results = global_results[0]

    gpu_name = torch.cuda.get_device_name().replace(" ", "_")
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{gpu_name}_{timestamp}"
    outfile = Path(f"autotune_{args.impl}_M_{args.M_range}_N_{N}_K_{K}_dtype_{args.dtype}_{suffix}.csv")
    with open(outfile, "w") as f:
        f.write("M,N,K,duration_ms\n")
        for M in Ms:
            duration_ms = perf_results.get(M, float("inf"))
            f.write(f"{M},{N},{K},{duration_ms}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]
    K, N = args.K, args.N

    if args.list:
        do_list(args)
    elif args.inspect:
        do_inspect(args)
    else:
        do_tune(args)
