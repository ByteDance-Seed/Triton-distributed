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
import sys

from contextlib import redirect_stdout

LAYER_CONFIGS = {
    "LLaMA-7B": {"N": 11008, "K": 4096},
    "LLaMA-3.1-8B": {"N": 14336, "K": 4096},
    "LLaMA-3.1-70B": {"N": 28672, "K": 8192},
    "LLaMA-3.1-405B": {"N": 53248, "K": 16384},
    "Mistral-7B": {"N": 14336, "K": 4096},
    "Qwen2-72B": {"N": 29568, "K": 8192},
    "GPT-3-175B": {"N": 49152, "K": 12288},
}


def assert_allclose(x: torch.Tensor, y: torch.Tensor, rtol, atol, verbose=True, allow_nan: bool = False,
                    allow_inf: bool = False, fail_if_too_small: bool = False):
    """
    fail_if_too_small: if `x` or `y` is not much larger than `atol * 5`, raise error.
    """
    if not allow_nan:
        if torch.any(x.isnan()):
            raise RuntimeError(f"x has nan: {x}")
        if torch.any(y.isnan()):
            raise RuntimeError(f"y has nan: {y}")

    if not allow_inf:
        if torch.any(x.isinf()):
            raise RuntimeError(f"x has inf: {x}")
        if torch.any(y.isinf()):
            raise RuntimeError(f"y has inf: {y}")

    if fail_if_too_small:
        if (x_abs_mean := torch.abs(x).float().mean()) < atol * 5:
            raise RuntimeError(f"x is too small for atol: {x_abs_mean} vs atol {atol}")
        if (y_abs_mean := torch.abs(y).float().mean()) < atol * 5:
            raise RuntimeError(f"y is too small for atol: {y_abs_mean} vs atol {atol}")

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
