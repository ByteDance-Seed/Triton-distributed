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

from ..builtin_base import builtin, Builtin

# ==============================================================================
# __shfl_sync -- read from specific lane
# ==============================================================================


def codegen_shfl_sync(mask, value, src):
    return Builtin(
        body="",
        includes=[],
        return_val=f"__shfl_sync({mask}, {value}, {src})",
    )


@builtin(eval_return_type=lambda mask_type, value_type, src_type: value_type, codegen_func=codegen_shfl_sync)
def __shfl_sync(mask, value, src):
    """
    Warp shuffle value from src lane.
    """
    raise RuntimeError("__shfl_sync should never be called in compilation")


# ==============================================================================
# __shfl_xor_sync -- XOR shuffle (butterfly reduction)
# ==============================================================================


def codegen_shfl_xor_sync(mask, value, lane_mask):
    return Builtin(body="", includes=[], return_val=f"__shfl_xor_sync({mask}, {value}, {lane_mask})")


@builtin(eval_return_type=lambda mask_type, value_type, lane_mask_type: value_type, codegen_func=codegen_shfl_xor_sync)
def shfl_xor_sync(mask, value, lane_mask):
    """Warp shuffle XOR -- exchange with lane (laneId ^ lane_mask)."""
    raise RuntimeError("shfl_xor_sync should never be called in compilation")


# ==============================================================================
# __shfl_up_sync -- shift up (prefix scan)
# ==============================================================================


def codegen_shfl_up_sync(mask, value, delta):
    return Builtin(body="", includes=[], return_val=f"__shfl_up_sync({mask}, {value}, {delta})")


@builtin(eval_return_type=lambda mask_type, value_type, delta_type: value_type, codegen_func=codegen_shfl_up_sync)
def shfl_up_sync(mask, value, delta):
    """Warp shuffle up -- read from lane (laneId - delta)."""
    raise RuntimeError("shfl_up_sync should never be called in compilation")


# ==============================================================================
# __shfl_down_sync -- shift down
# ==============================================================================


def codegen_shfl_down_sync(mask, value, delta):
    return Builtin(body="", includes=[], return_val=f"__shfl_down_sync({mask}, {value}, {delta})")


@builtin(eval_return_type=lambda mask_type, value_type, delta_type: value_type, codegen_func=codegen_shfl_down_sync)
def shfl_down_sync(mask, value, delta):
    """Warp shuffle down -- read from lane (laneId + delta)."""
    raise RuntimeError("shfl_down_sync should never be called in compilation")
