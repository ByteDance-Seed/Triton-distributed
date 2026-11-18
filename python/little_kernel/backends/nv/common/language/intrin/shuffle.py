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
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
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
"""First-class warp shuffle intrinsics (CUDA ``__shfl_*_sync`` family).

These are the language-level replacement for per-frontend ``ll.asm``/inline
CUDA warp-exchange builtins (e.g. half-split partner exchange in rotary
kernels, butterfly warp reductions, warp prefix scans, lane broadcast).

Legality / convergence contract (declared, not just documented):

* Every op emits a ``*_sync`` CUDA intrinsic, i.e. a convergent warp
  collective.  The kernel author must guarantee that every lane named in
  ``mask`` executes the call together under uniform control flow; the
  canonical full-warp mask is ``0xffffffff``.
* The declared memory effect is ``R_RMEM_WARP_SYNC`` on the ``value``
  operand (see little_kernel/core/memory_effects.py): the returned value may
  be read from a DIFFERENT lane's register, so the op must not be CSE'd or
  reordered across a warp synchronization or a divergence boundary.  The op
  has NO GMEM/SMEM/TMEM effect; it is pure with respect to memory spaces
  other than the warp-wide register file, and it is pure under replay with
  the same mask/lane arguments (idempotent read, no write effect).
* The result type equals the ``value`` operand type; CUDA natively supports
  1/2/4/8-byte scalar types (int, unsigned, float, double, long long, half,
  bf16, ...).  Wider values (e.g. 16-byte vectors) are NOT expressible as a
  single shuffle; shuffle their 32-bit components or keep them in-register.
"""

from little_kernel.core.memory_effects import Effect, R_RMEM_WARP_SYNC
from little_kernel.language.builtin_base import builtin, Builtin, BuiltinOp

# Declared effect shared by the whole family: reads a register from an
# arbitrary lane named by the lane/src/delta argument.
_SHFL_EFFECTS = (Effect(R_RMEM_WARP_SYNC, "value"), )

# ==============================================================================
# shfl_sync -- read from specific lane
# ==============================================================================


@builtin
class shfl_sync(BuiltinOp):
    """Warp shuffle value from src lane.

    Emits ``__shfl_sync(mask, value, src)``: every lane named in ``mask``
    receives ``value`` from lane ``src`` (uniform across participating
    lanes).  Convergent warp collective -- see module docstring.
    """

    effects = _SHFL_EFFECTS

    @staticmethod
    def return_type(mask_type, value_type, src_type):
        return value_type

    @staticmethod
    def codegen(mask, value, src):
        return Builtin(
            body="",
            includes=[],
            return_val=f"__shfl_sync({mask}, {value}, {src})",
        )


# Backward-compatible alias: the indexed shuffle shipped under the
# dunder-prefixed name first; frontends (and design/ helpers) that already
# use ``ll.__shfl_sync`` keep working.
__shfl_sync = shfl_sync

# ==============================================================================
# shfl_xor_sync -- XOR shuffle (butterfly reduction)
# ==============================================================================


@builtin
class shfl_xor_sync(BuiltinOp):
    """Warp shuffle XOR -- exchange with lane (laneId ^ lane_mask).

    Emits ``__shfl_xor_sync(mask, value, lane_mask)``.  Butterfly reduction
    building block.  Convergent warp collective -- see module docstring.
    """

    effects = _SHFL_EFFECTS

    @staticmethod
    def return_type(mask_type, value_type, lane_mask_type):
        return value_type

    @staticmethod
    def codegen(mask, value, lane_mask):
        return Builtin(body="", includes=[], return_val=f"__shfl_xor_sync({mask}, {value}, {lane_mask})")


# ==============================================================================
# shfl_up_sync -- shift up (prefix scan)
# ==============================================================================


@builtin
class shfl_up_sync(BuiltinOp):
    """Warp shuffle up -- read from lane (laneId - delta).

    Emits ``__shfl_up_sync(mask, value, delta)``.  Warp prefix-scan building
    block; lanes with ``laneId < delta`` keep their own value.  Convergent
    warp collective -- see module docstring.
    """

    effects = _SHFL_EFFECTS

    @staticmethod
    def return_type(mask_type, value_type, delta_type):
        return value_type

    @staticmethod
    def codegen(mask, value, delta):
        return Builtin(body="", includes=[], return_val=f"__shfl_up_sync({mask}, {value}, {delta})")


# ==============================================================================
# shfl_down_sync -- shift down
# ==============================================================================


@builtin
class shfl_down_sync(BuiltinOp):
    """Warp shuffle down -- read from lane (laneId + delta).

    Emits ``__shfl_down_sync(mask, value, delta)``; lanes with
    ``laneId + delta >= width`` keep their own value.  Convergent warp
    collective -- see module docstring.
    """

    effects = _SHFL_EFFECTS

    @staticmethod
    def return_type(mask_type, value_type, delta_type):
        return value_type

    @staticmethod
    def codegen(mask, value, delta):
        return Builtin(body="", includes=[], return_val=f"__shfl_down_sync({mask}, {value}, {delta})")


# ==============================================================================
# Width-parameterized variants (segmented shuffles)
#
# CUDA's __shfl_*_sync intrinsics take an optional 4th `width` argument that
# partitions the warp into independent power-of-two segments of `width` lanes
# (each segment behaves like its own warp: deltas do not cross segment
# boundaries). This is the generic primitive for row-parallel reductions that
# place multiple rows per warp (e.g. 2 rows x 16 lanes). Legality contract
# (caller's responsibility, same as the CUDA intrinsic):
#   * width must be a power of two in [1, 32];
#   * all lanes named in `mask` must execute the same shfl with the same
#     width (convergence requirement, unchanged from the 3-arg forms).
# ==============================================================================


@builtin
class shfl_sync_width(BuiltinOp):
    """Segmented shuffle from lane (segBase + srcLane), width lanes/segment."""

    @staticmethod
    def return_type(mask_type, value_type, src_type, width_type):
        return value_type

    @staticmethod
    def codegen(mask, value, src, width):
        return Builtin(body="", includes=[], return_val=f"__shfl_sync({mask}, {value}, {src}, {width})")


@builtin
class shfl_xor_sync_width(BuiltinOp):
    """Segmented XOR shuffle, width lanes/segment."""

    @staticmethod
    def return_type(mask_type, value_type, lane_mask_type, width_type):
        return value_type

    @staticmethod
    def codegen(mask, value, lane_mask, width):
        return Builtin(body="", includes=[], return_val=f"__shfl_xor_sync({mask}, {value}, {lane_mask}, {width})")


@builtin
class shfl_up_sync_width(BuiltinOp):
    """Segmented shuffle up, width lanes/segment."""

    @staticmethod
    def return_type(mask_type, value_type, delta_type, width_type):
        return value_type

    @staticmethod
    def codegen(mask, value, delta, width):
        return Builtin(body="", includes=[], return_val=f"__shfl_up_sync({mask}, {value}, {delta}, {width})")


@builtin
class shfl_down_sync_width(BuiltinOp):
    """Segmented shuffle down, width lanes/segment."""

    @staticmethod
    def return_type(mask_type, value_type, delta_type, width_type):
        return value_type

    @staticmethod
    def codegen(mask, value, delta, width):
        return Builtin(body="", includes=[], return_val=f"__shfl_down_sync({mask}, {value}, {delta}, {width})")
