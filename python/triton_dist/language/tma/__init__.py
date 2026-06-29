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
"""Device-side TMA abstractions for Triton-distributed.

Submodules
----------
  ``tensormap`` -- device-side tensormap creation constants and helpers.
  ``pipeline``  -- PipelineState, TmaPipeline aggregate types.

One-stop import::

    from triton_dist.language.tma import (
        PipelineState, TmaPipeline, create_tmap_2d, ELEM_BF16, SWIZZLE_NONE,
    )
"""
from .tensormap import (
    ELEM_BF16,
    ELEM_F32,
    FILL_NONE,
    INTERLEAVE_NONE,
    SWIZZLE_128B,
    SWIZZLE_NONE,
    create_tmap_2d,
    tmap_update_address,
)
from .pipeline import (
    PipelineState,
    TmaPipeline,
    store_and_wait,
)

__all__ = [
    # tensormap
    "ELEM_BF16",
    "ELEM_F32",
    "SWIZZLE_128B",
    "SWIZZLE_NONE",
    "INTERLEAVE_NONE",
    "FILL_NONE",
    "create_tmap_2d",
    "tmap_update_address",
    # pipeline
    "PipelineState",
    "TmaPipeline",
    "store_and_wait",
]
