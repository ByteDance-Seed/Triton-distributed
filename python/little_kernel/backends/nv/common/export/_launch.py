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
"""Launch-contract validation and planning.

One schema for the top-level launch record and every pipeline stage: grid
shape expressions, static/shape-dependent block dimensions, per-axis caps,
dynamic shared memory, occupancy requests and cluster shapes. Planning fills
the resolved fields emitters read without re-deriving any rule.
"""

from __future__ import annotations

from little_kernel.backends.nv.common.export._schema import LAUNCH_KEYS, _identifier, _positive_int, reject_unknown_fields
from little_kernel.backends.nv.common.export._shapes import _shape_expr


def _validate_launch(launch: dict, ranks: dict, launch_scalars: set, optional_inputs: set, host_lists: dict) -> dict:
    """Validate one launch record (grid/block/shared/cluster) and annotate it.

    Shared by the top-level launch record and every pipeline stage record so
    both observe exactly the same schema and host-side shape-expression
    language. Mutates ``launch`` in place, filling the ``_grid_cpp``,
    ``_block_cpp``, ``_block``, ``_block_expr``, ``_block_dynamic``,
    ``_threads``, ``_shared``, ``_shared_cpp``, ``_min_blocks``, ``_cluster``
    and ``_min_blocks_per_sm`` fields the renderers read.
    """
    for key in ("grid", "block"):
        values = launch.get(key)
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"launch.{key} must contain three values")
    # Single-schema ownership: a typo in a launch option is a hard error.
    reject_unknown_fields(launch, LAUNCH_KEYS, "launch")
    # Launch expressions are host-side sizes resolved at launch time; an
    # optional tensor's sizes are never valid there, so its name is removed
    # from the ranks map and any reference fails validation.
    launch_ranks = {key: value for key, value in ranks.items() if key not in optional_inputs}
    launch["_grid_cpp"] = [
        _shape_expr(value, launch_ranks, launch_scalars, optional_inputs=optional_inputs, lists=host_lists)
        for value in launch["grid"]
    ]
    # launch.block entries are positive ints, or _shape_expr strings evaluated
    # on the host at launch time (same language as launch.grid). Kernel
    # indexing uses runtime blockDim, so only __launch_bounds__/num_threads
    # needs a static bound. The resolved launch record carries the full
    # schema every downstream consumer reads (``_threads``/``_block_cpp`` and
    # ``_block``/``_block_expr``/``_block_dynamic``): dynamic manifests (any
    # expression dimension) may declare per-axis caps ("block_cap") and/or a
    # product cap ("block_max_threads"); missing per-axis caps default to the
    # CUDA per-axis hardware limits, a missing product cap to the caps
    # product, so cap checks still imply the 1024-thread bound before launch.
    block_cpp = []
    block_const = []
    for value in launch["block"]:
        if isinstance(value, str):
            block_cpp.append(_shape_expr(value, launch_ranks, launch_scalars, lists=host_lists))
            block_const.append(None)
        elif type(value) is int:
            const = _positive_int(value, "block dimension")
            block_const.append(const)
            block_cpp.append(str(const))
        else:
            raise ValueError("block dimension must be a positive integer or shape expression")
    launch["_block_cpp"] = block_cpp
    launch["_block_expr"] = [const is None for const in block_const]
    launch["_block_dynamic"] = any(launch["_block_expr"])
    if not launch["_block_dynamic"]:
        launch["_block"] = block_const
        threads = block_const[0] * block_const[1] * block_const[2]
        if threads > 1024:
            raise ValueError("block has more than 1024 threads")
    else:
        cap = launch.get("block_cap")
        if cap is not None:
            if not isinstance(cap, list) or len(cap) != 3:
                raise ValueError("launch.block_cap must list three axis caps when block uses shape expressions")
            launch["_block"] = [_positive_int(value, "block_cap dimension") for value in cap]
        else:
            launch["_block"] = [1024, 64, 64]
        for const, cap_value in zip(block_const, launch["_block"]):
            if const is not None and const > cap_value:
                raise ValueError("constant block dimensions must not exceed block_cap")
        for cap_value, axis, limit in zip(launch["_block"], "xyz", (1024, 64, 64)):
            if cap_value > limit:
                raise ValueError(f"block_cap {axis} exceeds the CUDA per-axis limit of {limit}")
        # A declared block_max_threads is the product cap (checked in
        # [1, 1024] below), so per-axis caps may span the hardware box and
        # launch legality is deferred to the emitted host-side product check;
        # when the product cap is absent it defaults to the caps product,
        # which becomes _threads (the compile-time
        # num_threads/__launch_bounds__ ceiling) and must itself respect the
        # 1024-thread bound.
        threads = launch.get("block_max_threads")
        if threads is None:
            threads = launch["_block"][0] * launch["_block"][1] * launch["_block"][2]
            if threads > 1024:
                raise ValueError("block has more than 1024 threads")
        elif type(threads) is not int or not 0 < threads <= 1024:
            raise ValueError("shape-dependent launch.block requires launch.block_max_threads in [1, 1024]")
        elif any(const is not None and const > threads for const in block_const):
            raise ValueError("constant block dimensions must not exceed block_max_threads")
    launch["_threads"] = threads
    shared = launch.get("shared_memory_bytes", 0)
    if shared == "auto" or type(shared) is int:
        launch["_shared"] = shared
        launch["_shared_cpp"] = None
    elif isinstance(shared, str):
        launch["_shared"] = None
        launch["_shared_cpp"] = _shape_expr(shared, launch_ranks, launch_scalars, lists=host_lists)
    else:
        raise ValueError("shared_memory_bytes must be nonnegative, 'auto' or a shape expression")
    if type(launch["_shared"]) is int and launch["_shared"] < 0:
        raise ValueError("shared_memory_bytes must be nonnegative, 'auto' or a shape expression")
    if type(launch.get("disable_var_hoisting", False)) is not bool:
        raise ValueError("disable_var_hoisting must be boolean")
    min_blocks = launch.get("max_blocks_per_sm")
    if min_blocks is not None and (type(min_blocks) is not int or not 1 <= min_blocks <= 32):
        raise ValueError("max_blocks_per_sm must be an integer in [1, 32]")
    launch["_min_blocks"] = min_blocks
    cluster = launch.get("cluster_shape")
    if cluster is not None:
        if not isinstance(cluster, list) or len(cluster) != 3:
            raise ValueError("cluster_shape must contain three dimensions")
        launch["_cluster"] = [_positive_int(value, "cluster dimension") for value in cluster]
        if launch["_cluster"][0] * launch["_cluster"][1] * launch["_cluster"][2] > 8:
            raise ValueError("cluster has more than eight CTAs")
    else:
        launch["_cluster"] = None
    min_blocks = launch.get("min_blocks_per_sm")
    if min_blocks is not None:
        # Opt-in occupancy request: the entry kernel is emitted with
        # __launch_bounds__(blockThreads, min_blocks_per_sm) so ptxas honors
        # the frontend's min-blocks register budget. Absent the field the
        # generated source is byte-identical to a plain __global__.
        launch["_min_blocks_per_sm"] = _positive_int(min_blocks, "min_blocks_per_sm")
        # Reject num_threads * min_blocks_per_sm products that no supported
        # GPU can schedule (2048 threads-per-SM is the architecture ceiling);
        # `threads` is the compile-time block-thread ceiling that becomes the
        # __launch_bounds__ first argument, so this condition matches the
        # emitted launch legality exactly.
        if threads * launch["_min_blocks_per_sm"] > 2048:
            raise ValueError("min_blocks_per_sm * block threads would exceed the 2048 "
                             "threads-per-SM hardware limit")
    else:
        launch["_min_blocks_per_sm"] = None
    return launch
