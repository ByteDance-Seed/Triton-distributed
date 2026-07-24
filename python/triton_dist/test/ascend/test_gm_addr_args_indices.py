# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the MIT License.
#
# This test constructs MLIR IR containing distributed dialect operators as strings,
# calls add_convert_triton_distributed_to_hivm to convert them to hivm.custom,
# and verifies that the gm_addr_args_indices attribute on each custom operator is correct.
# Pure IR-level test, does not require NPU hardware.
#
# gm_addr_args_indices records operand indices in custom operators that are:
#   1. Of type triton::PointerType (scalar !tt.ptr<T>);
#   2. Function entry block arguments (tt.func parameters).
# Tensor pointers (tensor<...x!tt.ptr<T>>), pointers that are not function parameters
# (such as results from tt.addptr), and non-pointer operands should not appear in it.
#
# Note: conftest.py in this directory imports torch at the top level. In environments
# without NPU driver installed, set TORCH_DEVICE_BACKEND_AUTOLOAD=0 before running.
# This test itself does not depend on torch or NPU driver.

import re
import tempfile
import os

from triton._C import libtriton


def _make_context():
    """Construct an MLIRContext with all relevant dialects loaded."""
    ir = libtriton.ir
    distributed = libtriton.distributed
    ascend = libtriton.ascend
    ctx = ir.context()
    ctx.disable_multithreading()
    ir.load_dialects(ctx)  # triton + built-in dialect
    distributed.ir.load_dialects(ctx)  # distributed dialect
    ascend.ir.load_dialects(ctx)  # hivm + annotation + scope
    ascend.load_dialects(ctx)  # triton ascend dialect
    return ctx


def _parse(ctx, ir_str):
    """Construct IR from string, write to temp file and parse as module."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
        f.write(ir_str)
        path = f.name
    try:
        return libtriton.ir.parse_mlir_module(path, ctx)
    finally:
        os.unlink(path)


def _run_pass(module, ctx):
    """Run the distributed -> hivm conversion pass on the module."""
    pm = libtriton.ir.pass_manager(ctx)
    libtriton.distributed.ascend_passes.ttgpuir.add_convert_triton_distributed_to_hivm(pm)
    pm.run(module)


def _extract_gm_addr_indices(module_str):
    """Extract symbol and gm_addr_args_indices from each hivm.custom operator in printed IR.

    Returns [(symbol, [indices...]), ...] in order of appearance.
    """
    results = []
    for line in module_str.splitlines():
        if "hir.custom" not in line:
            continue
        sym_match = re.search(r'symbol = "([^"]+)"', line)
        idx_match = re.search(r"gm_addr_args_indices = array<i32([^>]*)>", line)
        symbol = sym_match.group(1) if sym_match else "?"
        if not idx_match:
            indices = None  # attribute missing
        else:
            body = idx_match.group(1)
            # shaped like ": 0, 1" or "" (empty array)
            body = body.lstrip(":").strip()
            if body:
                indices = [int(x.strip()) for x in body.split(",")]
            else:
                indices = []
        results.append((symbol, indices))
    return results


# An IR segment containing multiple distributed operators:
#   %arg0 : !tt.ptr<f32>  -- function parameter, scalar pointer, used as sigAddr for notify, should be captured
#   %arg1 : i32           -- function parameter, non-pointer, should not be captured
#   %arg2 : !tt.ptr<i32>  -- function parameter, scalar pointer, used as barrierPtr for wait, should be captured
#   %arg3 : !tt.ptr<f32>  -- function parameter, scalar pointer, used as symmAddr for symm_at, should be captured
#   %addptr               -- tt.addptr result, pointer but not a function parameter, used as input for consume_token,
#                            should not be captured (verify non-parameter pointers are excluded).
# Results of each operator are consumed by tt.store to avoid being removed by DCE.
WAIT_CONSUME_IR = r"""
module {
  tt.func public @kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                         %arg1: i32 {tt.divisibility = 16 : i32},
                         %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                         %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %addptr = tt.addptr %arg0, %c1_i32 : !tt.ptr<f32>, i32
    %tok = distributed.wait %arg2, %arg1, %c1_i32, scope = gpu semantic = acquire : !tt.ptr<i32>, i32, i32 -> i32
    %ct = distributed.consume_token %addptr, %tok, : !tt.ptr<f32>, i32 -> !tt.ptr<f32>
    %remote = distributed.symm_at %arg3, %arg1, : !tt.ptr<f32>
    "distributed.notify"(%arg0, %c1_i64, %arg1) {sigOp = 1 : i32, commScope = 1 : i32} : (!tt.ptr<f32>, i64, i32) -> ()
    %p0 = tt.splat %ct : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
    tt.store %p0, %cst : tensor<1x!tt.ptr<f32>>
    %p1 = tt.splat %remote : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
    tt.store %p1, %cst : tensor<1x!tt.ptr<f32>>
    tt.return
  }
}
"""


def test_gm_addr_args_indices_wait_consume_notify():
    """Verify gm_addr_args_indices after wait/consume_token/symm_at/notify conversion."""
    ctx = _make_context()
    module = _parse(ctx, WAIT_CONSUME_IR)
    _run_pass(module, ctx)
    out = module.str()

    indices = _extract_gm_addr_indices(out)
    assert indices, "No hivm.custom operators found after conversion"
    by_symbol = {sym: idx for sym, idx in indices}

    # wait: operands (%arg2: ptr, %arg1: i32, %c1: i32), only %arg2 is a pointer parameter -> [0]
    assert by_symbol["aclshmem_wait_int32"] == [0], (f"wait's gm_addr_args_indices should be [0], actual: "
                                                     f"{by_symbol['aclshmem_wait_int32']}")
    # consume_token: operands (%addptr: ptr non-parameter, %tok: i32), no pointer parameters -> []
    assert by_symbol["aclshmem_consume_token_float_ptr_1d"] == [], (
        f"consume_token's gm_addr_args_indices should be [], actual: "
        f"{by_symbol['aclshmem_consume_token_float_ptr_1d']}")
    # symm_at: operands (%arg3: ptr parameter, %arg1: i32), only %arg3 -> [0]
    #   symbol is derived from symmAddr's pointee type (f32 -> "float"), i.e., aclshmem_ptr_float
    assert by_symbol["aclshmem_ptr_float"] == [0], (f"symm_at's gm_addr_args_indices should be [0], actual: "
                                                    f"{by_symbol['aclshmem_ptr_float']}")
    # notify: operands (%arg0: ptr parameter, %c1_i64: i64, %arg1: i32), only %arg0 -> [0]
    assert by_symbol["aclshmem_int64_p"] == [0], (f"notify's gm_addr_args_indices should be [0], actual: "
                                                  f"{by_symbol['aclshmem_int64_p']}")


# get_rank / get_num_ranks clear their operands to ValueRange() during conversion,
# so the custom operators have no operands, and gm_addr_args_indices should be an empty array.
# Both are Pure operators, and their results need to be consumed to avoid DCE removal; here we write them out via tt.store.
GET_RANK_IR = r"""
module {
  tt.func public @kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                         %arg1: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %rank = distributed.get_rank %arg1 : i32
    %nranks = distributed.get_num_ranks %arg1 : i32
    %p0 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %r0 = tt.splat %rank : i32 -> tensor<1xi32>
    tt.store %p0, %r0 : tensor<1x!tt.ptr<i32>>
    %n0 = tt.splat %nranks : i32 -> tensor<1xi32>
    tt.store %p0, %n0 : tensor<1x!tt.ptr<i32>>
    tt.return
  }
}
"""


def test_gm_addr_args_indices_get_rank_empty():
    """get_rank/get_num_ranks have no pointer operands, gm_addr_args_indices should be empty."""
    ctx = _make_context()
    module = _parse(ctx, GET_RANK_IR)
    _run_pass(module, ctx)
    out = module.str()

    indices = _extract_gm_addr_indices(out)
    assert indices, "No hivm.custom operators found after conversion"
    by_symbol = {sym: idx for sym, idx in indices}

    # get_rank/get_num_ranks operands are cleared, no pointer parameters -> []
    assert ("aclshmem_my_pe" in by_symbol), f"get_rank's corresponding custom operator not found: {by_symbol}"
    assert by_symbol["aclshmem_my_pe"] == [], (f"get_rank's gm_addr_args_indices should be [], actual: "
                                               f"{by_symbol['aclshmem_my_pe']}")
    assert ("aclshmem_n_pes" in by_symbol), f"get_num_ranks' corresponding custom operator not found: {by_symbol}"
    assert by_symbol["aclshmem_n_pes"] == [], (f"get_num_ranks' gm_addr_args_indices should be [], actual: "
                                               f"{by_symbol['aclshmem_n_pes']}")


# When consume_token's input is a tensor pointer (tensor<...x!tt.ptr<T>>),
# its type is RankedTensorType rather than triton::PointerType, and should not be captured.
# wait's barrierPtr is still a scalar pointer parameter %arg0, should be captured -> [0].
# consume_token operands (%ptr_tensor: tensor pointer, %tok: i32) are both not scalar pointer parameters -> [].
TENSOR_PTR_IR = r"""
module {
  tt.func public @kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                         %arg1: i32 {tt.divisibility = 16 : i32},
                         %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %ptr_tensor = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
    %tok = distributed.wait %arg0, %arg1, %c1_i32, scope = gpu semantic = acquire : !tt.ptr<i32>, i32, i32 -> i32
    %ct = distributed.consume_token %ptr_tensor, %tok, : tensor<1x!tt.ptr<f32>>, i32 -> tensor<1x!tt.ptr<f32>>
    tt.store %ct, %cst : tensor<1x!tt.ptr<f32>>
    tt.return
  }
}
"""


def test_gm_addr_args_indices_tensor_ptr_not_captured():
    """Tensor pointers (tensor<...x!tt.ptr<T>>) should not be captured by gm_addr_args_indices."""
    ctx = _make_context()
    module = _parse(ctx, TENSOR_PTR_IR)
    _run_pass(module, ctx)
    out = module.str()

    indices = _extract_gm_addr_indices(out)
    assert indices, "No hivm.custom operators found after conversion"
    # This IR only has one wait and one consume_token, extract them in order of appearance
    wait_sym, wait_idx = indices[0]
    consume_sym, consume_idx = indices[1]
    assert (wait_sym == "aclshmem_wait_int32"), f"First custom should be wait, actual: {wait_sym}"
    assert (consume_sym == "aclshmem_consume_token_float_ptr_1d"
            ), f"Second custom should be consume_token, actual: {consume_sym}"
    # wait: barrierPtr=%arg0(scalar pointer parameter) -> [0]
    assert wait_idx == [0], f"wait's gm_addr_args_indices should be [0], actual: {wait_idx}"
    # consume_token: input=%ptr_tensor(tensor pointer, not PointerType) -> []
    assert consume_idx == [], (
        f"consume_token's gm_addr_args_indices should be [] (tensor pointer should not be captured), "
        f"actual: {consume_idx}")
