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
"""bf16 grid-constant TMA descriptors end to end (semantic key
ir.tma.grid_constant_descriptor.grid_const_param).

The manifest-side descriptor ABI (integrations/vibe_cuda_eval.py
``tma_descriptors`` validation + host ``cuTensorMapEncodeTiled`` emission)
is only half of the contract: the typed IR entry kernel must ask for the
descriptor as ``ll.const[ll.grid_constant[ll.TmaDescriptor]]`` so codegen
emits a ``__grid_constant__ CUtensorMap`` parameter, and the generated
launcher must forward that same CUtensorMap *by value*. Before this path
existed, listing a descriptor name in ``kernel_args`` lowered to a plain
``void*`` argument, and the generated translation unit failed with
"cannot convert void* to CUtensorMap" against the host-encoded descriptor.

These tests pin the bf16 rank-2 tile-copy path on CPU (no nvcc/GPU):
  * bf16 descriptors validate and encode as
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 with element_size 2;
  * the compiled entry signature carries
    ``const __grid_constant__ TmaDescriptor dA`` with the
    ``using TmaDescriptor = CUtensorMap;`` alias;
  * the .cu launcher declares/forwards ``CUtensorMap dA`` by value (never
    a void* transport or a reinterpret_cast on the descriptor name);
  * the binding host-encodes dA from the manifest extents/box/swizzle and
    passes it by value into the launcher;
  * the TMA 2D load pairs mbarrier::complete_tx::bytes with the
    mbarrier arrive/expect_tx/wait intrinsics and coexists with the
    warpgroup setmaxnreg inc/dec producer/consumer register stages.
"""

import json
from pathlib import Path

from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _TMA_DTYPES, _load_kernel, _load_spec, _render_binding, _render_cuda

DRAFT_SOURCE = """
import little_kernel as lk
import little_kernel.language as ll

BM = 8
BN = 64


@lk.ll_kernel(backend='cuda', is_entry=True)
def tma_bf16_tile_copy(dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
                       out: ll.Tensor[ll.bfloat16]) -> ll.void:
    tile = ll.alloc_shared_memory("s_tile", ll.bfloat16, BM * BN)
    bar = ll.alloc_shared_memory("s_bar", ll.int64, 1)
    ll.init_smem_barrier(bar, 1)
    # Producer stage hands registers to the TMA/copy path before issuing.
    ll.warpgroup_reg_dealloc(24)
    ll.mbarrier_arrive_and_expect_tx(bar, BM * BN * 2)
    ll.tma_load_2d_1sm(dA, bar, tile, 0, 0)
    ll.mbarrier_wait(bar, 0)
    # Consumer stage reclaims the warpgroup register budget afterwards.
    ll.warpgroup_reg_alloc(40)
    ll.__syncthreads()
"""


def _spec(oob_fill=True):
    return {
        "function":
        "lk_forward",
        "kernel":
        "tma_bf16_tile_copy",
        "inputs": [{"name": "A", "dtype": "bfloat16", "rank": 2}],
        "outputs": [{"name": "out", "dtype": "bfloat16", "shape": ["A[0]", "A[1]"]}],
        "scalars": [],
        "launch": {"grid": [1, 1, 1], "block": [128, 1, 1]},
        "tma_descriptors": [
            {
                "name": "dA", "tensor": "A", "rank": 2, "inner": "A[1]", "outer": "A[0]", "outer_stride": "A[1]", "box":
                [64, 8], "swizzle_bytes": 0, "oob_fill": oob_fill
            },
        ],
        "kernel_args": ["dA", "out"],
    }


def _lower(tmp_path: Path, *, oob_fill=True):
    (tmp_path / "draft.py").write_text(DRAFT_SOURCE)
    (tmp_path / "export.json").write_text(json.dumps(_spec(oob_fill)))
    spec = _load_spec(tmp_path / "export.json")
    kernel = _load_kernel(tmp_path / "draft.py", spec["_kernel"])
    return spec, _render_cuda(spec, kernel), _render_binding(spec)


def test_bf16_tma_dtype_table_entry():
    dtype, element_size = _TMA_DTYPES["bfloat16"]
    assert dtype == "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16"
    assert element_size == 2


def test_bf16_descriptor_validates(tmp_path):
    spec, _, _ = _lower(tmp_path)
    (item, ) = spec["tma_descriptors"]
    assert item["_dtype"] == "bfloat16"
    assert item["_box"] == [64, 8]
    assert item["_rank"] == 2
    assert "dA" in spec["_descriptor_names"]


def test_grid_constant_entry_signature_and_by_value_forwarding(tmp_path):
    _, cuda, _ = _lower(tmp_path)
    # The IR grid_constant[TmaDescriptor] annotation lowers to a
    # __grid_constant__ CUtensorMap entry parameter (alias emitted once).
    assert "using TmaDescriptor = CUtensorMap;" in cuda
    assert ("extern \"C\" __global__ void agent_tma_bf16_tile_copy(\n"
            "    const __grid_constant__ TmaDescriptor dA,") in cuda
    # The launcher transports the descriptor by value, exactly as the
    # host binding supplies it: no void* parameter and no cast on dA.
    assert "void* dA" not in cuda
    assert ">(dA)" not in cuda
    launcher_decl = ("extern \"C\" cudaError_t agent_lk_launch_lk_forward(\n"
                     "    CUtensorMap dA,")
    assert launcher_decl in cuda
    # The kernel launch forwards the bare by-value descriptor.
    assert ">>>(\n        dA,\n" in cuda


def test_tma_load_pairs_mbarrier_completion_and_setmaxnreg(tmp_path):
    _, cuda, _ = _lower(tmp_path)
    # cp.async.bulk.tensor.2d ... mbarrier::complete_tx::bytes is the TMA
    # completion transaction matched by arrive-and-expect-tx + wait.
    assert "cp.async.bulk.tensor.2d.shared::cluster.global." \
           "mbarrier::complete_tx::bytes" in cuda
    assert "mbarrier.arrive.expect_tx" in cuda
    assert "mbarrier.try_wait.parity" in cuda
    # Producer/consumer register stages coexist in the same translation
    # unit (warpgroup setmaxnreg dec before issue, inc after wait).
    assert "setmaxnreg.dec.sync.aligned.u32 24;" in cuda
    assert "setmaxnreg.inc.sync.aligned.u32 40;" in cuda


def test_binding_host_encodes_bf16_descriptor_and_passes_by_value(tmp_path):
    _, _, binding = _lower(tmp_path)
    assert ("agent_lk_make_tma_2d(&dA, A.data_ptr(), A.size(1), A.size(0), "
            "A.size(1), 64, 8, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, "
            "CU_TENSOR_MAP_SWIZZLE_NONE, "
            "CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA)") in binding
    assert "    CUtensorMap dA;" in binding
    assert "agent_lk_launch_lk_forward(dA, out.data_ptr()" in binding


def test_binding_and_launcher_descriptor_abi_agree(tmp_path):
    # The exact reported regression: launcher parameter type and binding
    # argument type must both be CUtensorMap (never void*).
    _, cuda, binding = _lower(tmp_path)
    assert "agent_lk_launch_lk_forward(\n    CUtensorMap dA," in cuda
    assert "agent_lk_launch_lk_forward(CUtensorMap dA," in binding


def test_bf16_descriptor_store_spelling_uses_no_oob_fill(tmp_path):
    # TMA stores require OOB fill none; the same host-encode path must
    # render that spelling for bf16 descriptors.
    spec = _spec(oob_fill=False)
    (tmp_path / "draft.py").write_text(DRAFT_SOURCE)
    (tmp_path / "export.json").write_text(json.dumps(spec))
    loaded = _load_spec(tmp_path / "export.json")
    binding = _render_binding(loaded)
    assert ("CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, CU_TENSOR_MAP_SWIZZLE_NONE, "
            "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)") in binding
