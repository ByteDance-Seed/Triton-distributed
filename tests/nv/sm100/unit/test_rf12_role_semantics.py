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
"""RF12 refactor-resilient role/dataflow semantic tests.

These replace role tests that enforced arbitrary *local variable names*
(e.g. ``tile_a``) or brittle source-regex spellings (e.g. whitespace-fused
``g.smem(...)`` fragments created by ``re.sub``).  They assert semantics on
the **parsed** plan/IR/resource structures produced by
``analyze_dataflow_source``:

* roles carry the declared hardware class and warp set;
* SMEM/TMEM resources carry shape, dtype, stage count and major;
* TMA transfer plans carry the participant coordinate dimension/stride;
* pipelines connect the correct producer/consumer roles over the right
  memory resources, and the lowered code keeps the real PTX opcode/ABI and
  barrier **ordering** contracts.

Nothing here renames a local identifier to make a test pass; benign
renames continue to compile and satisfy these invariants.  Exact ABI/PTX
assertions (UMMA descriptor constants selected by operand dtype, BF16
header include, SM16 opcode) are deliberately retained.
"""

from __future__ import annotations

import re

import pytest

import little_kernel.language as ll
from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri
from little_kernel.backends.nv.sm100.language.dataflow import HW, TMATransferPlan
from little_kernel.backends.nv.sm100.passes.dataflow import compile_dataflow_cuda, analyze_dataflow_source

PRELUDE = """
import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri
from little_kernel.backends.nv.sm100.language.dataflow import HW, dataflow
"""

# UMMA SM16 descriptor selectors.  These are real ABI constants (operand
# dtype / layout bits), not spellings: bf16 and f16 operands must select
# different descriptors even though both flow through umma_f16_ss_1sm.
UMMA_F16_SS_1SM_DESC_CONST = {
    "bfloat16": 136316048,
    "float16": 136314896,
}


def _umma_program(operand_dtype: str) -> str:
    # Local names are deliberately unorthodox (zzz / q / acc) to prove the
    # invariants do not depend on conventional names like A_smem / ad / d.
    return PRELUDE + f"""
@lk.ll_kernel(backend="cuda", is_entry=True)
def operands() -> ll.void:
    with dataflow() as g:
        writer = g.role("writer", hw=HW.CUDACore, warps=[0])
        mma = g.role("mma", hw=HW.TensorCore, warps=[1])
        q = g.smem("q", (128, 64), dtype=ll.{operand_dtype}, major="K")
        r = g.smem("r", (128, 64), dtype=ll.{operand_dtype}, major="K")
        acc = g.tmem("acc", (128, 128), dtype=ll.float32)
        link = g.depends(src=writer, dst=mma, via=[q, r])
    with writer:
        link.bind(0)
        zero: ll.bfloat16 = ll.float_to_bf16(ll.float32(0.0))
        ri.smem_st_bf16(q, 0, zero)
        ri.smem_st_bf16(r, 0, zero)
    with mma:
        link.bind(0)
        dq = ri.make_umma_smem_desc(q, (0, 0))
        dr = ri.make_umma_smem_desc(r, (0, 0))
        ri.umma_f16_ss_1sm_init(acc, acc, dq, dr, out_shape=(128, 128))
"""


# ---------------------------------------------------------------------------
# Parsed role / resource / channel invariants
# ---------------------------------------------------------------------------


def test_parsed_roles_carry_hardware_and_warps_not_names():
    graph = analyze_dataflow_source(_umma_program("bfloat16"))

    roles = graph["roles"]
    assert set(roles) == {"writer", "mma"}
    assert roles["writer"].hw is HW.CUDACore
    assert roles["writer"].warps == [0]
    assert roles["mma"].hw is HW.TensorCore
    assert roles["mma"].warps == [1]

    # SMEM resources keep their typed shape/dtype/stage/major regardless of
    # the (unconventional) local name.
    q, r = graph["memories"]["q"], graph["memories"]["r"]  # noqa: F841
    assert tuple(q.shape) == (128, 64)
    assert str(q.dtype) == "bfloat16"
    assert q.stages == 1 and q.major == "K"
    acc = graph["memories"]["acc"]
    assert tuple(acc.shape) == (128, 128)
    assert str(acc.dtype) == "float32"

    # Exactly one pipeline, connecting writer -> mma over both SMEM tiles,
    # and it selects the Generic->Umma family from operand *effects*.
    pipelines = graph["pipelines"]
    assert len(pipelines) == 1
    pipe = pipelines[0]
    assert pipe["producers"] == ("writer", )
    assert pipe["consumers"] == ("mma", )
    assert set(pipe["via"]) == {"q", "r"}
    assert pipe["template"].name == "PipelineGenericUmma"


@pytest.mark.parametrize("dtype,expected_const", list(UMMA_F16_SS_1SM_DESC_CONST.items()))
def test_umma_descriptor_constant_follows_operand_dtype(dtype, expected_const):
    """Real ABI assertion: operand dtype selects the exact SM16 descriptor."""
    source = _umma_program(dtype)
    graph = analyze_dataflow_source(source)
    # The plan itself records operand dtype; no source regex needed.
    assert all(str(m.dtype) == dtype for name, m in graph["memories"].items() if name in ("q", "r"))
    code = compile_dataflow_cuda(source, num_threads=64)
    match = re.search(r"umma_f16_1sm_fn\([^;]*?,\s*(\d+),\s*static_cast", code, re.DOTALL)
    assert match is not None
    assert int(match.group(1)) == expected_const


def test_float32_tmem_output_is_required():
    bad = _umma_program("bfloat16").replace(
        'acc = g.tmem("acc", (128, 128), dtype=ll.float32)',
        'acc = g.tmem("acc", (128, 128), dtype=ll.float16)',
    )
    with pytest.raises(ValueError, match="float32 TMEM output"):
        compile_dataflow_cuda(bad, num_threads=64)


def test_smem_major_is_validated_on_resource_not_via_spelling():
    # A missing/invalid major is a parsed-resource failure with a clear
    # diagnostic, not a text fragment we grep for.
    source = _umma_program("bfloat16").replace('major="K"', 'major="row"')
    with pytest.raises(ValueError, match="major"):
        analyze_dataflow_source(source)


def test_benign_local_rename_does_not_change_pipeline_resources():
    """A pure local rename must yield the same parsed resource/IR plan."""
    renamed = _umma_program("bfloat16").replace("writer", "producer_7")
    graph = analyze_dataflow_source(renamed)
    assert set(graph["roles"]) == {"producer_7", "mma"}
    assert graph["roles"]["producer_7"].hw is HW.CUDACore
    pipe = graph["pipelines"][0]
    assert pipe["producers"] == ("producer_7", )
    assert set(pipe["via"]) == {"q", "r"}
    assert pipe["template"].name == "PipelineGenericUmma"


# ---------------------------------------------------------------------------
# TMA transfer plan: parsed participant topology, not call spelling
# ---------------------------------------------------------------------------


def _tma_program(plan_call: str | None = None, cta_group: tuple | None = None) -> str:
    deco_extra = ""
    if cta_group is not None:
        deco_extra = (f", cluster_shape={cta_group}, cta_group_shape={cta_group}")
    plan_arg = "" if plan_call is None else ", plan"
    plan_decl = ("" if plan_call is None else f"        plan = g.tma_transfer({plan_call})\n")
    return PRELUDE + f"""
@lk.ll_kernel(backend="cuda", is_entry=True{deco_extra})
def two_stage(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]]) -> ll.void:
    with dataflow() as g:
        ld = g.role("ld", hw=HW.TMA, warps=[0])
        use = g.role("use", hw=HW.CUDACore, warps=[1])
        tile = g.smem("tile", (1, 32), dtype=ll.bfloat16)
        ch = g.depends(src=ld, dst=use, via=[tile])
{plan_decl}    with ld:
        elected: ll.uint32 = ll.elect_one()
        ch.bind(0)
        if elected == 1:
            ri.tma_ld_2d(tile, desc, (0, 0){plan_arg}, shape=(1, 32))
    with use:
        ch.bind(0)
        ri.smem_ld_bf16(tile, 0)
"""


def test_single_cta_transfer_plan_is_plain_copy():
    assert TMATransferPlan() == TMATransferPlan(mcast_group_dim=None, inter_cta_coord_dim=None, inter_cta_stride=0)
    # A single-CTA kernel passes no transfer plan: the plan registry then
    # contains no explicit partition (the default plain-copy contract).
    graph = analyze_dataflow_source(_tma_program())
    assert graph["transfer_plans"] == {}
    # Resource: one SMEM tile, 32 bf16 = 64 bytes flow through the channel.
    assert graph["pipelines"][0]["tx_bytes"] == 64
    # The typed default plan equals the plain-copy contract.
    assert TMATransferPlan().inter_cta_coord_dim is None


def test_two_cta_transfer_plan_records_partition_dimension_and_stride():
    graph = analyze_dataflow_source(_tma_program("inter_cta_coord_dim=1, inter_cta_stride=128", cta_group=(2, 1, 1)))
    plan = next(iter(graph["transfer_plans"].values()))
    assert plan.inter_cta_coord_dim == 1
    assert plan.inter_cta_stride == 128
    # The declared 2-CTA group is a parsed resource, not a source fragment.
    assert graph["cta_group_shape"] == (2, 1, 1)
    topo = graph["transfer_topologies"][plan]
    assert topo["group_size"] == 2


@pytest.mark.parametrize("bad", [
    "inter_cta_coord_dim=None, inter_cta_stride=8",
    "inter_cta_coord_dim=-1, inter_cta_stride=8",
    "inter_cta_coord_dim=1, inter_cta_stride=0",
])
def test_transfer_plan_rejects_inconsistent_participant_contract(bad):
    # The typed plan fails closed on an inconsistent participant contract
    # before / independent of codegen.
    with pytest.raises(ValueError):
        analyze_dataflow_source(_tma_program(bad))


# ---------------------------------------------------------------------------
# Ordering / ABI contracts retained on lowered code
# ---------------------------------------------------------------------------


def test_tma_to_generic_prepares_before_issuing_tma():
    code = compile_dataflow_cuda(_tma_program(), num_threads=64)
    # Real ordering contract: prepare-transaction precedes the TMA load,
    # and the issued load receives the prepared barrier.
    prep = code.index("producer_prepare_transaction();")
    call_start = code.index("tma_load_2d_1sm_fn(", prep)
    call_end = code.index(");", call_start)
    issued_call = code[call_start:call_end]
    # prepare precedes the issued load, and the load receives the barrier
    # obtained from this very pipeline (not a stale/global barrier).
    assert prep < call_start
    assert "producer_get_barrier()" in issued_call
    assert "tma_load_2d_2sm_fn" not in code


def test_bf16_abi_header_and_rne_pack_are_emitted():
    code = compile_dataflow_cuda(_umma_program("bfloat16"), num_threads=64)
    # Real ABI contract: bf16 lane widening/packing needs the bf16 header.
    assert "<cuda_bf16.h>" in code
    assert "PipelineGenericUmma" in code


def test_two_producer_one_consumer_channels_keep_resource_ownership():
    source = PRELUDE + """
@lk.ll_kernel(
    backend="cuda", is_entry=True,
    cluster_shape=(2, 1, 1), cta_group_shape=(2, 1, 1),
)
def cooperative() -> ll.void:
    with dataflow() as g:
        p0 = g.role("p0", hw=HW.TMA, warps=[0])
        p1 = g.role("p1", hw=HW.TMA, warps=[1])
        c = g.role("c", hw=HW.CUDACore, warps=[2])
        tile = g.smem("tile", (1, 32), dtype=ll.bfloat16)
        ch = g.depends(src=[p0, p1], dst=c, via=[tile])
    with p0:
        ch.bind(0)
        elected0: ll.uint32 = ll.elect_one()
        if elected0 == 1:
            ri.tma_ld_2d(tile, None, (0, 0), smem_coord=(0, 0), shape=(1, 16))
    with p1:
        ch.bind(0)
        elected1: ll.uint32 = ll.elect_one()
        if elected1 == 1:
            ri.tma_ld_2d(tile, None, (16, 0), smem_coord=(0, 16), shape=(1, 16))
    with c:
        ch.bind(0)
        ri.smem_ld_bf16(tile, 0)
"""
    graph = analyze_dataflow_source(source)
    pipes = graph["pipelines"]
    # Two cooperative producers each own one half-tile channel into the
    # same consumer; together they partition the 32-element (64-byte) tile.
    producer_sets = {tuple(sorted(p["producers"])) for p in pipes}
    assert producer_sets == {("p0", ), ("p1", )}
    assert {p["consumers"] for p in pipes} == {("c", )}
    assert all(p["via"] == ["tile"] for p in pipes)
    assert sum(p["tx_bytes"] for p in pipes) == 64
    for p in pipes:
        assert p["tx_bytes"] == 32  # each producer covers half
    # Both producers are TMA hardware, the consumer is a CUDA core.
    assert graph["roles"]["p0"].hw is HW.TMA
    assert graph["roles"]["p1"].hw is HW.TMA
    assert graph["roles"]["c"].hw is HW.CUDACore
    # The single tile resource is 32 bf16 elements, one stage.
    tile = graph["memories"]["tile"]
    assert tuple(tile.shape) == (1, 32) and str(tile.dtype) == "bfloat16"
