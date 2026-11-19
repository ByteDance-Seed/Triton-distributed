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

from __future__ import annotations

import ast
import importlib.util
import operator
import re
from pathlib import Path

import pytest

# Several role-proposal demo kernels (v2_groupgemm / fused 2-CTA variants)
# import torch at module load only to build host-side launch helpers.  The
# compiler-only invariants below do not execute torch, but ``exec``-ing those
# demo sources still requires the module to import.  Gate exactly those cases
# on capability instead of letting collection report a confusing hard failure;
# root acceptance runs them on the CUDA/torch image (worker18 also trims the
# incidental demo imports).  Tests over torch-free demos remain CPU-runnable.
_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_requires_torch_demos = pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason="demo kernel module imports torch; run on the CUDA/torch acceptance image",
)


# Demo files whose top-level source (or a direct role_proposal sibling
# import) needs torch.  Derived by scanning source after TWO_CTA_KERNEL_FILES
# is declared (see below), not hard-coded, so a demo that drops its torch
# import becomes CPU-runnable automatically.
def _source_imports_torch(filename: str, _seen: set[str] | None = None) -> bool:
    # Imports inside host-only functions do not affect source loading.
    from tests.nv.sm100.dataflow._source_paths import read_role_demo
    for node in ast.parse(read_role_demo(filename)).body:
        if isinstance(node, ast.Import) and any(item.name == "torch" or item.name.startswith("torch.")
                                                for item in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module == "torch" or (node.module or "").startswith("torch.")):
            return True
    return False


from little_kernel.core.memory_effects import R_SMEM_GENERIC_SYNC, R_SMEM_TCGEN05_COMMIT, R_SMEM_TMA_BULK, R_TMEM_TCGEN05_COMMIT, R_TMEM_TCGEN05_WAIT_LD, RW_TMEM_TCGEN05_COMMIT, W_SMEM_GENERIC_SYNC, W_SMEM_TMA_TX, W_TMEM_TCGEN05_COMMIT, W_TMEM_TCGEN05_WAIT_ST
from little_kernel.language.builtin_base import CTA_GROUP_ATTR, MEMORY_EFFECTS_ATTR
from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri
from little_kernel.backends.nv.sm100.language.dataflow import HW, Flow, TMATransferPlan
from little_kernel.backends.nv.sm100.codegen.dataflow.lowering import RoleResourceMaterializer, _tmem_stage_cols
from little_kernel.backends.nv.sm100.codegen.dataflow.layout import _layout_fields
from little_kernel.backends.nv.sm100.passes.dataflow.graph_construction import _plan_smem_layout
from little_kernel.backends.nv.sm100.passes.dataflow import compile_dataflow_cuda, analyze_dataflow_source
from little_kernel.backends.nv.sm100.passes.dataflow.buffer_overlay import MemoryMappingEntry, analyze_buffer_overlays
from little_kernel.backends.nv.sm100.passes.dataflow.channels import barrier_participation
from little_kernel.backends.nv.sm100.core.pipeline import PIPELINE_BY_EFFECT_KINDS

from tests.nv.sm100.dataflow._source_paths import read_role_demo, skip_if_demo_requires_torch

PRELUDE = """
import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri
from little_kernel.backends.nv.sm100.language.dataflow import HW, dataflow
"""

TWO_CTA_KERNEL_FILES = (
    "v2_gemm_sm100.py",
    "v2_groupgemm.py",
    "v2_ag_gemm_sm100.py",
    "v2_gemm_rs_sm100.py",
    "v2_gemm_a2a_sm100.py",
    "v2_a2a_gemm_sm100.py",
    "v2_a2a_gemm_fuse_sm100.py",
    "v2_ag_gemm_fuse_sm100.py",
    "v2_gemm_rs_fuse_sm100.py",
    "v2_gemm_a2a_fuse_sm100.py",
    "v2_dispatch_group_gemm_swiglu_fuse_sm100.py",
)

PAIR_TILE_M_BY_FILE = {
    "v2_gemm_sm100.py": 256,
    "v2_groupgemm.py": 256,
    "v2_ag_gemm_sm100.py": 256,
    "v2_gemm_rs_sm100.py": 256,
    "v2_gemm_a2a_sm100.py": 256,
    "v2_a2a_gemm_sm100.py": 512,
    "v2_a2a_gemm_fuse_sm100.py": 512,
    "v2_ag_gemm_fuse_sm100.py": 512,
    "v2_gemm_rs_fuse_sm100.py": 512,
    "v2_gemm_a2a_fuse_sm100.py": 512,
    "v2_dispatch_group_gemm_swiglu_fuse_sm100.py": 256,
}

TWO_CTA_KERNEL_THREADS = {
    filename: 544 if filename == "v2_gemm_rs_fuse_sm100.py" else 256
    for filename in TWO_CTA_KERNEL_FILES
}

# Populated after the file list is known: the set of demos that currently
# import torch (directly or through a role_proposal sibling) at module load.
_TORCH_IMPORTING_DEMO_FILES = frozenset({name for name in TWO_CTA_KERNEL_FILES if _source_imports_torch(name)})

REQUIRED_GEOMETRY_NAMES = frozenset({
    "PAIR_TILE_M",
    "PAIR_TILE_N",
    "CTA_TILE_M",
    "CTA_TILE_N_SHARD",
    "K_TILE",
    "MMA_WAVE_M",
    "NUM_MMA_M_WAVES",
    "ST_TILE_M",
    "ST_TILE_N",
})

LEGACY_GEOMETRY_NAMES = (
    "BM",
    "BN",
    "BK",
    "CTA_M",
    "LOAD_N_PER_CTA",
    "STORE_BM",
    "STORE_BN",
    "CTA_N_SHARD",
    "TILE_K",
    "GEMM_TILE_K",
    "NUM_MMA_WAVES",
    "EPILOGUE_TILE_M",
    "EPILOGUE_TILE_N",
    "STORE_TILE_M",
    "STORE_TILE_N",
    "COMM_K",
    "RS_TILE_M",
    "RS_TILE_N",
    "DISPATCH_K",
)
LEGACY_GEOMETRY_PATTERN = re.compile(rf"(?<![A-Za-z0-9_])(?:{'|'.join(LEGACY_GEOMETRY_NAMES)})(?![A-Za-z0-9_])")

COMM_FRAGMENT_OWNERSHIP = {
    "v2_a2a_gemm_fuse_sm100.py": (
        ("COMM_TILE_M", "COMM_TILE_K"),
        ('comm_smem=g.smem("comm_smem",(COMM_TILE_M,COMM_TILE_K)', ),
    ),
    "v2_ag_gemm_fuse_sm100.py": (
        ("COMM_TILE_M", "COMM_TILE_K"),
        ('comm_smem=g.smem("comm_smem",(COMM_TILE_M,COMM_TILE_K)', ),
    ),
    "v2_gemm_rs_fuse_sm100.py": (
        ("COMM_TILE_M", "COMM_TILE_N"),
        (
            'rs_peer_smem=g.smem("rs_peer_smem",(COMM_TILE_M,COMM_TILE_N)',
            'rs_out_smem=g.smem("rs_out_smem",(COMM_TILE_M,COMM_TILE_N)',
        ),
    ),
    "v2_dispatch_group_gemm_swiglu_fuse_sm100.py": (
        ("COMM_TILE_K", ),
        ('dispatch_smem=g.smem("dispatch_smem",(COMM_TILE_K,)', ),
    ),
}

CONST_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
}


def _kernel_source(filename: str) -> str:
    # The authored role DSL kernel is located through the canonical example
    # mapping (examples/nv/sm100 preferred, in-package role_proposal until the
    # example move), never by importing a compatibility facade module.
    return read_role_demo(filename)


def _pipeline_header() -> Path:
    import little_kernel.backends.nv.sm100.runtime
    return Path(little_kernel.backends.nv.sm100.runtime.__file__).parent / "include" / "role_pipeline.cuh"


def test_runtime_pipeline_state_uses_composition() -> None:
    header = _pipeline_header().read_text(encoding="utf-8")
    assert ": PipelineState<Stages>" not in header
    assert header.count("PipelineState<Stages> state_;") == 10
    for name in (
            "PipelineUmmaUmma",
            "PipelineUmmaTmemLoad",
            "PipelineTmemStoreUmma",
            "PipelineTmemStoreTmemLoad",
    ):
        assert f"struct {name}" in header
    assert "PipelineUmmaGeneric" not in header
    body = lambda name: header[header.index(f"struct {name}"):header.index("\n};", header.index(f"struct {name}"))]
    assert "async_proxy_fence();" in body("PipelineGenericUmma")
    assert "tmem_store_wait();" not in body("PipelineGenericUmma")
    assert "tmem_store_wait();" in body("PipelineTmemStoreUmma")
    assert body("PipelineUmmaUmma").count("umma_commit<CtaGroup>") == 2
    assert body("PipelineUmmaTmemLoad").count("umma_commit<CtaGroup>") == 1
    assert "tmem_store_wait();" in body("PipelineTmemStoreTmemLoad")
    assert "tcgen05_before_thread_sync();" in body("PipelineTmemStoreTmemLoad")
    assert header.count("void producer_prepare_transaction()") == 4
    assert header.count("if (!producer_prepared_)") == 4
    getters = re.findall(r"uint64_t\s*\*\s*producer_get_barrier\(\)\s*const\s*\{([^{}]+)\}", header)
    assert len(getters) == 4
    assert all("return producer_[" in body and "arrive" not in body for body in getters)
    assert "producer_barrier()" not in header
    assert header.count("void producer_tail()") == 1
    assert "void tma_store_wait()" not in header


def test_pipeline_semantics_live_on_operand_effects() -> None:
    effects = {item.operand: item.kind for item in getattr(ri.umma_f16_ss_2sm, MEMORY_EFFECTS_ATTR)}
    assert effects == {
        "d_mem": RW_TMEM_TCGEN05_COMMIT,
        "desc_a.mem": R_SMEM_TCGEN05_COMMIT,
        "desc_b.mem": R_SMEM_TCGEN05_COMMIT,
    }
    assert getattr(ri.umma_f16_ss_2sm, CTA_GROUP_ATTR) == 2
    assert getattr(ri.tmem_ld_32dp_32bxn, MEMORY_EFFECTS_ATTR)[0].kind == R_TMEM_TCGEN05_WAIT_LD
    assert not hasattr(ri.umma_f16_ss_2sm, "__lk_pipeline_op_kind__")
    kblock_effects = {item.operand: item.kind for item in getattr(ri.umma_f16_ss_2sm_kblock, MEMORY_EFFECTS_ATTR)}
    assert kblock_effects["d_mem"] == RW_TMEM_TCGEN05_COMMIT
    assert getattr(ri.umma_f16_ss_2sm_kblock, CTA_GROUP_ATTR) == 2


def test_umma_smem_major_is_explicit_and_fail_closed() -> None:
    assert Flow().smem("v", (128, 64), dtype=object(), major="MN").major == "MN"
    with pytest.raises(ValueError, match="SMEM major must be 'K' or 'MN'"):
        Flow().smem("a", (128, 64), dtype=object(), major="row")

    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def missing_major() -> ll.void:
    with dataflow() as g:
        produce = g.role("produce", hw=HW.CUDACore, warps=[0])
        consume = g.role("consume", hw=HW.TensorCore, warps=[1])
        a = g.smem("a", (128, 64), dtype=ll.bfloat16)
        b = g.smem("b", (128, 64), dtype=ll.bfloat16, major="K")
        d = g.tmem("d", (128, 128), dtype=ll.float32)
        stage = g.depends(src=produce, dst=consume, via=[a, b])
    with produce:
        stage.bind(0)
        zero: ll.bfloat16 = ll.float_to_bf16(ll.float32(0.0))
        ri.smem_st_bf16(a, 0, zero)
        ri.smem_st_bf16(b, 0, zero)
    with consume:
        stage.bind(0)
        ad = ri.make_umma_smem_desc(a, (0, 0))
        bd = ri.make_umma_smem_desc(b, (0, 0))
        ri.umma_f16_ss_1sm_init(d, d, ad, bd, out_shape=(128, 128))
"""
    with pytest.raises(ValueError, match=r'a: UMMA SMEM operand requires explicit .*major="K" or "MN"'):
        analyze_dataflow_source(source)


@pytest.mark.parametrize(
    ("extent", "swizzle"),
    ((16, 32), (32, 64), (64, 128), (96, 64), (128, 128)),
)
def test_canonical_smem_layout_selects_widest_exact_swizzle(extent: int, swizzle: int) -> None:
    plan = _plan_smem_layout(Flow().smem("a", (128, extent), dtype=ri.bfloat16, major="K"))
    assert plan.swizzle_bytes == swizzle
    atom, lbo, sbo, layout_type = _layout_fields(plan)
    assert (atom, lbo, sbo, layout_type) == (
        swizzle // 2,
        16,
        8 * swizzle,
        {32: 6, 64: 4, 128: 2}[swizzle],
    )


def test_tmem_allocator_packs_dtypes_and_skips_alias_storage() -> None:
    flow = Flow()
    a = flow.tmem("a", (128, 64), dtype=ri.float32)
    flow.tmem("alias", (128, 64), dtype=ri.float32)
    packed = flow.tmem("packed", (128, 128), dtype=ri.bfloat16)
    mapping = [MemoryMappingEntry("alias", "a", space="tmem")]
    graph = {
        "roles": {},
        "memories": flow.memories,
        "pipelines": [],
        "mem_events": [],
        "memory_mapping_plan": mapping,
    }
    graph["buffer_overlay"] = analyze_buffer_overlays(graph, mapping)
    materializer = RoleResourceMaterializer(graph)

    assert _tmem_stage_cols(a) == _tmem_stage_cols(packed) == 64
    assert _tmem_stage_cols(Flow().tmem("sf", (128, 8), dtype=ri.uint8), cta_group=2) == 4
    assert materializer.tmem_offsets == {"a": 0, "packed": 64, "alias": 0}
    assert materializer.tmem_total_cols == 128


def test_single_cta_tmem_finalization_uses_one_barrier() -> None:
    flow = Flow()
    flow.role("mma", hw=HW.TensorCore, warps=[0])
    flow.tmem("acc", (32, 32), dtype=ri.float32)
    mapping = [MemoryMappingEntry("acc", "acc", space="tmem")]
    graph = {
        "roles": flow.roles,
        "memories": flow.memories,
        "pipelines": [],
        "mem_events": [],
        "memory_mapping_plan": mapping,
    }
    graph["buffer_overlay"] = analyze_buffer_overlays(graph, mapping)

    finalization = RoleResourceMaterializer(graph)._finalize_resources()

    assert len(finalization) == 2
    assert finalization[0].spelling == "__syncthreads();"
    assert "tmem_dealloc<1>" in finalization[1].spelling


def test_umma_idesc_uses_operand_dtypes_and_rejects_non_f32_output() -> None:
    source = _kernel_source("v2_gemm_sm100.py")
    f16_source = source.replace(
        'A_smem = g.smem("A_smem", (CTA_TILE_M, K_TILE), dtype=ll.bfloat16',
        'A_smem = g.smem("A_smem", (CTA_TILE_M, K_TILE), dtype=ll.float16',
    ).replace(
        'B_smem = g.smem("B_smem", (CTA_TILE_N_SHARD, K_TILE), dtype=ll.bfloat16',
        'B_smem = g.smem("B_smem", (CTA_TILE_N_SHARD, K_TILE), dtype=ll.float16',
    )
    assert "272629776" in compile_dataflow_cuda(f16_source, num_threads=256)

    invalid_d = source.replace(
        'C_tmem = g.tmem("C_tmem", (PAIR_TILE_M, PAIR_TILE_N), dtype=ll.float32',
        'C_tmem = g.tmem("C_tmem", (PAIR_TILE_M, PAIR_TILE_N), dtype=ll.float16',
    )
    with pytest.raises(ValueError, match="requires a float32 TMEM output"):
        compile_dataflow_cuda(invalid_d, num_threads=256)


def test_core_pipeline_matrix_has_exactly_ten_effect_pairs() -> None:
    assert set(PIPELINE_BY_EFFECT_KINDS) == {
        (producer, consumer)
        for producer in (
            W_SMEM_GENERIC_SYNC,
            W_SMEM_TMA_TX,
            W_TMEM_TCGEN05_COMMIT,
            W_TMEM_TCGEN05_WAIT_ST,
        )
        for consumer in ((R_SMEM_GENERIC_SYNC, R_SMEM_TMA_BULK,
                          R_SMEM_TCGEN05_COMMIT) if producer.space.value == "smem" else (R_TMEM_TCGEN05_COMMIT,
                                                                                         R_TMEM_TCGEN05_WAIT_LD))
    }


def _const_value(node: ast.expr, constants: dict[str, object]) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return constants[node.id]
    if isinstance(node, ast.Tuple):
        return tuple(_const_value(element, constants) for element in node.elts)
    if isinstance(node, ast.BinOp) and type(node.op) in CONST_BINOPS:
        return CONST_BINOPS[type(node.op)](
            _const_value(node.left, constants),
            _const_value(node.right, constants),
        )
    raise ValueError("not a static module constant")


def _module_geometry(source: str) -> tuple[set[str], dict[str, object]]:
    assigned_names: set[str] = set()
    constants: dict[str, object] = {}
    for statement in ast.parse(source).body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        assigned_names.add(target.id)
        try:
            constants[target.id] = _const_value(statement.value, constants)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pass
    return assigned_names, constants


@pytest.mark.parametrize("filename", TWO_CTA_KERNEL_FILES)
def test_two_cta_kernels_use_canonical_geometry_vocabulary(filename: str) -> None:
    source = _kernel_source(filename)
    assigned_names, geometry = _module_geometry(source)
    assert REQUIRED_GEOMETRY_NAMES <= assigned_names
    assert LEGACY_GEOMETRY_PATTERN.search(source) is None
    assert "NUM_MWAVES" not in source
    assert "NUM_M_WAVES" not in source

    assert geometry["PAIR_TILE_M"] == PAIR_TILE_M_BY_FILE[filename]
    assert geometry["PAIR_TILE_N"] == 256
    assert geometry["K_TILE"] == 64
    assert geometry["CTA_TILE_M"] * 2 == geometry["PAIR_TILE_M"]
    assert geometry["CTA_TILE_N_SHARD"] * 2 == geometry["PAIR_TILE_N"]
    assert geometry["MMA_WAVE_M"] == 128
    assert geometry["NUM_MMA_M_WAVES"] * geometry["MMA_WAVE_M"] == geometry["CTA_TILE_M"]
    assert geometry["ST_TILE_M"] == geometry["MMA_WAVE_M"]
    assert geometry["ST_TILE_N"] == 64


@pytest.mark.parametrize("filename", TWO_CTA_KERNEL_FILES)
def test_two_cta_kernels_express_participant_ownership(filename: str) -> None:
    source = _kernel_source(filename)
    compact = re.sub(r"\s+", "", source)
    assert 'A_smem=g.smem("A_smem",(CTA_TILE_M,K_TILE)' in compact
    assert 'B_smem=g.smem("B_smem",(CTA_TILE_N_SHARD,K_TILE)' in compact
    assert 'C_smem=g.smem("C_smem",(ST_TILE_M,ST_TILE_N)' in compact

    for plan_name, stride in (("A", "CTA_TILE_M"), ("B", "CTA_TILE_N_SHARD")):
        plan_match = re.search(rf"{plan_name}_plan\s*=\s*g\.tma_transfer\((.*?)\n\s*\)", source, re.DOTALL)
        assert plan_match is not None
        plan = plan_match.group(1)
        assert "inter_cta_coord_dim=1" in plan
        assert f"inter_cta_stride={stride}" in plan

    for role in ("a", "b", "load", "tma", "mma", "epi"):
        assert f"tile_{role}: ll.uint32 = cta_id" not in source


@pytest.mark.parametrize("filename", tuple(COMM_FRAGMENT_OWNERSHIP))
def test_communication_fragments_use_canonical_geometry(filename: str) -> None:
    source = _kernel_source(filename)
    assigned_names, _ = _module_geometry(source)
    required_names, fragments = COMM_FRAGMENT_OWNERSHIP[filename]
    assert set(required_names) <= assigned_names
    compact = re.sub(r"\s+", "", source)
    for fragment in fragments:
        assert fragment in compact


@pytest.mark.parametrize("filename",
                         tuple(name for name in TWO_CTA_KERNEL_FILES if name in _TORCH_IMPORTING_DEMO_FILES))
@_requires_torch_demos
def test_torch_demo_codegen_smoke(filename: str) -> None:
    _assert_two_cta_kernel_codegen(filename)


@pytest.mark.parametrize("filename",
                         tuple(name for name in TWO_CTA_KERNEL_FILES if name not in _TORCH_IMPORTING_DEMO_FILES))
def test_two_cta_kernel_codegen_smoke(filename: str) -> None:
    _assert_two_cta_kernel_codegen(filename)


def _assert_two_cta_kernel_codegen(filename: str) -> None:
    skip_if_demo_requires_torch(filename)
    code = compile_dataflow_cuda(_kernel_source(filename), num_threads=TWO_CTA_KERNEL_THREADS[filename])
    assert 'extern "C"' in code
    if filename == "v2_dispatch_group_gemm_swiglu_fuse_sm100.py":
        pipe = "PIPE_dispatch_load_to_dispatch_store_dispatch_smem"
        assert "__device__ __forceinline__ float swiglu_f32(float gate, float up)" in code
        assert code.count("__device__ __forceinline__ float swiglu_f32(float gate, float up)") == 1
        assert "swiglu_f32(__uint_as_float(" in code
        assert code.count("__device__ __forceinline__ uint32_t align_up_u32(uint32_t value, uint32_t alignment)") == 1
        assert code.count("align_up_u32(") == 4
        assert "PipelineTmaTmaStore<1, 16384, 1, 1>" in code
        assert "PipelineTmaTmaStoreSameRole" not in code
        assert "TmaStoreStream __lk_tma_store_stream_dispatch_store" in code
        assert f"{pipe}.consumer_release();" in code
        assert f"{pipe}.retire_completed();" in code
    else:
        helper = "__device__ __forceinline__ void tmem_ld_32dp_f32_to_smem_bf16_x8"
        assert code.count(helper) == 1
        # Static unrolling replicates call sites, but never helper definitions.
        assert code.count("tmem_ld_32dp_f32_to_smem_bf16_x8(") > 1
        assert code.count("__device__ __forceinline__ void tmem_ld_32dp_32bxn_x8_fn") == 1


def test_tmem_composite_intrinsic_regions_are_metadata_driven() -> None:
    graph = analyze_dataflow_source(_kernel_source("v2_gemm_sm100.py"))
    events = [event for event in graph["mem_events"] if event.intrin == "tmem_ld_32dp_f32_to_smem_bf16_x8"]
    src_events = [event for event in events if event.mem == "C_tmem"]
    dst_events = [event for event in events if event.mem == "C_smem"]
    assert [event.logical_offset for event in src_events] == list(range(0, 256, 8))
    assert {event.logical_extent for event in src_events} == {8}
    assert {event.effect_kind for event in src_events} == {R_TMEM_TCGEN05_WAIT_LD}
    assert all(event.logical_offset is None and event.logical_extent is None for event in dst_events)


def test_fa4_uses_public_32dp_x32_tmem_epilogue() -> None:
    source = _kernel_source("v2_fa4.py")
    assert 'sV = g.smem("sV", (HEAD_DIM, BN), dtype=ll.bfloat16, stages=KV_STAGE, major="MN")' in source
    assert source.count("ri.tmem_ld_32dp_f32_to_smem_bf16_x32(") == 2
    assert "tmem_f32_to_smem_bf16_32x" not in source
    code = compile_dataflow_cuda(source, num_threads=512)
    helper = "__device__ __forceinline__ void tmem_ld_32dp_f32_to_smem_bf16_x32"
    assert code.count(helper) == 1
    # The ordinary pipeline may unroll call sites; helper deduplication and
    # the two authored epilogue sites are the stable contracts.
    assert code.count("tmem_ld_32dp_f32_to_smem_bf16_x32(") > 1
    assert source.count("ri.tmem_ld_32dp_f32_to_smem_bf16_x32(") == 2
    assert code.count("__device__ __forceinline__ void tmem_ld_32dp_32bxn_x32_fn") == 1
    assert "PipelineUmmaTmemLoad" in code
    assert "PipelineTmemStoreUmma" in code
    assert "PipelineUmmaUmma" in code
    assert "PipelineUmmaGeneric" not in code
    assert "tmem_st_fence_fn();" not in code
    assert "encode_umma_smem_desc_fn" in code
    assert "vd_hi_0 = encode_umma_smem_desc_fn" in code
    assert ", 16384, 1024, 2);" in code
    assert "umma_f16_ts_1sm_kblock_fn" not in code
    assert "umma_f16_ts_1sm_fn(__lk_tmem_out0, __lk_tmem_prob0_hi + 24" in code
    assert "136381584" in code


def test_default_tma_transfer_plan_is_single_cta_plain_copy() -> None:
    expected = TMATransferPlan(
        mcast_group_dim=None,
        inter_cta_coord_dim=None,
        inter_cta_stride=0,
    )
    assert TMATransferPlan() == expected
    assert Flow().tma_transfer() == expected


@pytest.mark.parametrize("plan_arg", ["", ", None"])
def test_tma_load_without_plan_lowers_to_cg1(plan_arg: str) -> None:
    source = PRELUDE + f"""
@lk.ll_kernel(backend="cuda", is_entry=True)
def default_tma(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]]) -> ll.void:
    with dataflow() as g:
        load = g.role("load", hw=HW.TMA, warps=[0])
        consume = g.role("consume", hw=HW.CUDACore, warps=[1])
        tile = g.smem("tile", (1, 32), dtype=ll.bfloat16)
        stage = g.depends(src=load, dst=consume, via=[tile])
    with load:
        issuer: ll.uint32 = ll.elect_one()
        stage.bind(0)
        if issuer == 1:
            ri.tma_ld_2d(tile, desc, (0, 0){plan_arg}, shape=(1, 32))
    with consume:
        stage.bind(0)
        ri.smem_ld_bf16(tile, 0)
"""
    code = compile_dataflow_cuda(source, num_threads=64)
    pipe = "PIPE_load_to_consume_tile"
    assert "tma_load_2d_1sm_fn(&desc" in code
    assert "tma_load_2d_2sm_fn" not in code
    assert code.count(f"{pipe}.producer_prepare_transaction();") == 1
    assert f"{pipe}.producer_get_barrier()" in code
    assert f"{pipe}.producer_barrier()" not in code


def test_multiple_tma_loads_prepare_one_runtime_transaction() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def split_tma(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]]) -> ll.void:
    with dataflow() as g:
        load = g.role("load", hw=HW.TMA, warps=[0])
        consume = g.role("consume", hw=HW.CUDACore, warps=[1])
        tile = g.smem("tile", (1, 64), dtype=ll.bfloat16)
        stage = g.depends(src=load, dst=consume, via=[tile])
    with load:
        issuer: ll.uint32 = ll.elect_one()
        stage.bind(0)
        if issuer == 1:
            ri.tma_ld_2d(tile, desc, (0, 0), smem_coord=(0, 0), shape=(1, 32))
            ri.tma_ld_2d(tile, desc, (32, 0), smem_coord=(0, 32), shape=(1, 32))
    with consume:
        stage.bind(0)
        ri.smem_ld_bf16(tile, 0)
"""
    code = compile_dataflow_cuda(source, num_threads=64)
    pipe = "PIPE_load_to_consume_tile"
    assert code.count(f"{pipe}.producer_prepare_transaction();") == 2
    assert code.count(f"{pipe}.producer_get_barrier()") == 2


def test_optional_tma_load_prepares_only_inside_issue_path() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def optional_tma(
    desc: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    valid: ll.int32,
) -> ll.void:
    with dataflow() as g:
        load = g.role("load", hw=HW.TMA, warps=[0])
        consume = g.role("consume", hw=HW.CUDACore, warps=[1])
        tile = g.smem("tile", (1, 32), dtype=ll.bfloat16)
        stage = g.depends(src=load, dst=consume, via=[tile])
    with load:
        issuer: ll.uint32 = ll.elect_one()
        stage.bind(0)
        if issuer == 1:
            if valid != 0:
                ri.tma_ld_2d(tile, desc, (0, 0), shape=(1, 32))
    with consume:
        stage.bind(0)
        ri.smem_ld_bf16(tile, 0)
"""
    code = compile_dataflow_cuda(source, num_threads=64)
    pipe = "PIPE_load_to_consume_tile"
    # Structural invariant (not brace spelling): the optional TMA transaction
    # is only prepared/issued inside BOTH the issuer election and the dynamic
    # `valid` predicate, while the producer commit is unconditional.  Assert
    # lexical nesting order rather than an exact regex of parenthesization
    # (constant folding may collapse redundant parentheses).
    issuer = code.index("if (issuer == 1)")
    valid = code.index("if (valid != 0)", issuer)
    prepare = code.index(f"{pipe}.producer_prepare_transaction();", valid)
    issue = code.index("tma_load_2d_1sm_fn(", prepare)
    issue_end = code.index(";", issue)
    assert "&desc" in code[issue:issue_end]
    assert f"{pipe}.producer_get_barrier()" in code[issue:issue_end]
    commit = code.index(f"{pipe}.producer_commit();", issue)
    assert issuer < valid < prepare < issue < commit
    assert code.count(f"{pipe}.producer_prepare_transaction();") == 1
    # Exactly one issue call site (the other occurrence is the helper
    # definition `void tma_load_2d_1sm_fn(...)`).
    assert code.count("tma_load_2d_1sm_fn(") == 2
    assert "void tma_load_2d_1sm_fn(" in code


def test_rejects_mismatched_transaction_streams() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def stream_mismatch(n: ll.int32):
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        c = g.role("c", hw=HW.CUDACore, warps=[1])
        tile = g.smem("tile", (32,), dtype=ll.float32)
        stage = g.depends(src=p, dst=c, via=[tile])
    with p:
        i: ll.int32 = 0
        while i < n:
            stage.bind(i)
            ri.smem_st_f32(tile, 0, ll.val_cast(0.0, ll.float32))
            i = i + 1
    with c:
        j: ll.int32 = 0
        while j < n + 1:
            stage.bind(j)
            ri.smem_ld_f32(tile, 0)
            j = j + 1
"""
    with pytest.raises(ValueError, match="transaction stream mismatch"):
        analyze_dataflow_source(source)


def test_accepts_equivalent_even_step_transaction_streams() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def even_step_stream(n: ll.int32):
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        c = g.role("c", hw=HW.CUDACore, warps=[1])
        tile = g.smem("tile", (32,), dtype=ll.float32, stages=2)
        stage = g.depends(src=p, dst=c, via=[tile])
    with p:
        i: ll.int32 = 0
        while i < n * 2:
            stage.bind(i)
            ri.smem_st_f32(tile, 0, ll.val_cast(0.0, ll.float32))
            i = i + 1
    with c:
        j: ll.int32 = 0
        while j < n * 2:
            stage.bind(j)
            ri.smem_ld_f32(tile, 0)
            stage.bind(j + 1)
            ri.smem_ld_f32(tile, 0)
            j = j + 2
"""
    analyze_dataflow_source(source)


def test_accepts_equivalent_pair_and_flat_transaction_streams() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def pair_stream(n: ll.int32):
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        c = g.role("c", hw=HW.CUDACore, warps=[1])
        tile = g.smem("tile", (32,), dtype=ll.float32, stages=2)
        stage = g.depends(src=p, dst=c, via=[tile])
    with p:
        pair: ll.int32 = 0
        while pair < n:
            stage.bind(pair * 2)
            ri.smem_st_f32(tile, 0, ll.float32(0.0))
            stage.bind(pair * 2 + 1)
            ri.smem_st_f32(tile, 0, ll.float32(0.0))
            pair = pair + 1
    with c:
        chunk: ll.int32 = 0
        while chunk < n * 2:
            stage.bind(chunk)
            ri.smem_ld_f32(tile, 0)
            chunk = chunk + 1
"""
    analyze_dataflow_source(source)


def test_accepts_equivalent_unrolled_and_flat_transaction_streams() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def unrolled_stream(n: ll.int32):
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        c = g.role("c", hw=HW.CUDACore, warps=[1])
        tile = g.smem("tile", (32,), dtype=ll.float32, stages=2)
        stage = g.depends(src=p, dst=c, via=[tile])
    with p:
        pair: ll.int32 = 0
        while pair < n:
            for offset in ll.unroll(range(2)):
                stage.bind(pair * 2 + offset)
                ri.smem_st_f32(tile, 0, ll.float32(0.0))
            pair = pair + 1
    with c:
        chunk: ll.int32 = 0
        while chunk < n * 2:
            stage.bind(chunk)
            ri.smem_ld_f32(tile, 0)
            chunk = chunk + 1
"""
    analyze_dataflow_source(source)


def test_accepts_pipeline_prologue_followed_by_loop() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def prologue_stream(n: ll.int32):
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        c = g.role("c", hw=HW.CUDACore, warps=[1])
        tile = g.smem("tile", (32,), dtype=ll.float32, stages=3)
        stage = g.depends(src=p, dst=c, via=[tile])
    with p:
        stage.bind(0)
        ri.smem_st_f32(tile, 0, ll.float32(0.0))
        stage.bind(1)
        ri.smem_st_f32(tile, 0, ll.float32(0.0))
        i: ll.int32 = 0
        while i < n:
            stage.bind(i + 2)
            ri.smem_st_f32(tile, 0, ll.float32(0.0))
            i = i + 1
    with c:
        j: ll.int32 = 0
        while j < n:
            stage.bind(j)
            ri.smem_ld_f32(tile, 0)
            j = j + 1
        stage.bind(n)
        ri.smem_ld_f32(tile, 0)
        stage.bind(n + 1)
        ri.smem_ld_f32(tile, 0)
"""
    analyze_dataflow_source(source)


def test_rejects_ambiguous_memory_ownership() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def ambiguous_owner():
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        c0 = g.role("c0", hw=HW.CUDACore, warps=[1])
        c1 = g.role("c1", hw=HW.CUDACore, warps=[2])
        tile = g.smem("tile", (32,), dtype=ll.float32)
        s0 = g.depends(src=p, dst=c0, via=[tile])
        s1 = g.depends(src=p, dst=c1, via=[tile])
    with p:
        s0.bind(0)
        s1.bind(0)
        ri.smem_st_f32(tile, 0, ll.val_cast(0.0, ll.float32))
    with c0:
        s0.bind(0)
        ri.smem_ld_f32(tile, 0)
    with c1:
        s1.bind(0)
        ri.smem_ld_f32(tile, 0)
"""
    with pytest.raises(ValueError, match="belongs to multiple channels"):
        analyze_dataflow_source(source)


def test_rejects_incomplete_tma_region() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def incomplete_tma(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]]):
    with dataflow() as g:
        p = g.role("p", hw=HW.TMA, warps=[0])
        c = g.role("c", hw=HW.CUDACore, warps=[1])
        tile = g.smem("tile", (1, 32), dtype=ll.float32)
        stage = g.depends(src=p, dst=c, via=[tile])
    with p:
        stage.bind(0)
        ri.tma_ld_2d(tile, desc, (0, 0), smem_coord=(0, 16), shape=(1, 16))
    with c:
        stage.bind(0)
        ri.smem_ld_f32(tile, 0)
"""
    with pytest.raises(ValueError, match="cover 16 of 32 elements"):
        analyze_dataflow_source(source)


def test_accepts_cooperative_tma_coverage() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def cooperative_tma(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]]):
    with dataflow() as g:
        p0 = g.role("p0", hw=HW.TMA, warps=[0])
        p1 = g.role("p1", hw=HW.TMA, warps=[1])
        c = g.role("c", hw=HW.CUDACore, warps=[2])
        tile = g.smem("tile", (1, 32), dtype=ll.float32)
        stage = g.depends(src=[p0, p1], dst=c, via=[tile])
    with p0:
        stage.bind(0)
        ri.tma_ld_2d(tile, desc, (0, 0), smem_coord=(0, 0), shape=(1, 16))
    with p1:
        stage.bind(0)
        ri.tma_ld_2d(tile, desc, (16, 0), smem_coord=(0, 16), shape=(1, 16))
    with c:
        stage.bind(0)
        ri.smem_ld_f32(tile, 0)
"""
    graph = analyze_dataflow_source(source)
    assert [pipeline["tx_bytes"] for pipeline in graph["pipelines"]] == [64, 64]


def test_rejects_shifted_stage_overlay() -> None:

    class SMem:

        def __init__(self) -> None:
            self.shape = (8, )
            self.dtype = "float32"
            self.stages = 2

    graph = {"memories": {"a": SMem(), "b": SMem()}}
    mapping = [
        MemoryMappingEntry("a", "storage", offset=0, stage_stride=32),
        MemoryMappingEntry("b", "storage", offset=32, stage_stride=32),
    ]
    with pytest.raises(ValueError, match="shifted-stage overlay"):
        analyze_buffer_overlays(graph, mapping)


def test_zero_extent_view_at_nonzero_offset_has_no_overlap() -> None:

    class Mem:

        def __init__(self) -> None:
            self.shape = (8, )
            self.dtype = "float32"
            self.stages = 1

    graph = {"memories": {"storage": Mem(), "view": Mem()}}
    mapping = [MemoryMappingEntry("view", "storage", offset=32, extent=0)]
    analysis = analyze_buffer_overlays(graph, mapping)
    assert not analysis.pairs


def test_rejects_read_only_channel_alias_before_codegen() -> None:
    source = PRELUDE + """
from little_kernel.backends.nv.sm100.passes.dataflow.buffer_overlay import MemoryMappingEntry

@lk.ll_kernel(backend="cuda", is_entry=True)
def read_only_alias() -> ll.void:
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        c = g.role("c", hw=HW.CUDACore, warps=[1])
        storage = g.smem("storage", (1,), dtype=ll.bfloat16)
        view = g.smem("view", (1,), dtype=ll.bfloat16)
        g.memory_mapping_plan = [MemoryMappingEntry("view", "storage", space="smem")]
        stage = g.depends(src=p, dst=c, via=[view])
    with p:
        stage.bind(0)
        ri.smem_st_bf16(storage, 0, ll.float_to_bf16(ll.float32(0.0)))
    with c:
        stage.bind(0)
        ri.smem_ld_bf16(view, 0)
"""
    with pytest.raises(ValueError, match=r"read-only channel alias 'view' -> 'storage' is unsafe"):
        analyze_dataflow_source(source)


def test_same_producer_overlay_reuses_source_release_barrier() -> None:
    source = PRELUDE + """
from little_kernel.backends.nv.sm100.passes.dataflow.buffer_overlay import MemoryMappingEntry

@lk.ll_kernel(backend="cuda", is_entry=True)
def same_producer_overlay() -> ll.void:
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        c = g.role("c", hw=HW.CUDACore, warps=[1])
        a = g.smem("a", (1,), dtype=ll.float32)
        b = g.smem("b", (1,), dtype=ll.float32)
        g.memory_mapping_plan = [MemoryMappingEntry("b", "a", space="smem")]
        stage_a = g.depends(src=p, dst=c, via=[a])
        stage_b = g.depends(src=p, dst=c, via=[b])
    with p:
        stage_a.bind(0)
        ri.smem_st_f32(a, 0, ll.float32(1.0))
        stage_b.bind(0)
        ri.smem_st_f32(b, 0, ll.float32(2.0))
    with c:
        stage_a.bind(0)
        ri.smem_ld_f32(a, 0)
        stage_b.bind(0)
        ri.smem_ld_f32(b, 0)
"""
    code = compile_dataflow_cuda(source, num_threads=64)
    assert code.count("PIPE_p_to_c_a.producer_acquire();") == 2
    assert "PIPE_p_to_c_b_a" not in code
    assert code.index("PIPE_p_to_c_a.consumer_release();") < code.index("PIPE_p_to_c_b.consumer_wait();")
    with pytest.raises(ValueError, match="same-producer reuse currently requires single-stage"):
        compile_dataflow_cuda(source.replace("dtype=ll.float32)", "dtype=ll.float32, stages=2)"), num_threads=64)


def test_overlay_read_after_full_reinitialization_is_safe() -> None:
    source = PRELUDE + """
from little_kernel.backends.nv.sm100.passes.dataflow.buffer_overlay import MemoryMappingEntry

@lk.ll_kernel(backend="cuda", is_entry=True)
def reinitialized_overlay() -> ll.void:
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        c = g.role("c", hw=HW.CUDACore, warps=[1])
        a = g.smem("a", (1,), dtype=ll.float32)
        b = g.smem("b", (1,), dtype=ll.float32)
        g.memory_mapping_plan = [MemoryMappingEntry("b", "a", space="smem")]
        stage_a = g.depends(src=p, dst=c, via=[a])
        stage_b = g.depends(src=p, dst=c, via=[b])
    with p:
        stage_a.bind(0)
        ri.smem_st_f32(a, 0, ll.float32(1.0))
        stage_b.bind(0)
        ri.smem_st_f32(b, 0, ll.float32(2.0))
        ri.smem_ld_f32(b, 0)
    with c:
        stage_a.bind(0)
        ri.smem_ld_f32(a, 0)
        stage_b.bind(0)
        ri.smem_ld_f32(b, 0)
"""
    compile_dataflow_cuda(source, num_threads=64)


def test_transitive_channel_avoids_redundant_overlay_acquire() -> None:
    source = PRELUDE + """
from little_kernel.backends.nv.sm100.passes.dataflow.buffer_overlay import MemoryMappingEntry

@lk.ll_kernel(backend="cuda", is_entry=True)
def transitive_overlay() -> ll.void:
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        c = g.role("c", hw=HW.CUDACore, warps=[1])
        a = g.smem("a", (1,), dtype=ll.float32)
        bridge = g.smem("bridge", (1,), dtype=ll.float32)
        b = g.smem("b", (1,), dtype=ll.float32)
        g.memory_mapping_plan = [MemoryMappingEntry("b", "a", space="smem")]
        stage_a = g.depends(src=p, dst=c, via=[a])
        stage_bridge = g.depends(src=c, dst=p, via=[bridge])
        stage_b = g.depends(src=p, dst=c, via=[b])
    with p:
        stage_a.bind(0)
        ri.smem_st_f32(a, 0, ll.float32(1.0))
        stage_bridge.bind(0)
        ri.smem_ld_f32(bridge, 0)
        stage_b.bind(0)
        ri.smem_st_f32(b, 0, ll.float32(2.0))
    with c:
        stage_a.bind(0)
        ri.smem_ld_f32(a, 0)
        stage_bridge.bind(0)
        ri.smem_st_f32(bridge, 0, ll.float32(3.0))
        stage_b.bind(0)
        ri.smem_ld_f32(b, 0)
"""
    code = compile_dataflow_cuda(source, num_threads=64)
    assert code.count("PIPE_p_to_c_a.producer_acquire();") == 1
    assert "PIPE_p_to_c_b_a" not in code


def test_canonical_channel_tracks_exact_read_only_typed_view() -> None:
    source = PRELUDE + """
from little_kernel.backends.nv.sm100.passes.dataflow.buffer_overlay import MemoryMappingEntry

@lk.ll_kernel(backend="cuda", is_entry=True)
def typed_read_view() -> ll.void:
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        c = g.role("c", hw=HW.CUDACore, warps=[1])
        storage = g.smem("storage", (64, 128), dtype=ll.bfloat16, stages=2, major="K")
        view = g.smem("view", (128, 64), dtype=ll.bfloat16, stages=2, major="MN")
        g.memory_mapping_plan = [MemoryMappingEntry("view", "storage", space="smem", extent=16384, stage_stride=16384)]
        stage = g.depends(src=p, dst=c, via=[storage])
    with p:
        stage.bind(0)
        ri.smem_st_bf16(storage, 0, ll.float_to_bf16(ll.float32(0.0)))
    with c:
        stage.bind(0)
        ri.smem_ld_bf16(storage, 0)
        ri.smem_ld_bf16(view, 0)
"""
    graph = analyze_dataflow_source(source)
    reads = [
        event for event in graph["mem_events"]
        if event.role == "c" and event.mem == "storage" and event.access.value == "r"
    ]
    assert len(reads) == 2
    assert not any(event.mem == "view" for event in graph["mem_events"])


def test_tma_store_stream_separates_independent_release_frontiers() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def independent_stores(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]]):
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        s = g.role("s", hw=HW.TMA, warps=[1])
        a = g.smem("a", (1, 32), dtype=ll.float32, stages=3)
        b = g.smem("b", (1, 32), dtype=ll.float32, stages=5)
        sa = g.depends(src=p, dst=s, via=[a])
        sb = g.depends(src=p, dst=s, via=[b])
    with p:
        sa.bind(0)
        ri.smem_st_f32(a, 0, ll.val_cast(0.0, ll.float32))
        sb.bind(0)
        ri.smem_st_f32(b, 0, ll.val_cast(0.0, ll.float32))
    with s:
        issuer: ll.uint32 = ll.elect_one()
        sa.bind(0)
        if issuer == 1:
            ri.tma_st_2d(a, desc, (0, 0), shape=(1, 32))
        sb.bind(0)
        if issuer == 1:
            ri.tma_st_2d(b, desc, (0, 1), shape=(1, 32))
"""
    stream = analyze_dataflow_source(source)["tma_store_streams"][0]
    assert len(stream["groups"]) == 2
    assert [len(group["completions"]) for group in stream["groups"]] == [1, 1]
    assert stream["groups"][0]["completions"] != stream["groups"][1]["completions"]


def test_tma_store_stream_retires_coupled_producer_stages_together() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def coupled_producer_stages(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]]) -> ll.void:
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        s = g.role("s", hw=HW.TMA, warps=[1])
        a = g.smem("a", (1, 32), dtype=ll.float32)
        b = g.smem("b", (1, 32), dtype=ll.float32)
        sa = g.depends(src=p, dst=s, via=[a])
        sb = g.depends(src=p, dst=s, via=[b])
    with p:
        sa.bind(0)
        sb.bind(0)
        for _ in ll.unroll(range(1)):
            ri.smem_st_f32(a, 0, ll.val_cast(0.0, ll.float32))
            ri.smem_st_f32(b, 0, ll.val_cast(0.0, ll.float32))
    with s:
        issuer: ll.uint32 = ll.elect_one()
        sa.bind(0)
        if issuer == 1:
            ri.tma_st_2d(a, desc, (0, 0), shape=(1, 32))
        sb.bind(0)
        if issuer == 1:
            ri.tma_st_2d(b, desc, (0, 1), shape=(1, 32))
"""
    code = compile_dataflow_cuda(source, num_threads=64)
    assert "__lk_tma_store_stream_s.wait<1>()" in code
    assert "__lk_tma_store_stream_s.wait<0>()" in code


def test_tma_store_stream_keeps_joint_channel_in_one_group() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def joint_store(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]]):
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        s = g.role("s", hw=HW.TMA, warps=[1])
        a = g.smem("a", (1, 32), dtype=ll.float32, stages=4)
        b = g.smem("b", (1, 32), dtype=ll.float32, stages=4)
        stage = g.depends(src=p, dst=s, via=[a, b])
    with p:
        stage.bind(0)
        ri.smem_st_f32(a, 0, ll.val_cast(0.0, ll.float32))
        ri.smem_st_f32(b, 0, ll.val_cast(0.0, ll.float32))
    with s:
        issuer: ll.uint32 = ll.elect_one()
        stage.bind(0)
        if issuer == 1:
            ri.tma_st_2d(a, desc, (0, 0), shape=(1, 32))
            ri.tma_st_2d(b, desc, (0, 1), shape=(1, 32))
"""
    stream = analyze_dataflow_source(source)["tma_store_streams"][0]
    assert len(stream["groups"]) == 1
    assert len(stream["groups"][0]["completions"]) == 1


def test_tma_store_stream_uses_last_source_use_for_interleaved_stores() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def interleaved_stores(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]]):
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        s = g.role("s", hw=HW.TMA, warps=[1])
        a = g.smem("a", (1, 32), dtype=ll.float32, stages=2)
        b = g.smem("b", (1, 32), dtype=ll.float32, stages=2)
        sa = g.depends(src=p, dst=s, via=[a])
        sb = g.depends(src=p, dst=s, via=[b])
    with p:
        sa.bind(0)
        ri.smem_st_f32(a, 0, ll.val_cast(0.0, ll.float32))
        sb.bind(0)
        ri.smem_st_f32(b, 0, ll.val_cast(0.0, ll.float32))
    with s:
        issuer: ll.uint32 = ll.elect_one()
        sa.bind(0)
        sb.bind(0)
        if issuer == 1:
            ri.tma_st_2d(a, desc, (0, 0), smem_coord=(0, 0), shape=(1, 16))
            ri.tma_st_2d(b, desc, (0, 1), smem_coord=(0, 0), shape=(1, 16))
            ri.tma_st_2d(a, desc, (16, 0), smem_coord=(0, 16), shape=(1, 16))
            ri.tma_st_2d(b, desc, (16, 1), smem_coord=(0, 16), shape=(1, 16))
"""
    stream = analyze_dataflow_source(source)["tma_store_streams"][0]
    assert len(stream["groups"]) == 2
    assert [len(group["completions"]) for group in stream["groups"]] == [1, 1]


def test_rejects_tma_store_depth_above_hardware_limit() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def many_store_stages(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]]) -> ll.void:
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        s = g.role("s", hw=HW.TMA, warps=[1])
        tile = g.smem("tile", (1, 32), dtype=ll.float32, stages=10)
        stage = g.depends(src=p, dst=s, via=[tile])
    with p:
        stage.bind(0)
        ri.smem_st_f32(tile, 0, ll.val_cast(0.0, ll.float32))
    with s:
        issuer: ll.uint32 = ll.elect_one()
        stage.bind(0)
        if issuer == 1:
            ri.tma_st_2d(tile, desc, (0, 0), shape=(1, 32))
"""
    with pytest.raises(ValueError, match="depth 10 exceeds 8"):
        compile_dataflow_cuda(source, num_threads=64)


def test_tma_store_stream_commits_from_the_whole_domain() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def rotating_issuer(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]], owner: ll.int32) -> ll.void:
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        s = g.role("s", hw=HW.TMA, warps=[1])
        tile = g.smem("tile", (1, 32), dtype=ll.float32, stages=2)
        stage = g.depends(src=p, dst=s, via=[tile])
    lane: ll.int32 = ll.get_lane_idx()
    with p:
        stage.bind(0)
        ri.smem_st_f32(tile, 0, ll.val_cast(0.0, ll.float32))
    with s:
        issuer: ll.bool_ = lane == owner
        stage.bind(0)
        if issuer:
            ri.tma_st_2d(tile, desc, (0, 0), shape=(1, 32))
"""
    code = compile_dataflow_cuda(source, num_threads=64)
    assert "__lk_tma_store_stream_s.commit();" in code
    assert "TmaStoreStream __lk_tma_store_stream_s" in code
    assert "commit(0, bool(issuer)" not in code


def test_rejects_tma_store_without_named_issuer_guard() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def missing_issuer(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]]) -> ll.void:
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        s = g.role("s", hw=HW.TMA, warps=[1])
        tile = g.smem("tile", (1, 32), dtype=ll.float32)
        stage = g.depends(src=p, dst=s, via=[tile])
    with p:
        stage.bind(0)
        ri.smem_st_f32(tile, 0, ll.val_cast(0.0, ll.float32))
    with s:
        stage.bind(0)
        ri.tma_st_2d(tile, desc, (0, 0), shape=(1, 32))
"""
    with pytest.raises(ValueError, match="named issuer guard"):
        analyze_dataflow_source(source)


def test_rejects_tma_issuer_defined_inside_payload_predicate() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def conditional_issuer(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]], valid: ll.int32) -> ll.void:
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        s = g.role("s", hw=HW.TMA, warps=[1])
        tile = g.smem("tile", (1, 32), dtype=ll.float32)
        stage = g.depends(src=p, dst=s, via=[tile])
    with p:
        stage.bind(0)
        ri.smem_st_f32(tile, 0, ll.val_cast(0.0, ll.float32))
    with s:
        stage.bind(0)
        if valid != 0:
            issuer: ll.uint32 = ll.elect_one()
            if issuer == 1:
                ri.tma_st_2d(tile, desc, (0, 0), shape=(1, 32))
"""
    with pytest.raises(ValueError, match="before any payload predicate"):
        analyze_dataflow_source(source)


def test_same_role_tma_store_uses_static_stream() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def same_role_store(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]]) -> ll.void:
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0, 1, 2, 3])
        tile = g.smem("tile", (1, 128), dtype=ll.float32, stages=3)
        stage = g.depends(src=p, dst=p, via=[tile])
    lane: ll.int32 = ll.get_lane_idx()
    warp: ll.int32 = ll.threadIdx_x() // 32
    with p:
        stage.bind(0)
        ri.smem_st_f32(tile, 0, ll.val_cast(0.0, ll.float32))
        if warp == 0:
            issuer: ll.bool_ = lane == 7
            if issuer:
                ri.tma_st_2d(tile, desc, (0, 0), shape=(1, 32))
"""
    code = compile_dataflow_cuda(source, num_threads=128)
    assert "PipelineTmaStoreSameRole<3, 128>" in code
    assert "TmaStoreStream __lk_tma_store_stream_p" in code
    assert "__lk_tma_store_stream_p.commit();" in code
    assert "switch (" not in code


def test_tma_store_issuer_outside_loop_commits_each_iteration() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def looped_store(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]], count: ll.int32) -> ll.void:
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        tile = g.smem("tile", (1, 32), dtype=ll.float32)
        stage = g.depends(src=p, dst=p, via=[tile])
    with p:
        issuer: ll.uint32 = ll.elect_one()
        chunk: ll.int32 = 0
        while chunk < count:
            stage.bind(chunk)
            ri.smem_st_f32(tile, 0, ll.val_cast(0.0, ll.float32))
            if issuer == 1:
                ri.tma_st_2d(tile, desc, (0, chunk), shape=(1, 32))
            chunk = chunk + 1
"""
    code = compile_dataflow_cuda(source, num_threads=32)
    # Order invariant: enter loop, then warp-commit the store stream, then
    # advance the loop counter.  Do not pin exact parenthesization (const
    # folding removes redundant parens).
    loop = code.index("while (chunk < count)")
    commit = code.index("__lk_tma_store_stream_p.commit();", loop)
    advance = code.index("chunk = chunk + 1;", loop)
    assert loop < commit < advance


def test_tma_store_payload_loop_commits_outside_issuer_guard() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def looped_store(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]], count: ll.int32) -> ll.void:
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0])
        tile = g.smem("tile", (1, 32), dtype=ll.float32)
        stage = g.depends(src=p, dst=p, via=[tile])
    with p:
        issuer: ll.uint32 = ll.elect_one()
        chunk: ll.int32 = 0
        while chunk < count:
            stage.bind(chunk)
            ri.smem_st_f32(tile, 0, ll.val_cast(0.0, ll.float32))
            if issuer == 1:
                for peer in range(3):
                    ri.tma_st_2d(tile, desc, (0, chunk + peer), shape=(1, 32))
            chunk = chunk + 1
"""
    code = compile_dataflow_cuda(source, num_threads=32)
    loop = code.index("while (chunk < count)")
    commit = code.index("__lk_tma_store_stream_p.commit();", loop)
    advance = code.index("chunk = chunk + 1;", loop)
    assert loop < commit < advance

    # Both the payload loop and issuer guard must close before the warp commit.
    prefix = code[code.index("if (issuer == 1)", loop):commit]
    assert prefix.count("{") == prefix.count("}")


def test_rejects_same_role_tma_store_before_generic_write() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def same_role_store_first(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]]) -> ll.void:
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0, 1, 2, 3])
        tile = g.smem("tile", (1, 128), dtype=ll.float32, stages=3)
        stage = g.depends(src=p, dst=p, via=[tile])
    lane: ll.int32 = ll.get_lane_idx()
    warp: ll.int32 = ll.threadIdx_x() // 32
    with p:
        stage.bind(0)
        if warp == 0:
            issuer: ll.bool_ = lane == 7
            if issuer:
                ri.tma_st_2d(tile, desc, (0, 0), shape=(1, 128))
        ri.smem_st_f32(tile, 0, ll.val_cast(0.0, ll.float32))
"""
    with pytest.raises(ValueError, match="producer effects before TMA store"):
        compile_dataflow_cuda(source, num_threads=128)


def test_same_role_tma_relay_uses_one_cursor_pipeline() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def same_role_tma_relay(
    src: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dst: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    count: ll.int32,
) -> ll.void:
    with dataflow() as g:
        relay = g.role("relay", hw=HW.TMA, warps=[0])
        tile = g.smem("tile", (128, 64), dtype=ll.bfloat16, stages=3)
        stage = g.depends(src=relay, dst=relay, via=[tile])
    with relay:
        i: ll.int32 = 0
        while i < count:
            stage.bind(i)
            issuer: ll.uint32 = ll.elect_one()
            if issuer == 1:
                ri.tma_ld_2d(tile, src, (0, 0), shape=(128, 64))
                ri.tma_st_2d(tile, dst, (0, 0), shape=(128, 64))
            i = i + 1
"""
    code = compile_dataflow_cuda(source, num_threads=32)
    pipe = "PIPE_relay_to_relay_tile"
    assert "PipelineTmaTmaStoreSameRole<3, 16384, 1>" in code
    assert f"{pipe}_consumer" not in code
    assert "TmaStoreStream" not in code
    assert f"{pipe}.consumer_submit();" not in code
    assert f"{pipe}.consumer_release();" in code
    assert f"{pipe}.producer_tail();" in code


def test_rejects_same_role_tma_relay_store_before_load() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def same_role_tma_relay_store_first(
    src: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dst: ll.const[ll.grid_constant[ll.TmaDescriptor]],
) -> ll.void:
    with dataflow() as g:
        relay = g.role("relay", hw=HW.TMA, warps=[0])
        tile = g.smem("tile", (128, 64), dtype=ll.bfloat16, stages=3)
        stage = g.depends(src=relay, dst=relay, via=[tile])
    with relay:
        stage.bind(0)
        issuer: ll.uint32 = ll.elect_one()
        if issuer == 1:
            ri.tma_st_2d(tile, dst, (0, 0), shape=(128, 64))
            ri.tma_ld_2d(tile, src, (0, 0), shape=(128, 64))
"""
    with pytest.raises(ValueError, match="producer effects before TMA store"):
        compile_dataflow_cuda(source, num_threads=32)


def test_rejects_same_role_tma_relay_optional_store() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def same_role_tma_relay_optional_store(
    src: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dst: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    valid: ll.int32,
) -> ll.void:
    with dataflow() as g:
        relay = g.role("relay", hw=HW.TMA, warps=[0])
        tile = g.smem("tile", (128, 64), dtype=ll.bfloat16, stages=3)
        stage = g.depends(src=relay, dst=relay, via=[tile])
    with relay:
        stage.bind(0)
        issuer: ll.uint32 = ll.elect_one()
        if issuer == 1:
            ri.tma_ld_2d(tile, src, (0, 0), shape=(128, 64))
            do_store: ll.bool_ = valid != 0
            if do_store:
                ri.tma_st_2d(tile, dst, (0, 0), shape=(128, 64))
"""
    with pytest.raises(ValueError, match="control-path/cardinality mismatch"):
        compile_dataflow_cuda(source, num_threads=32)


def test_tma_load_to_tma_store_uses_relay_pipeline() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def tma_relay(
    src: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dst: ll.const[ll.grid_constant[ll.TmaDescriptor]],
) -> ll.void:
    with dataflow() as g:
        load = g.role("load", hw=HW.TMA, warps=[0])
        store = g.role("store", hw=HW.TMA, warps=[1])
        tile = g.smem("tile", (128, 64), dtype=ll.bfloat16, stages=2)
        stage = g.depends(src=load, dst=store, via=[tile])
    with load:
        load_issuer: ll.uint32 = ll.elect_one()
        stage.bind(0)
        if load_issuer == 1:
            ri.tma_ld_2d(tile, src, (0, 0), shape=(128, 64))
    with store:
        store_issuer: ll.uint32 = ll.elect_one()
        stage.bind(0)
        if store_issuer == 1:
            ri.tma_st_2d(tile, dst, (0, 0), shape=(128, 64))
"""
    code = compile_dataflow_cuda(source, num_threads=64)
    assert "PipelineTmaTmaStore<2, 16384, 1, 1>" in code
    assert "TmaStoreStream __lk_tma_store_stream_store" in code
    assert "__lk_tma_store_stream_store.commit();" in code
    assert "PIPE_load_to_store_tile.consumer_release();" in code
    assert "PIPE_load_to_store_tile.retire_completed();" in code
    assert "consumer_submit" not in code
    assert re.search(
        r"__lk_tma_store_stream_store\.commit\(\);\s*"
        r"PIPE_load_to_store_tile\.consumer_release\(\);",
        code,
    )
    assert re.search(
        r"__lk_tma_store_stream_store\.wait<\d+>\(\);\s*"
        r"PIPE_load_to_store_tile\.retire_completed\(\);",
        code,
    )
    assert "__lk_tma_store_stream_store.tail();" in code
    assert "PIPE_load_to_store_tile.producer_tail();" not in code


def test_raw_1d_tma_relay_uses_pointer_bulk_copies() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def raw_tma_relay(src: ll.ptr[ll.bfloat16], dst: ll.ptr[ll.bfloat16]) -> ll.void:
    with dataflow() as g:
        load = g.role("load", hw=HW.TMA, warps=[0])
        store = g.role("store", hw=HW.TMA, warps=[1])
        tile = g.smem("tile", (8192,), dtype=ll.bfloat16)
        stage = g.depends(src=load, dst=store, via=[tile])
    with load:
        load_issuer: ll.uint32 = ll.elect_one()
        stage.bind(0)
        if load_issuer == 1:
            ri.tma_ld_1d(tile, src, 16384)
    with store:
        store_issuer: ll.uint32 = ll.elect_one()
        stage.bind(0)
        if store_issuer == 1:
            ri.tma_st_1d(tile, dst, 16384)
"""
    code = compile_dataflow_cuda(source, num_threads=64)
    assert "PipelineTmaTmaStore<1, 16384, 1, 1>" in code
    assert "tma_copy_1d_g2s" in code
    assert "tma_copy_1d_s2g" in code
    assert "prefetch_tma_descriptor" not in code


def test_rejects_tma_load_without_elected_issuer() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def missing_load_issuer(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]]) -> ll.void:
    with dataflow() as g:
        load = g.role("load", hw=HW.TMA, warps=[0])
        consume = g.role("consume", hw=HW.CUDACore, warps=[1])
        tile = g.smem("tile", (1, 32), dtype=ll.float32)
        stage = g.depends(src=load, dst=consume, via=[tile])
    with load:
        stage.bind(0)
        ri.tma_ld_2d(tile, desc, (0, 0), shape=(1, 32))
    with consume:
        stage.bind(0)
        ri.smem_ld_f32(tile, 0)
"""
    with pytest.raises(ValueError, match="named issuer guard"):
        compile_dataflow_cuda(source, num_threads=64)


def test_cg2_tma_to_generic_uses_pair_leader_and_cta_count() -> None:
    source = PRELUDE + """
@lk.ll_kernel(
    backend="cuda",
    is_entry=True,
    cluster_shape=(2, 1, 1),
    cta_group_shape=(2, 1, 1),
)
def cg2_tma_generic(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]]) -> ll.void:
    with dataflow() as g:
        load = g.role("load", hw=HW.TMA, warps=[0])
        consume = g.role("consume", hw=HW.CUDACore, warps=[1])
        tile = g.smem("tile", (128, 64), dtype=ll.bfloat16)
        stage = g.depends(src=load, dst=consume, via=[tile])
        plan = g.tma_transfer(
            mcast_group_dim=None,
            inter_cta_coord_dim=None,
            inter_cta_stride=0,
        )
    with load:
        issuer: ll.uint32 = ll.elect_one()
        stage.bind(0)
        if issuer == 1:
            ri.tma_ld_2d(tile, desc, (0, 0), plan, shape=(128, 64))
    with consume:
        stage.bind(0)
        ri.smem_ld_bf16(tile, 0)
"""
    code = compile_dataflow_cuda(source, num_threads=64)
    assert ("PipelineTmaGeneric<1, 32768, 2, 64, 2, false, 0, 32>") in code


def test_multicast_tma_to_generic_counts_destination_ctas() -> None:
    source = PRELUDE + """
@lk.ll_kernel(
    backend="cuda",
    is_entry=True,
    cluster_shape=(4, 4, 1),
    cta_group_shape=(2, 1, 1),
)
def mcast_tma_generic(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]]) -> ll.void:
    with dataflow() as g:
        load = g.role("load", hw=HW.TMA, warps=[0])
        consume = g.role("consume", hw=HW.CUDACore, warps=[1])
        tile = g.smem("tile", (128, 64), dtype=ll.bfloat16)
        stage = g.depends(src=load, dst=consume, via=[tile])
        plan = g.tma_transfer(
            mcast_group_dim="y",
            inter_cta_coord_dim=None,
            inter_cta_stride=0,
        )
    with load:
        issuer: ll.uint32 = ll.elect_one()
        stage.bind(0)
        if issuer == 1:
            ri.tma_ld_2d(tile, desc, (0, 0), plan, shape=(128, 64))
    with consume:
        stage.bind(0)
        ri.smem_ld_bf16(tile, 0)
"""
    code = compile_dataflow_cuda(source, num_threads=64)
    assert ("PipelineTmaGeneric<1, 32768, 1, 256, 2, true, 0, 32>") in code
    assert "group_index<4, 4, 1, 2, 1, 1>()" in code
    assert "? 13107 :" in code


def test_same_role_store_pipelines_get_unique_named_barriers() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def two_same_role_stores(desc: ll.const[ll.grid_constant[ll.TmaDescriptor]]) -> ll.void:
    with dataflow() as g:
        p = g.role("p", hw=HW.CUDACore, warps=[0, 1])
        a = g.smem("a", (1, 32), dtype=ll.float32, stages=2)
        b = g.smem("b", (1, 32), dtype=ll.float32, stages=2)
        sa = g.depends(src=p, dst=p, via=[a])
        sb = g.depends(src=p, dst=p, via=[b])
    lane: ll.int32 = ll.get_lane_idx()
    warp: ll.int32 = ll.threadIdx_x() // 32
    with p:
        sa.bind(0)
        ri.smem_st_f32(a, 0, ll.val_cast(0.0, ll.float32))
        if warp == 0:
            issuer0: ll.bool_ = lane == 7
            if issuer0:
                ri.tma_st_2d(a, desc, (0, 0), shape=(1, 32))
        sb.bind(0)
        ri.smem_st_f32(b, 0, ll.val_cast(0.0, ll.float32))
        if warp == 1:
            issuer1: ll.bool_ = lane == 7
            if issuer1:
                ri.tma_st_2d(b, desc, (0, 1), shape=(1, 32))
"""
    code = compile_dataflow_cuda(source, num_threads=64)
    assert "PipelineTmaStoreSameRole<2, 64> PIPE_p_to_p_a" in code
    assert "PipelineTmaStoreSameRole<2, 64, 2> PIPE_p_to_p_b" in code


def test_generic_pipeline_counts_thread_participants() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def generic_thread_counts() -> ll.void:
    with dataflow() as g:
        produce = g.role("produce", hw=HW.CUDACore, warps=[0])
        consume = g.role("consume", hw=HW.CUDACore, warps=[1, 2])
        tile = g.smem("tile", (32,), dtype=ll.float32, stages=2)
        stage = g.depends(src=produce, dst=consume, via=[tile])
    with produce:
        stage.bind(0)
        ri.smem_st_f32(tile, 0, ll.val_cast(1.0, ll.float32))
    with consume:
        stage.bind(0)
        ri.smem_ld_f32(tile, 0)
"""
    code = compile_dataflow_cuda(source, num_threads=96)
    assert "PipelineGenericGeneric<2, 32, 64>" in code
    assert "uint64_t PIPE_produce_to_consume_tile_producer[2];" in code
    assert "uint64_t PIPE_produce_to_consume_tile_consumer[2];" in code
    assert ("PIPE_produce_to_consume_tile(__lk_smem.PIPE_produce_to_consume_tile_producer, "
            "__lk_smem.PIPE_produce_to_consume_tile_consumer)") in code


def test_generic_smem_to_umma_selects_proxy_fenced_family() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def generic_to_umma() -> ll.void:
    with dataflow() as g:
        produce = g.role("produce", hw=HW.CUDACore, warps=[0])
        consume = g.role("consume", hw=HW.TensorCore, warps=[1])
        a = g.smem("a", (128, 64), dtype=ll.bfloat16, major="K")
        b = g.smem("b", (128, 64), dtype=ll.bfloat16, major="K")
        d = g.tmem("d", (128, 128), dtype=ll.float32)
        stage = g.depends(src=produce, dst=consume, via=[a, b])
    with produce:
        stage.bind(0)
        zero: ll.bfloat16 = ll.float_to_bf16(ll.float32(0.0))
        ri.smem_st_bf16(a, 0, zero)
        ri.smem_st_bf16(b, 0, zero)
    with consume:
        issuer: ll.uint32 = ll.elect_one()
        stage.bind(0)
        if issuer == 1:
            ad: ll.uint64 = ri.make_umma_smem_desc(a, (0, 0))
            bd: ll.uint64 = ri.make_umma_smem_desc(b, (0, 0))
            ri.umma_f16_ss_1sm_init(d, d, ad, bd, out_shape=(128, 128))
"""
    code = compile_dataflow_cuda(source, num_threads=64)
    assert "PipelineGenericUmma<1, 32, 1, 1>" in code
    assert "PIPE_produce_to_consume_a_b.producer_commit();" in code


def test_tmem_store_to_tmem_load_selects_wait_family() -> None:
    source = PRELUDE + """
@lk.ll_kernel(backend="cuda", is_entry=True)
def tmem_store_to_load() -> ll.void:
    with dataflow() as g:
        produce = g.role("produce", hw=HW.CUDACore, warps=[0])
        consume = g.role("consume", hw=HW.CUDACore, warps=[1])
        tile = g.tmem("tile", (128, 32), dtype=ll.float32)
        stage = g.depends(src=produce, dst=consume, via=[tile])
    with produce:
        stage.bind(0)
        zeros = ll.zeros([4], dtype=ll.uint32, scope="local")
        ri.tmem_st_32dp_32bxn(tile, ll.uint32(tile), zeros, 4)
    with consume:
        stage.bind(0)
        values = ll.empty([4], dtype=ll.uint32, scope="local")
        ri.tmem_ld_32dp_32bxn(tile, ll.uint32(tile), values, 4)
        ri.tmem_ld_fence(tile)
"""
    code = compile_dataflow_cuda(source, num_threads=64)
    assert "PipelineTmemStoreTmemLoad<1, 32, 32>" in code
    assert "tmem_st_32dp_32bxn_x4_fn" in code
    assert "tmem_ld_32dp_32bxn_x4_fn" in code
    assert code.index("tmem_ld_fence_fn();") < code.index("PIPE_produce_to_consume_tile.consumer_release();")

    with pytest.raises(ValueError, match="declares 4 registers but tmem_ld_32dp_32bxn length is 8"):
        compile_dataflow_cuda(source.replace("values, 4", "values, 8"), num_threads=64)


def _groupgemm_with_generic_umma(accum: str) -> tuple[str, str]:
    source = _kernel_source("v2_gemm_sm100.py")
    monolithic = """                    ri.umma_f16_ss_2sm_kblock(
                        C_tmem, ad, bd, K_TILE, out_shape=(PAIR_TILE_M, PAIR_TILE_N), zero_init=kb == 0,
                    )"""
    explicit = """                    for ki in ll.unroll(range(0, K_TILE, 16)):
                        ad_ki = ri.make_umma_smem_desc(A_smem, (0, ki))
                        bd_ki = ri.make_umma_smem_desc(B_smem, (0, ki))
                        if kb == 0 and ki == 0:
                            ri.umma_f16_ss_2sm_init(C_tmem, ll.uint32(C_tmem), ad_ki, bd_ki, out_shape=(PAIR_TILE_M, PAIR_TILE_N))
                        else:
                            ri.umma_f16_ss_2sm_accum(C_tmem, ll.uint32(C_tmem), ad_ki, bd_ki, out_shape=(PAIR_TILE_M, PAIR_TILE_N))"""
    generic = f"""                    for ki in ll.unroll(range(0, K_TILE, 16)):
                        ad_ki = ri.make_umma_smem_desc(A_smem, (0, ki))
                        bd_ki = ri.make_umma_smem_desc(B_smem, (0, ki))
                        ri.umma_f16_ss_2sm(
                            C_tmem, ll.uint32(C_tmem), ad_ki, bd_ki,
                            ll.uint32({accum}), out_shape=(PAIR_TILE_M, PAIR_TILE_N),
                        )"""
    import textwrap
    tree = ast.parse(source)
    expected = ast.parse(textwrap.dedent(monolithic)).body[0]
    matches = [node for node in ast.walk(tree) if ast.dump(node) == ast.dump(expected)]
    assert len(matches) == 1

    def replaced(replacement):

        class ReplaceCall(ast.NodeTransformer):

            def visit_Expr(self, node):
                if ast.dump(node) == ast.dump(expected):
                    return ast.parse(textwrap.dedent(replacement)).body
                return self.generic_visit(node)

        result = ReplaceCall().visit(ast.parse(source))
        return ast.unparse(ast.fix_missing_locations(result))

    return replaced(explicit), replaced(generic)


def test_generic_and_explicit_umma_generate_identical_cuda() -> None:
    explicit, generic = _groupgemm_with_generic_umma("(kb != 0) or (ki != 0)")
    explicit_code = compile_dataflow_cuda(explicit, num_threads=256)
    generic_code = compile_dataflow_cuda(generic, num_threads=256)
    assert generic_code == explicit_code
    # The dynamic accumulator predicate survives lowering and is passed as the
    # final static_cast<uint32_t> accum flag.  Parenthesization is whatever
    # const folding produces; assert the predicate structurally.
    assert re.search(
        r"umma_f16_2sm_fn\(static_cast<uint32_t>\(__lk_tmem_C_tmem\), ad_ki, bd_ki, 272630928, "
        r"static_cast<uint32_t>\(kb != 0 \|\| ki != 0\)\);",
        generic_code,
    )


def test_kblock_crosses_sw64_atoms_via_logical_descriptors() -> None:
    monolithic = _kernel_source("v2_gemm_sm100.py").replace("K_TILE = 64", "K_TILE = 96")
    code = compile_dataflow_cuda(monolithic, num_threads=256)
    assert code.count("umma_f16_2sm_fn(") == 13  # helper + 2 modes * 6 K16 instructions
    # Constant folding resolves logical K16 coordinates to descriptor address
    # units: each 32-column atom advances 512 units, its second half 2 units.
    # Both operands must preserve all non-address descriptor bits.
    for operand in ("ad", "bd"):
        offsets = {int(x) for x in re.findall(rf"{operand} & 18446744073709535232 \| {operand} \+ (\d+) & 16383", code)}
        assert offsets == {512 * (k // 32) + 2 * ((k % 32) // 16) for k in range(16, 96, 16)}
    assert ", 16, 512, 4)" in code

    out_of_bounds = monolithic.replace(
        "ri.make_umma_smem_desc(A_smem, (0, 0))",
        "ri.make_umma_smem_desc(A_smem, (0, 16))",
    )
    with pytest.raises(ValueError, match="descriptor coord exceeds logical shape"):
        compile_dataflow_cuda(out_of_bounds, num_threads=256)


def test_generic_umma_rejects_unproven_first_point() -> None:
    _, generic = _groupgemm_with_generic_umma("kb < num_k_blocks")
    with pytest.raises(ValueError, match="dynamic UMMA accum"):
        compile_dataflow_cuda(generic, num_threads=256)


@_requires_torch_demos
def test_kblock_zero_init_keeps_precise_effects_and_dynamic_codegen() -> None:
    source = _kernel_source("v2_gemm_sm100.py")
    graph = analyze_dataflow_source(source)
    d_effects = {
        event.effect_kind
        for event in graph["mem_events"]
        if event.intrin == "umma_f16_ss_2sm_kblock" and event.mem == "C_tmem"
    }
    assert d_effects == {W_TMEM_TCGEN05_COMMIT, RW_TMEM_TCGEN05_COMMIT}

    code = compile_dataflow_cuda(source, num_threads=256)
    assert "umma_f16_2sm_kblock_fn" not in code
    assert "umma_f16_2sm_fn(__lk_tmem_C_tmem, ad, bd, 272630928" in code
    # All four K16 instructions stay in the first sw64 atom. Test the
    # folded address offsets rather than a pre-fold expression spelling.
    for operand in ("ad", "bd"):
        offsets = {int(x) for x in re.findall(rf"{operand} & 18446744073709535232 \| {operand} \+ (\d+) & 16383", code)}
        assert offsets == {2, 4, 6}

    invalid = source.replace("zero_init=kb == 0", "zero_init=kb < num_k_blocks")
    with pytest.raises(ValueError, match="dynamic UMMA accum"):
        compile_dataflow_cuda(invalid, num_threads=256)

    invalid_k = source.replace("K_TILE, out_shape=", "K_TILE - 1, out_shape=")
    with pytest.raises(ValueError, match="logical K extent must be a positive multiple of 16"):
        compile_dataflow_cuda(invalid_k, num_threads=256)

    staged_code = _staged_groupgemm_code()
    assert "__lk_cpp_expr" not in staged_code
    assert "__lk_tmem_C_tmem + PIPE_mma_to_epi_C_tmem.producer_stage() * 256" in staged_code


def _staged_groupgemm_code() -> str:
    return compile_dataflow_cuda(_kernel_source("v2_groupgemm.py"), num_threads=256)


def _groupgemm_source(cluster_shape: tuple[int, int, int]) -> str:
    source = _kernel_source("v2_groupgemm.py")
    old = "cluster_shape=(CLUSTER_SIZE, 1, 1),"
    assert source.count(old) == 1
    return source.replace(old, f"cluster_shape={cluster_shape},")


@_requires_torch_demos
def test_barrier_analysis_requires_validated_cta_group_shape() -> None:
    skip_if_demo_requires_torch("v2_groupgemm.py")
    graph = analyze_dataflow_source(_groupgemm_source((2, 1, 1)))
    graph["cta_group_shape"] = None
    with pytest.raises(ValueError, match="missing compiler-validated cta_group_shape"):
        barrier_participation(graph, graph["pipelines"][0])


@_requires_torch_demos
@pytest.mark.parametrize(
    "shape",
    [(2, 1, 1), (2, 2, 1), (4, 1, 1), (4, 2, 1), (4, 4, 1), (2, 4, 1), (8, 1, 1)],
)
def test_cg2_grouped_gemm_accepts_supported_cluster_shapes(shape: tuple[int, int, int], ) -> None:
    skip_if_demo_requires_torch("v2_groupgemm.py")
    code = compile_dataflow_cuda(_groupgemm_source(shape), num_threads=256)
    assert "cta_group_is_leader<2>()" in code
    assert "tmem_alloc<2>" in code


@_requires_torch_demos
def test_dispatch_ab_transfer_plans_derive_topology_and_barriers() -> None:
    skip_if_demo_requires_torch("v2_dispatch_group_gemm_swiglu_fuse_sm100.py")
    source = _kernel_source("v2_dispatch_group_gemm_swiglu_fuse_sm100.py")
    code = compile_dataflow_cuda(source, num_threads=256)

    assert "tma_load_2d_2sm_mcast_fn(&dA" in code
    assert "tma_load_3d_2sm_fn(&dB" in code
    assert "PipelineTmaUmma<5, 65536, 2, 2, 2, true, 15>" in code
    assert "tma_multicast_mask<2, 2, 1, 2, 1, 1, 1>()" in code
    assert "tma_multicast_is_issuer<2, 2, 1, 2, 1, 1, 1>()" in code
    assert "cluster_rank() ==" not in code
    assert "& 0xFEFFFFFF" in code
    assert "mapa.shared::cluster" not in code


@_requires_torch_demos
def test_extent_one_multicast_plan_lowers_to_plain_tma() -> None:
    skip_if_demo_requires_torch("v2_groupgemm.py")
    source = _groupgemm_source((2, 1, 1))
    source, replacements = re.subn(
        r"(A_plan\s*=\s*g\.tma_transfer\(\s*)mcast_group_dim=None",
        r'\1mcast_group_dim="y"',
        source,
        count=1,
    )
    assert replacements == 1
    code = compile_dataflow_cuda(source, num_threads=256)
    assert "tma_load_2d_2sm_fn(&dA" in code
    assert "tma_load_2d_2sm_mcast_fn(&dA" not in code


@_requires_torch_demos
def test_planned_tma_rejects_multicast_axis_in_base_coord() -> None:
    skip_if_demo_requires_torch("v2_groupgemm.py")
    source = _groupgemm_source((2, 2, 1))
    source = source.replace(
        "mcast_group_dim=None,",
        'mcast_group_dim="y",',
        1,
    )
    source = source.replace(
        "offset_m: ll.int32 = ll.int32(coord_m_ab * PAIR_TILE_M)",
        'offset_m: ll.int32 = ll.int32('
        'coord_m_ab * PAIR_TILE_M + ri.group_coord("y") * PAIR_TILE_M)',
    )
    with pytest.raises(ValueError, match="independent of.*mcast_group_dim"):
        compile_dataflow_cuda(source, num_threads=256)


@_requires_torch_demos
def test_planned_tma_requires_explicit_cta_group_shape() -> None:
    skip_if_demo_requires_torch("v2_groupgemm.py")
    source = _groupgemm_source((2, 1, 1))
    source = source.replace(", cta_group_shape=(2, 1, 1)", "")
    with pytest.raises(ValueError, match="requires explicit cta_group_shape"):
        compile_dataflow_cuda(source, num_threads=256)


@_requires_torch_demos
def test_cta_group_shape_must_divide_cluster_shape() -> None:
    skip_if_demo_requires_torch("v2_groupgemm.py")
    with pytest.raises(ValueError, match="must divide cluster_shape"):
        compile_dataflow_cuda(_groupgemm_source((3, 1, 1)), num_threads=256)


@_requires_torch_demos
def test_cluster_shape_over_16_ctas_is_rejected() -> None:
    skip_if_demo_requires_torch("v2_groupgemm.py")
    with pytest.raises(ValueError, match="16-CTA cluster limit"):
        compile_dataflow_cuda(_groupgemm_source((4, 8, 1)), num_threads=256)


def test_legacy_cg2_role_intrinsics_are_removed() -> None:
    assert not hasattr(ri, "tma_ld_2d_cg2")
    assert not hasattr(ri, "tma_ld_3d_cg2")
    assert not hasattr(ri, "tma_ld_2d_cg2_multicast_peer")
    assert not hasattr(ri, "tma_ld_2d_cg2_multicast")
