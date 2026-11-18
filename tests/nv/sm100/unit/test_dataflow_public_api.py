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
"""RF05/RF06 regressions for the promoted role/dataflow compiler.

These tests are CPU-only and never require Torch or a GPU. They cover:

* the public CPU role/dataflow API and immutable canonical pipeline;
* scalar kernels never triggering role passes;
* typed lowered IR (declared pointer/scalar result types, effects, dump);
* no opaque ``_lk_cpp_*`` producers and no ``int32`` catch-all;
* compiler imports never reading neighboring example files;
* legacy string helpers delegating to the one canonical pipeline;
* the runtime backend header and its explicit CUDA Driver dependency.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from little_kernel.core import ir
from little_kernel.core.ir import IROpEffect
from little_kernel.core.type_system import Pointer, bool_, uint32, uint64
import little_kernel.backends.nv.sm100.passes.dataflow as dataflow
from little_kernel.backends.nv.sm100.passes.dataflow import DATAFLOW_PIPELINE, dataflow_frontend, dataflow_pass_list, tree_has_dataflow

REPO_PYTHON = Path(__file__).resolve().parents[4] / "python"
GEMM_DEMO = REPO_PYTHON.parent / "examples/nv/sm100/gemm_sm100.py"


def _gemm_source() -> str:
    return GEMM_DEMO.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Public API / pipeline immutability
# --------------------------------------------------------------------------- #


def test_pipeline_is_immutable_and_copies_are_independent():
    assert isinstance(DATAFLOW_PIPELINE, tuple)
    assert DATAFLOW_PIPELINE
    for entry in DATAFLOW_PIPELINE:
        assert isinstance(entry, tuple) and len(entry) == 2
        fn, stable_id = entry
        assert callable(fn)
        assert stable_id.startswith("role.")

    first = dataflow_pass_list()
    second = dataflow_pass_list()
    assert first == second
    first.append(object())
    assert len(first) == len(second) + 1  # copies are private


def test_dataflow_frontend_is_noop_for_scalar_kernel():
    scalar = textwrap.dedent("""
        import little_kernel as lk
        import little_kernel.language as ll

        @lk.ll_kernel(backend='cuda', is_entry=True)
        def add(x: ll.int32, y: ll.int32) -> ll.int32:
            return x + y
        """)
    tree = ast.parse(scalar)
    before = ast.dump(tree)
    assert tree_has_dataflow(tree) is False

    ctx: dict = {}
    result = dataflow_frontend(tree, ctx)
    assert ast.dump(result) == before  # untouched
    assert "__LK_DATAFLOW_GRAPHS__" not in ctx


def test_role_declarations_are_supported_public_api():
    from little_kernel.backends.nv.sm100.language.dataflow import HW, Flow, dataflow

    flow = Flow()
    tma = flow.role("tma", name="tma", hw=HW.TMA, warps=[0])
    a = flow.smem("A_smem", (64, 64), dtype="bfloat16", stages=2, major="K")
    mma = flow.role("mma", name="mma", hw=HW.TensorCore, warps=[1])
    key = flow.depends(src=tma, dst=mma, via=[a])
    assert flow.roles["tma"] is tma
    assert flow.memories["A_smem"] is a
    assert len(flow.channels) == 1 and key.id == 0


# --------------------------------------------------------------------------- #
# Typed lowered IR (RF06)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def lowered_gemm():
    source = _gemm_source()
    module, ctx, _kernel = dataflow.lower_dataflow_source(source, {"M": 1024, "N": 1024, "K": 1024, "num_ctas": 128})
    return module, ctx


def test_gemm_lowered_tree_has_only_typed_nodes(lowered_gemm):
    module, _ctx = lowered_gemm
    nodes = list(ir.iter_lowered_nodes(module))
    assert nodes

    legacy = [n for n in ast.walk(module) if ir.is_legacy_cpp_node(n)]
    assert legacy == []  # no _lk_cpp_* producers

    errors = ir.verify_lowered_ir(module, raise_on_error=False)
    assert errors == []

    for node in nodes:
        if isinstance(node, ir.TargetExpression):
            assert node.result_type is not None
            assert not isinstance(node.result_type, type(None))


def test_pointer_target_expressions_keep_pointer_types(lowered_gemm):
    module, _ctx = lowered_gemm
    exprs = [n for n in ir.iter_lowered_nodes(module) if isinstance(n, ir.TargetExpression)]
    pointer_exprs = [n for n in exprs if isinstance(n.result_type, Pointer)]
    assert pointer_exprs  # barrier handles / SMEM / TMEM addresses
    for node in pointer_exprs:
        assert node.result_type.inner_type is not None

    barrier_handles = [n for n in exprs if "get_barrier" in n.spelling]
    assert barrier_handles
    for node in barrier_handles:
        assert node.result_type == Pointer(uint64)


def test_target_type_inference_has_no_int32_catchall():
    from little_kernel.backends.nv.sm100.codegen.dataflow.target_ops import infer_target_type

    assert infer_target_type("PIPE.producer_get_barrier()") == Pointer(uint64)
    assert infer_target_type("__lk_tmem_C") == uint32
    assert infer_target_type("little_kernel::role_pipeline::group_coord<1,1,1,0>()") == uint32
    assert infer_target_type("little_kernel::role_pipeline::cta_group_is_leader<2>()") == bool_
    with pytest.raises(TypeError, match="Unknown target expression"):
        infer_target_type("unknown_backend_call()")


def test_dataflow_intrinsics_do_not_shadow_native_registry():
    import little_kernel.backends.nv.sm100.language.intrin.umma as native
    import little_kernel.backends.nv.common.language.intrin.memory as memory
    from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as logical
    from little_kernel.language.builtin_base import get_intrin_ctx, MEMORY_EFFECTS_ATTR
    registry = get_intrin_ctx()
    assert logical.umma_f16_ss_2sm is not native.umma_f16_ss_2sm
    assert registry["umma_f16_ss_2sm"] is native.umma_f16_ss_2sm
    assert registry["dataflow.umma_f16_ss_2sm"] is logical.umma_f16_ss_2sm
    assert logical.gmem_ld_f32x4 is memory.gmem_ld_f32x4
    assert getattr(logical.gmem_ld_f32x4, MEMORY_EFFECTS_ATTR)


def test_declared_smem_element_type_and_encoded_tmem_address(lowered_gemm):
    from little_kernel.core.type_system import bfloat16
    nodes = list(ir.iter_lowered_nodes(lowered_gemm[0]))
    resources = [n for n in nodes if isinstance(n, ir.ResourceReference)]
    assert any(n.result_type == Pointer(bfloat16) for n in resources)
    assert any(isinstance(n, ir.StageReference) and n.result_type == Pointer(bfloat16) for n in nodes)
    from little_kernel.backends.nv.sm100.codegen.dataflow.target_ops import cpp_raw_expr
    assert cpp_raw_expr("__lk_tmem_acc").result_type == uint32


def test_public_kernel_and_string_helper_use_identical_pipeline():
    from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
    from little_kernel.core.passes import PASSES
    from examples.nv.sm100.gemm_sm100 import gemm_sm100_kernel
    source = Path(gemm_sm100_kernel.py_func.__code__.co_filename).read_text()
    emitted = gemm_sm100_kernel.compile(PASSES["cuda"], codegen_cuda, num_threads=256)
    reference = dataflow.compile_dataflow_cuda(source, num_threads=256)
    assert emitted == reference.split("\n", 1)[1]


def test_target_statement_effects_are_preserved(lowered_gemm):
    module, _ctx = lowered_gemm
    stmts = [n for n in ir.iter_lowered_nodes(module) if isinstance(n, ir.TargetStatement)]
    barriers = [n for n in stmts if n.effect is IROpEffect.BARRIER]
    assert barriers  # __syncthreads / cluster_sync / transactions retained
    for node in stmts:
        assert node.effect is not IROpEffect.PURE
        assert node.section in ("main", "builtin")


def test_typed_dump_is_explicit(lowered_gemm):
    module, _ctx = lowered_gemm
    node = next(iter(ir.iter_lowered_nodes(module)))
    dumped = ast.dump(node)
    assert "result_type" in dumped or "effect" in dumped
    assert "_lk_cpp" not in dumped


# --------------------------------------------------------------------------- #
# No example reads at import time
# --------------------------------------------------------------------------- #


def test_compiler_imports_do_not_read_example_sources(tmp_path):
    import os
    code = """
import sys
def forbid_assets(event, args):
    if event == 'open' and args and isinstance(args[0], str):
        if '/examples/' in args[0] or '/assets/' in args[0] or '/benchmarks/' in args[0]:
            raise AssertionError('Compiler read repository asset: ' + args[0])
sys.addaudithook(forbid_assets)
import little_kernel.backends.nv.sm100.passes.dataflow
import little_kernel.backends.nv.sm100.codegen.dataflow.lowering
print('clean')
"""
    proc = subprocess.run(
        ["uv", "run", "--locked", "--no-sync", "--project",
         str(REPO_PYTHON.parent), "python", "-c", code], cwd=tmp_path, capture_output=True, text=True,
        env=dict(os.environ, PYTHONPATH=str(REPO_PYTHON)))
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout


def test_retired_prototype_namespace_has_no_directory_or_import():
    import importlib.util
    assert not (REPO_PYTHON / "little_kernel/role_proposal").exists()
    assert importlib.util.find_spec("little_kernel.role_proposal") is None


# --------------------------------------------------------------------------- #
# One canonical pipeline + public CPU compile
# --------------------------------------------------------------------------- #


def test_dataflow_implementations_have_explicit_owners():
    from little_kernel.backends.nv.sm100.passes.dataflow.normalization import compile_time_branch_prune_pass
    from little_kernel.backends.nv.sm100.codegen.dataflow.lowering import role_materialize_pass
    from little_kernel.backends.nv.sm100.core.pipeline import PipelineSpec
    assert dataflow_frontend.__module__ == "little_kernel.backends.nv.sm100.passes.dataflow.pipeline"
    assert compile_time_branch_prune_pass.__module__ == "little_kernel.backends.nv.sm100.passes.dataflow.normalization"
    assert role_materialize_pass.__module__ == "little_kernel.backends.nv.sm100.codegen.dataflow.lowering"
    assert PipelineSpec.__module__ == "little_kernel.backends.nv.sm100.core.pipeline"


def test_role_gemm_compiles_via_public_cpu_api():
    source = _gemm_source()
    cuda = dataflow.compile_dataflow_cuda(source, {"M": 1024, "N": 1024, "K": 1024, "num_ctas": 128})

    # Real role CUDA semantics are preserved.
    assert "__global__" in cuda
    assert "role_pipeline.cuh" in cuda
    assert "PIPE_tma_to_mma" in cuda
    assert "tcgen05" in cuda
    assert "SharedStorage" in cuda
    # Backend text is not hidden on ordinary AST identifiers.
    assert "__lk_cpp_expr" not in cuda


# --------------------------------------------------------------------------- #
# Runtime backend header resource
# --------------------------------------------------------------------------- #


def test_runtime_header_states_cuda_driver_dependency():
    header = (REPO_PYTHON / "little_kernel/backends/nv/sm100/runtime/include" / "role_pipeline.cuh")
    assert header.exists()
    text = header.read_text(encoding="utf-8")
    # Explicit Driver dependency for NVCC 13.0 standalone compilation.
    assert "#include <cuda.h>" in text
    assert "CUtensorMap" in text


def test_emitted_include_points_at_runtime_owner(lowered_gemm):
    module, _ctx = lowered_gemm
    includes = [n.include for n in ir.iter_lowered_nodes(module) if isinstance(n, ir.TargetInclude)]
    assert includes == ['"little_kernel/backends/nv/sm100/runtime/include/role_pipeline.cuh"']
    # The old prototype include spelling is gone.
    assert not any("role_proposal" in inc for inc in includes)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
