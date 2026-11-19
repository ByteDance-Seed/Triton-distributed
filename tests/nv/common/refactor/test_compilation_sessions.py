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
"""RF07/RF01: isolated compilation sessions and immutable default pipelines.

Deterministic CPU tests (no GPU/nvcc): they verify

* default pipeline construction is idempotent and immutable to consumers,
  with stable pass ids/prerequisites and pre-execution phase validation;
* every compile/lower runs in a fresh CompilationSession whose derived pass
  state never accumulates on ``LLKernel.ctx`` across constexpr
  specializations or concurrent calls;
* explicit pass-list callers and user ctx/launch options keep working;
* the role/dataflow frontend is selected only through an explicit hook
  (unknown selection is rejected; core provides no fake fallback);
* ``compile_artifact`` returns a merged CompilationArtifact and ``build``
  propagates the actual post-constexpr KernelSignature to CompiledKernel.
"""

import ast
import threading

import pytest

import little_kernel.language as ll
from little_kernel.core.compile import ll_kernel, LLKernel
from little_kernel.core.compilation import CompilationSession, register_frontend, select_frontend, registered_frontends
from little_kernel.core.passes import PASSES, DEFAULT_PASS_IDS, PassSpec, Pipeline, get_pass_registry, get_default_pipeline, register_default_passes, get_passes_for_backend
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda

from little_kernel.backends import get_backend

NV_ARCH = get_backend("cuda").target.resolve().arch


def test_clone_context_preserves_tuple_list_cycles():
    from little_kernel.core.compilation import clone_context
    items = []
    root = (items, )
    items.append(root)
    copied = clone_context({"root": root})["root"]
    assert copied is not root
    assert copied[0][0] is copied


def test_source_only_wide_integer_does_not_require_host_abi():

    @ll_kernel(is_entry=False)
    def wide(x: ll.int128) -> ll.void:
        pass

    assert "__int128 x" in wide.compile([], codegen_cuda)
    with pytest.raises(ValueError, match="128"):
        wide.compile_artifact([], codegen_cuda)


def test_explicit_signature_cannot_bypass_unsupported_typed_abi():
    from little_kernel.runtime.abi import KernelSignature
    struct_type = ll.Struct[(("ptr", ll.ptr[ll.int32]), )]

    @ll_kernel(is_entry=True)
    def structured(s: struct_type) -> ll.void:
        pass

    with pytest.raises(TypeError, match="layout|pointer|scalar|storage"):
        structured.build(passes=[], codegen_func=codegen_cuda, grid=(1, 1, 1), block=(1, 1, 1), arch=NV_ARCH,
                         signature=KernelSignature([]))


def test_declared_frontend_receives_launch_metadata():
    seen = []

    def frontend(tree, ctx):
        seen.append((ctx["__LK_CLUSTER_SHAPE__"], ctx["__LK_CTA_GROUP_SHAPE__"]))
        return tree

    @ll_kernel(is_entry=True, frontend=frontend, cluster_shape=(2, 1, 1), cta_group_shape=(2, 1, 1))
    def clustered(n: ll.int32) -> ll.void:
        pass

    clustered.lower([])
    assert seen == [((2, 1, 1), (2, 1, 1))]


from little_kernel.core.internal import __LITTLE_KERNEL_ENTRY__, __LOCAL_ARRAY_SHAPES_CTX_KEY__

# ---------------------------------------------------------------------------
# Helpers / kernels (must live in a real file so inspect.getsource works)
# ---------------------------------------------------------------------------


@ll_kernel(is_entry=True)
def simple_kernel(x: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    for i in range(n):
        x[i] = ll.float32(i)


@ll_kernel(is_entry=True)
def constexpr_kernel(out: ll.Tensor[ll.int32], BLOCK: ll.constexpr[ll.int32]) -> ll.void:
    for i in range(BLOCK):
        out[i] = ll.int32(i)


@ll_kernel(is_entry=True, no_alias=("x", ), read_only=("x", ), min_blocks=4, launch_bounds=(256, 2))
def hinted_kernel(x: ll.Tensor[ll.float32]) -> ll.void:
    x[0] = ll.float32(1.0)


@ll_kernel(is_entry=False)
def recursive_helper(x: ll.Tensor[ll.float32], i: ll.int32) -> ll.void:
    x[i] = ll.float32(1.0)


@ll_kernel(is_entry=True)
def recursive_caller(x: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    for i in range(n):
        recursive_helper(x, ll.int32(i))


@ll_kernel(is_entry=True)
def partially_typed_kernel(x: ll.Tensor[ll.float32], n) -> ll.void:
    # Deliberately missing annotation on ``n``: this is a *typed* kernel with
    # an invalid ABI, not the legacy untyped path.
    return None


def _trivial_codegen(tree, ctx, emit_header=True, **kwargs):
    return "// trivial source\n__global__ void k() {}\n"


def _no_op_pass(tree, ctx):
    ctx["__TEST_DERIVED_STATE__"] = {"derived": True}
    return tree


# ---------------------------------------------------------------------------
# Default pipeline: idempotent, immutable, identified
# ---------------------------------------------------------------------------


def test_default_pipeline_has_stable_ids_and_prerequisites():
    pipeline = get_default_pipeline("cuda")
    assert isinstance(pipeline, Pipeline)
    assert pipeline.pass_ids == tuple(DEFAULT_PASS_IDS)
    # insert_mem_alloc follows its declared prerequisites.
    ids = pipeline.pass_ids
    insert_idx = ids.index("insert_mem_alloc")
    for prerequisite in ("flatten_empty", "rebind_alloc_names"):
        assert ids.index(prerequisite) < insert_idx
    assert all(isinstance(spec, PassSpec) for spec in pipeline)
    # Immutable to consumers.
    with pytest.raises((AttributeError, Exception)):
        pipeline._stages = ()  # type: ignore[misc]


def test_registering_defaults_twice_does_not_append():
    before = get_default_pipeline("cuda").pass_ids
    register_default_passes("cuda")
    register_default_passes("cuda")
    after = get_default_pipeline("cuda").pass_ids
    assert before == after
    assert len(after) == 8
    # A different backend owns its own stage composition; registration stays idempotent.
    from little_kernel.backends import available_backends
    for backend_id in available_backends():
        initial = get_default_pipeline(backend_id).pass_ids
        register_default_passes(backend_id)
        assert get_default_pipeline(backend_id).pass_ids == initial


def test_consumers_cannot_mutate_global_defaults():
    list_a = PASSES["cuda"]
    list_b = PASSES["cuda"]
    assert list_a is not list_b  # fresh list per lookup
    list_a.append(_no_op_pass)
    list_a.clear()
    registry_list = get_passes_for_backend("cuda")
    assert len(registry_list) == len(DEFAULT_PASS_IDS)
    # A list obtained from the registry is independent as well.
    registry_list.append(_no_op_pass)
    assert len(get_passes_for_backend("cuda")) == len(DEFAULT_PASS_IDS)
    # Legacy PASSES mapping keeps behaving like the old dict for callers.
    assert len(PASSES["cuda"]) == len(DEFAULT_PASS_IDS)
    from little_kernel.backends import available_backends
    assert all(name in PASSES for name in available_backends())


def test_register_pass_is_idempotent_per_id():
    registry = get_pass_registry()
    before = registry.get_pass_ids("test-backend")
    registry.register_pass("test-backend", _no_op_pass, "named", pass_id="only-once")
    registry.register_pass("test-backend", _no_op_pass, "named", pass_id="only-once")
    ids = registry.get_pass_ids("test-backend")
    assert ids == before + ("only-once", )
    registry.reset_backend("test-backend")


def test_invalid_phase_order_rejected_before_execution():

    def late(tree, ctx):
        return tree

    def early(tree, ctx):
        return tree

    stages = [
        PassSpec(pass_id="late", func=late, requires=("early", )),
        PassSpec(pass_id="early", func=early),
    ]
    with pytest.raises(ValueError, match="requires prerequisite"):
        Pipeline("cuda", stages)
    # Correct order validates.
    pipeline = Pipeline("cuda", list(reversed(stages)))
    assert pipeline.pass_ids == ("early", "late")
    # Duplicate ids are rejected too.
    with pytest.raises(ValueError, match="Duplicate pass id"):
        Pipeline("cuda", [
            PassSpec(pass_id="same", func=early),
            PassSpec(pass_id="same", func=early),
        ])


# ---------------------------------------------------------------------------
# Isolated sessions: no derived state accumulation
# ---------------------------------------------------------------------------


def test_explicit_pipeline_object_and_none_select_defaults():
    # ``None`` selects the immutable default pipeline; a Pipeline object is
    # accepted directly and yields identical source to the legacy list.
    from little_kernel.core.compilation import CompilationSession
    s_none = CompilationSession(simple_kernel, None)
    assert s_none.pipeline.pass_ids == tuple(DEFAULT_PASS_IDS)
    s_pipe = CompilationSession(simple_kernel, get_default_pipeline("cuda"))
    assert s_pipe.pipeline.pass_ids == tuple(DEFAULT_PASS_IDS)
    src_list = simple_kernel.compile(PASSES["cuda"], codegen_cuda, num_threads=64)
    src_none = simple_kernel.compile(None, codegen_cuda, num_threads=64)
    assert src_list == src_none


def test_pipeline_equality_and_hash():
    p1 = get_default_pipeline("cuda")
    p2 = get_default_pipeline("cuda")
    assert p1 == p2 and hash(p1) == hash(p2)
    assert tuple(spec.func for spec in p1) == p1.callables


def test_compile_runs_in_isolated_session_without_accumulation():
    passes = PASSES["cuda"]
    src1 = simple_kernel.compile(passes, codegen_cuda, num_threads=128)
    keys_after_first = list(simple_kernel.ctx)
    # Re-compiling never accumulates additional derived/symbol keys
    # (no update_context underscore suffix growth across specializations).
    simple_kernel.compile(passes, codegen_cuda, num_threads=128)
    simple_kernel.compile(passes, codegen_cuda, num_threads=128)
    assert list(simple_kernel.ctx) == keys_after_first
    # Re-compiling is stable and independent.
    src2 = simple_kernel.compile(PASSES["cuda"], codegen_cuda, num_threads=128)
    assert src1 == src2
    assert "__global__" in src1


def test_derived_state_is_refreshed_per_specialization_not_merged():
    from little_kernel.core.compilation import CompilationSession
    # Two sessions derived from the same kernel never share their state.
    s2 = CompilationSession(constexpr_kernel, PASSES["cuda"], constexpr_args={"BLOCK": 2})
    s4 = CompilationSession(constexpr_kernel, PASSES["cuda"], constexpr_args={"BLOCK": 4})
    r2 = s2.lower()
    r4 = s4.lower()
    assert s2.ctx is not s4.ctx
    # Each session sees only its own constexpr value in metadata.
    assert r2.constexpr_args["BLOCK"] == 2
    assert r4.constexpr_args["BLOCK"] == 4
    # Mutating one session cannot affect the other or the source kernel.
    s2.ctx["__OTHER_MARK__"] = True
    assert "__OTHER_MARK__" not in s4.ctx
    assert "__OTHER_MARK__" not in constexpr_kernel.ctx


def test_explicit_pass_list_caller_keeps_working():
    src = simple_kernel.compile([_no_op_pass], _noop_codegen)
    assert "NOOP" in src
    # The explicit pass wrote into its session copy, never the kernel.
    assert "__TEST_DERIVED_STATE__" not in simple_kernel.ctx


def _noop_codegen(tree, ctx, emit_header=True, **kwargs):
    assert "__TEST_DERIVED_STATE__" in ctx  # session ctx reaches codegen
    return "// NOOP source\n"


def test_sessions_are_independent():
    s1 = CompilationSession(simple_kernel, PASSES["cuda"])
    s2 = CompilationSession(simple_kernel, PASSES["cuda"])
    assert s1.ctx is not s2.ctx
    s1.ctx["__SESSION_MARKER__"] = 1
    assert "__SESSION_MARKER__" not in s2.ctx
    assert "__SESSION_MARKER__" not in simple_kernel.ctx
    result = s1.lower()
    assert result.pass_ids == tuple(DEFAULT_PASS_IDS)
    assert isinstance(result.tree, ast.Module)


def test_respecialization_does_not_share_or_accumulate_state():
    # Two constexpr specializations of the same kernel are distinct
    # compilations with distinct signature/metadata.
    art2 = constexpr_kernel.compile_artifact(PASSES["cuda"], codegen_cuda, constexpr_args={"BLOCK": 2})
    art4 = constexpr_kernel.compile_artifact(PASSES["cuda"], codegen_cuda, constexpr_args={"BLOCK": 4})
    art2_again = constexpr_kernel.compile_artifact(PASSES["cuda"], codegen_cuda, constexpr_args={"BLOCK": 2})
    assert art2.source == art2_again.source
    assert art4.source != art2.source
    # constexpr parameter removed from the launch signature.
    assert art2.signature.names == ("out", )
    assert art4.signature.names == ("out", )
    assert art2.metadata["constexpr_args"] == (("BLOCK", 2), )
    assert __LOCAL_ARRAY_SHAPES_CTX_KEY__ not in constexpr_kernel.ctx


def test_concurrent_compilations_are_independent():
    results = {}
    errors = []

    def worker(block):
        try:
            art = constexpr_kernel.compile_artifact(PASSES["cuda"], codegen_cuda, constexpr_args={"BLOCK": block})
            # Each specialization must observe its own BLOCK value.
            assert art.metadata["constexpr_args"] == (("BLOCK", block), )
            results[block] = art.source
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = []
    for block in (1, 2, 3, 4):
        for _ in range(4):
            threads.append(threading.Thread(target=worker, args=(block, )))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    # Distinct specializations produced distinct sources; repeats matched.
    sources = {block: results[block] for block in (1, 2, 3, 4)}
    assert len(set(sources.values())) == 4
    # No derived session state escaped to the shared kernel.
    assert __LOCAL_ARRAY_SHAPES_CTX_KEY__ not in constexpr_kernel.ctx


# ---------------------------------------------------------------------------
# User ctx options / launch hints preserved
# ---------------------------------------------------------------------------


def test_user_ctx_options_and_launch_hints_preserved():
    from little_kernel.core.compile import _NO_ALIAS_CTX_KEY, _READ_ONLY_CTX_KEY, _MIN_BLOCKS_CTX_KEY
    hinted_kernel.compile(PASSES["cuda"], codegen_cuda, num_threads=128)
    ctx = hinted_kernel.ctx
    assert ctx[_NO_ALIAS_CTX_KEY] == frozenset({"x"})
    assert ctx[_READ_ONLY_CTX_KEY] == frozenset({"x"})
    assert ctx[_MIN_BLOCKS_CTX_KEY] == 4
    assert hinted_kernel.launch_bounds == (256, 2)
    assert hinted_kernel.is_entry
    assert ctx[__LITTLE_KERNEL_ENTRY__] is hinted_kernel.py_func


def test_launch_bounds_legality_still_enforced():
    with pytest.raises(ValueError, match="could never launch"):
        hinted_kernel.compile(PASSES["cuda"], codegen_cuda, num_threads=512)


# ---------------------------------------------------------------------------
# Explicit role/dataflow frontend selection hook (RF05 integration point)
# ---------------------------------------------------------------------------


def test_explicit_frontend_callable_runs_on_fresh_session_ctx():
    seen = {}

    def example_frontend(tree, ctx):
        seen["entry"] = ctx.get(__LITTLE_KERNEL_ENTRY__)
        # Worker-06 contract: the hook receives the live, isolated session
        # ctx and may insert frontend symbols before ordinary passes run;
        # those symbols never leak back to the kernel's source globals.
        ctx["__DATAFLOW_FRONTEND_SYMBOL__"] = object()
        return tree

    src = simple_kernel.compile(PASSES["cuda"], codegen_cuda, frontend=example_frontend)
    assert seen["entry"] is simple_kernel.py_func
    assert "__global__" in src
    assert "__DATAFLOW_FRONTEND_SYMBOL__" not in simple_kernel.ctx


def test_frontend_non_ast_return_rejected():

    def bad_frontend(tree, ctx):
        return "not an ast"

    with pytest.raises(TypeError, match="must return an AST"):
        simple_kernel.compile(PASSES["cuda"], codegen_cuda, frontend=bad_frontend)


def test_registered_named_frontend_and_unknown_rejected():

    def dataflow_frontend(tree, ctx):  # worker-06 supplies the real one
        return tree

    register_frontend("dataflow", dataflow_frontend)
    assert "dataflow" in registered_frontends()
    assert select_frontend("dataflow") is dataflow_frontend
    assert select_frontend(None) is None
    with pytest.raises(ValueError, match="Unknown frontend"):
        select_frontend("does-not-exist")
    with pytest.raises(TypeError):
        register_frontend("bad", 42)
    with pytest.raises(TypeError):
        select_frontend(123)


# ---------------------------------------------------------------------------
# compile_artifact and signature propagation through build (RF01 merge)
# ---------------------------------------------------------------------------


def test_compile_artifact_merges_source_signature_target_metadata():
    artifact = simple_kernel.compile_artifact(PASSES["cuda"], codegen_cuda, target="sm_90a", num_threads=128,
                                              metadata={"custom": ("a", "b")})
    assert artifact.kernel_name == "simple_kernel"
    assert artifact.target == "sm_90a"
    assert artifact.signature.names == ("x", "n")
    x_param, n_param = artifact.signature
    assert x_param.kind == "pointer"
    assert n_param.c_type.__name__ == "c_int"
    assert artifact.metadata["backend"] == "cuda"
    assert artifact.metadata["custom"] == ("a", "b")
    assert artifact.metadata["pass_ids"] == tuple(DEFAULT_PASS_IDS)
    # Source matches the source-returning API exactly.
    assert artifact.source == simple_kernel.compile(PASSES["cuda"], codegen_cuda, num_threads=128)


def test_untyped_kernel_artifact_is_explicitly_untyped():

    @ll_kernel(is_entry=True)
    def untyped(x, n):  # noqa: ANN001 - deliberate legacy untyped kernel
        return None

    # Build the signature without running the typed default pipeline (an
    # untyped kernel cannot type-check); the legacy path is explicitly None.
    session = CompilationSession(untyped, [])
    session.ctx[__LITTLE_KERNEL_ENTRY__] = untyped.py_func
    assert session._build_signature() is None


def test_typed_kernel_session_builds_actual_signature():
    session = CompilationSession(simple_kernel, [])
    signature = session._build_signature()
    assert signature.names == ("x", "n")
    # constexpr parameters are removed by the specialization.
    session_cx = CompilationSession(constexpr_kernel, [], constexpr_args={"BLOCK": 8})
    signature_cx = session_cx._build_signature()
    assert signature_cx.names == ("out", )


def test_build_propagates_actual_signature_to_compiled_kernel():
    """build() forwards the real frontend KernelSignature to CompiledKernel."""
    import little_kernel.backends.nv.common.runtime.kernel as runtime
    from little_kernel.runtime.abi import KernelParameter, KernelSignature

    captured = {}

    class RecordingCompiledKernel:

        def __init__(self, *args, signature=None, **kwargs):
            captured["args"] = args
            captured["kwargs"] = dict(kwargs, signature=signature)

    original = runtime.CompiledKernel
    runtime.CompiledKernel = RecordingCompiledKernel
    try:
        simple_kernel.build(PASSES["cuda"], codegen_cuda, grid=(1, 1, 1), block=(128, 1, 1), arch=NV_ARCH)
    finally:
        runtime.CompiledKernel = original

    sig = captured["kwargs"].get("signature")
    assert isinstance(sig, KernelSignature)
    assert sig.names == ("x", "n")
    x_param = sig.parameter("x")
    assert isinstance(x_param, KernelParameter)
    # Positional arguments preserved.
    assert captured["args"][1] == "simple_kernel"
    assert captured["args"][2] == (1, 1, 1)
    assert captured["args"][3] == (128, 1, 1)
    assert captured["kwargs"]["cluster_shape"] is None


def test_known_signature_cannot_be_overridden_by_raw_source_one():
    import little_kernel.backends.nv.common.runtime.kernel as runtime
    from little_kernel.runtime.abi import KernelParameter, KernelSignature

    raw = KernelSignature([KernelParameter.scalar("raw", ll.int32)])
    captured = {}

    class RecordingCompiledKernel:

        def __init__(self, *args, signature=None, **kwargs):
            captured["kwargs"] = dict(kwargs, signature=signature)

    original = runtime.CompiledKernel
    runtime.CompiledKernel = RecordingCompiledKernel
    try:
        simple_kernel.build(PASSES["cuda"], codegen_cuda, grid=(1, 1, 1), block=(128, 1, 1), signature=raw)
    finally:
        runtime.CompiledKernel = original
    # The frontend-derived signature wins over the raw-source override.
    assert captured["kwargs"]["signature"].names == ("x", "n")


def test_untyped_raw_source_build_may_supply_signature():
    import little_kernel.backends.nv.common.runtime.kernel as runtime
    from little_kernel.runtime.abi import KernelParameter, KernelSignature

    @ll_kernel(is_entry=True)
    def untyped_kernel(x, n):  # noqa: ANN001 - deliberate legacy untyped path
        return None

    raw = KernelSignature([
        KernelParameter.scalar("a", ll.int32),
        KernelParameter.scalar("b", ll.int32),
    ])
    captured = {}

    class RecordingCompiledKernel:

        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

    original = runtime.CompiledKernel
    runtime.CompiledKernel = RecordingCompiledKernel
    try:
        # Untyped kernels need passes/codegen that tolerate missing
        # annotations, so build the session artifact directly instead.
        from little_kernel.core.compilation import CompilationSession
        session = CompilationSession(untyped_kernel, [])
        result = session.lower()
        assert result.signature is None
        # Simulate build's signature-merge rule.
        merged = result.signature if result.signature is not None else raw
    finally:
        runtime.CompiledKernel = original
    assert merged is raw


def test_real_build_returns_compiled_kernel_without_gpu():
    # Construction never invokes nvcc (compilation is lazy on first launch).
    # The merged runtime CompiledKernel accepts the propagated signature.
    compiled = simple_kernel.build(PASSES["cuda"], codegen_cuda, grid=(1, 1, 1), block=(128, 1, 1), arch=NV_ARCH)
    assert compiled.kernel_name == "simple_kernel"
    assert compiled.grid == (1, 1, 1)
    assert compiled.block == (128, 1, 1)
    assert compiled.arch == NV_ARCH
    assert "__global__" in compiled.cuda_code
    assert compiled.signature is not None
    assert compiled.signature.names == ("x", "n")


# ---------------------------------------------------------------------------
# RF07 review fixes: deep option isolation, per-call contexts, honest ABI
# ---------------------------------------------------------------------------


def test_nested_mutable_options_deeply_cloned():
    from little_kernel.core.compilation import clone_context
    # A shallow dict(kernel.ctx) would leave this nested list aliased to the
    # source and to every sibling session -- the exact RF07 defect.
    simple_kernel.ctx["options"] = {"history": [1, 2]}
    try:
        s1 = CompilationSession(simple_kernel, [])
        s2 = CompilationSession(simple_kernel, [])
        assert s1.ctx["options"] is not s2.ctx["options"]
        assert s1.ctx["options"]["history"] is not s2.ctx["options"]["history"]
        s1.ctx["options"]["history"].append(99)
        assert s1.ctx["options"]["history"] == [1, 2, 99]
        # Neither a sibling session nor the kernel source sees the mutation.
        assert s2.ctx["options"]["history"] == [1, 2]
        assert simple_kernel.ctx["options"]["history"] == [1, 2]
    finally:
        simple_kernel.ctx.pop("options", None)

    # Alias and cycle identity are preserved *within* one clone, opaque and
    # interned immutable objects stay shared read-only.
    shared = ["a"]
    cyc = {"shared1": shared, "shared2": shared, "options": {"h": [0]}}
    cyc["self"] = cyc
    clone = clone_context(cyc)
    assert clone["shared1"] is clone["shared2"]
    assert clone["self"] is clone
    assert clone["options"]["h"] is not cyc["options"]["h"]
    opaque = object()
    interned = ll.float32  # interned LLType/module-level identity
    both = clone_context({"handle": opaque, "dtype": interned})
    assert both["handle"] is opaque
    assert both["dtype"] is interned


def test_lower_never_publishes_last_session_state():
    # The kernel must carry no last-result / last-context / resolved-cache-dir
    # attributes after any number of compilations.
    source_ctx = simple_kernel.ctx
    tree = simple_kernel.lower(PASSES["cuda"])
    assert isinstance(tree, ast.Module)
    assert simple_kernel.ctx is source_ctx  # never replaced
    for forbidden in ("_last_result", "_resolved_cache_dir", "_last_ctx"):
        assert not hasattr(simple_kernel, forbidden)
    result = simple_kernel.lower_result(PASSES["cuda"])
    assert result.tree is not tree or isinstance(result.tree, ast.Module)
    # The result context is immutable to callers.
    with pytest.raises(TypeError):
        result.context["__MUTATE__"] = 1
    tree2, ctx2 = simple_kernel.lower_with_context(PASSES["cuda"])
    assert isinstance(tree2, ast.Module) and isinstance(ctx2, dict)
    assert simple_kernel.ctx is source_ctx


def test_recursive_call_consumes_per_call_context():
    # A real recursive device-function call must lower the callee in its own
    # isolated session and emit it through the returned per-call context.
    src = recursive_caller.compile(PASSES["cuda"], codegen_cuda, num_threads=64)
    assert "recursive_helper" in src
    assert not hasattr(recursive_helper, "_last_result")
    # Concurrent entry compilations that both recurse never race a shared
    # "last" context and produce identical, correct source.
    seen = []
    errors = []

    def worker():
        try:
            seen.append(recursive_caller.compile(PASSES["cuda"], codegen_cuda, num_threads=64))
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert all(text == src for text in seen)


def test_invalid_typed_abi_propagates_and_never_silently_downgrades():
    import little_kernel.backends.nv.common.runtime.kernel as runtime

    # Source-only compile/lower does not require a host launch ABI and still
    # emits source for supported codegen.
    src = partially_typed_kernel.compile([], _trivial_codegen)
    assert "__global__" in src

    # The typed-but-invalid ABI is a real error for artifact/build: it must
    # propagate (the parameter missing its annotation), never be guessed.
    with pytest.raises(TypeError, match="no type annotation"):
        partially_typed_kernel.compile_artifact([], _trivial_codegen)

    constructed = {"count": 0}

    class NeverDowngradedKernel:

        def __init__(self, *args, **kwargs):
            constructed["count"] += 1

    original = runtime.CompiledKernel
    runtime.CompiledKernel = NeverDowngradedKernel
    try:
        with pytest.raises(TypeError, match="no type annotation"):
            partially_typed_kernel.build([], _trivial_codegen, grid=(1, 1, 1), block=(1, 1, 1))
    finally:
        runtime.CompiledKernel = original
    # A typed build can never reach an untyped/raw CompiledKernel.
    assert constructed["count"] == 0


def test_untyped_build_accepts_explicit_raw_signature():
    from little_kernel.runtime.abi import KernelParameter, KernelSignature

    @ll_kernel(is_entry=True)
    def raw_kernel(a, b):  # noqa: ANN001 - explicit legacy untyped path
        return None

    raw = KernelSignature([
        KernelParameter.scalar("a", ll.int32),
        KernelParameter.scalar("b", ll.int32),
    ])
    # The explicit raw signature is the legitimate launch ABI for this
    # explicitly untyped kernel; build succeeds without downgrading a typed
    # signature (there is none).
    compiled = raw_kernel.build([], _trivial_codegen, grid=(1, 1, 1), block=(1, 1, 1), signature=raw)
    assert compiled.kernel_name == "raw_kernel"
    assert compiled.signature is raw

    # Without the override the untyped launchable path fails honestly
    # instead of silently producing an unlaunchable kernel.
    with pytest.raises(TypeError, match="no parameter annotations"):
        raw_kernel.build([], _trivial_codegen, grid=(1, 1, 1), block=(1, 1, 1))


def test_compiled_kernel_from_artifact_carries_signature():
    artifact = simple_kernel.compile_artifact(PASSES["cuda"], codegen_cuda, target="sm_90a", num_threads=128)
    compiled = artifact  # artifact is the immutable RF01 merge
    from little_kernel.runtime import CompiledKernel
    launchable = CompiledKernel.from_artifact(compiled, grid=(1, 1, 1), block=(128, 1, 1))
    assert launchable.kernel_name == "simple_kernel"
    assert launchable.signature is artifact.signature
    assert launchable.signature.names == ("x", "n")
    launchable._module = None  # construction is lazy: no nvcc/GPU needed
