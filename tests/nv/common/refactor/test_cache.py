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
"""RF02 regressions: immutable build identity and atomic cache publication.

Cache/ownership checks use deterministic fakes. One NVCC-gated compilation
also verifies that relative user include paths survive the build directory.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import little_kernel.backends.nv.common.runtime.kernel as kernel_mod
from little_kernel.core.artifact import CompilationArtifact
import little_kernel.backends.nv.common.runtime.cache as cache_mod
from little_kernel.runtime.abi import KernelParameter, KernelSignature
from little_kernel.backends.nv.common.runtime.cache import AtomicCubinCache, BuildIdentity, atomic_publish, compute_build_identity, default_toolchain_identity, normalize_options, scan_user_dependencies
from little_kernel.backends.nv.common.runtime.kernel import CompiledKernel
from little_kernel.backends.nv.common.runtime.target import CudaTarget

SOURCE = """
#include "kernel_header.h"
__global__ void k(int* out) { *out = MODE; }
""".strip()

# -----------------------------------------------------------------------------
# Identity completeness
# -----------------------------------------------------------------------------


def _identity(options=(), toolchain="nvcc:fake", headers=None, include_paths=None, source=SOURCE, target="sm_90"):
    return compute_build_identity(
        kernel_name="k",
        source=source,
        target_arch=target,
        options=list(options),
        include_paths=include_paths,
        extra_dependencies=headers or (),
        toolchain_id=toolchain,
    )


def test_mode_1_mode_2_do_not_collide():
    # The demonstrated defect: -DMODE=1 then -DMODE=2 shared one key.
    mode1 = _identity(options=["-DMODE=1"])
    mode2 = _identity(options=["-DMODE=2"])
    assert mode1.cache_key != mode2.cache_key
    assert mode1.options == ("-DMODE=1", )
    assert mode2.options == ("-DMODE=2", )


def test_option_order_duplicates_and_literal_content_are_significant():
    # argv semantics: later -D/-U override earlier ones; paired flags consume
    # the next token; duplicates and literal whitespace all matter.
    base = _identity(options=["-DMODE=1", "-DMODE=2"])
    reversed_opts = _identity(options=["-DMODE=2", "-DMODE=1"])
    assert base.cache_key != reversed_opts.cache_key

    once = _identity(options=["-DFLAG"])  # noqa: F841
    twice = _identity(options=["-DFLAG", "-DFLAG"])
    single = _identity(options=["-DFLAG"])
    assert twice.cache_key != single.cache_key

    # Paired -I consumes the following token, so these are different argv.
    paired = _identity(options=["-I", str("/inc"), "-DFLAG"])
    attached = _identity(options=["-I/inc", "-DFLAG"])
    assert paired.cache_key != attached.cache_key

    # Literal whitespace inside a token is never stripped away.
    literal = _identity(options=["-DMSG=\"a b\""])
    compact = _identity(options=["-DMSG=\"ab\""])
    assert literal.cache_key != compact.cache_key

    # Different math/register options still never collapse.
    r32 = _identity(options=["--maxrregcount=32"])
    r64 = _identity(options=["--maxrregcount=64"])
    assert r32.cache_key != r64.cache_key
    # Exactly identical argv shares an identity.
    assert base.cache_key == _identity(options=["-DMODE=1", "-DMODE=2"]).cache_key


def test_unresolved_auto_is_never_a_cache_key():
    with pytest.raises(ValueError, match="resolved"):
        BuildIdentity(kernel_name="k", source_hash="x" * 64, target_arch="auto")
    with pytest.raises(ValueError):
        _identity(target="AUTO")


def test_resolved_arch_participates_in_identity():
    sm86 = _identity(target=CudaTarget.from_arch("8.6"))
    sm90 = _identity(target="sm_90")
    sm90a = _identity(target="sm_90a")
    keys = {sm86.cache_key, sm90.cache_key, sm90a.cache_key}
    assert len(keys) == 3
    assert sm86.target_arch == "sm_86"
    assert sm90a.target_arch == "sm_90a"


def test_toolchain_identity_participates(tmp_path, monkeypatch):
    monkeypatch.setenv("LITTLE_KERNEL_TOOLCHAIN_ID", "nvcc:12.0")
    assert default_toolchain_identity() == "nvcc:12.0"
    a = _identity(toolchain="nvcc:11.8")
    b = _identity(toolchain="nvcc:12.0")
    assert a.cache_key != b.cache_key


def test_include_search_precedence_is_significant(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "mode.h").write_text("// variant A\n")
    (dir_b / "mode.h").write_text("// variant B\n")
    src = '#include "mode.h"\n__global__ void k() {}\n'

    ab = compute_build_identity(
        kernel_name="k",
        source=src,
        target_arch="sm_90",
        include_paths=[str(dir_a), str(dir_b)],
        toolchain_id="nvcc:fake",
    )
    ba = compute_build_identity(
        kernel_name="k",
        source=src,
        target_arch="sm_90",
        include_paths=[str(dir_b), str(dir_a)],
        toolchain_id="nvcc:fake",
    )
    assert ab.cache_key != ba.cache_key
    assert ab.include_paths[:2] == (
        os.path.realpath(str(dir_a)),
        os.path.realpath(str(dir_b)),
    )
    # Reversing actually selects the other resolved file content.
    dep_paths = {n for n, _ in ab.dependencies}
    assert any(n.endswith(os.path.join("a", "mode.h")) for n in dep_paths)
    dep_paths_b = {n for n, _ in ba.dependencies}
    assert any(n.endswith(os.path.join("b", "mode.h")) for n in dep_paths_b)


def test_quoted_include_resolves_relative_to_includer_first(tmp_path):
    root = tmp_path / "inc"
    nested = root / "nested"
    nested.mkdir(parents=True)
    # hdr.h lives in nested/; root is searched BEFORE nested in -I order.
    (nested / "hdr.h").write_text('#include "child.h"\n')
    (nested / "child.h").write_text("// nested child\n")
    # A shadow child at the first include root must NOT win over the
    # includer's own directory.
    (root / "child.h").write_text("// root child\n")
    src = '#include "hdr.h"\n__global__ void k() {}\n'
    deps = scan_user_dependencies(src, [str(root), str(nested)])
    chosen = [n for n, _ in deps if n.endswith("child.h")]
    assert len(chosen) == 1
    assert chosen[0].endswith(os.path.join("nested", "child.h"))
    assert any(n.endswith(os.path.join("nested", "hdr.h")) for n, _ in deps)


def test_user_angle_header_contents_participate(tmp_path):
    inc = tmp_path / "inc"
    inc.mkdir()
    (inc / "mode.h").write_text("#define MODE 1\n")
    src = '#include <mode.h>\n__global__ void k() {}\n'
    first = compute_build_identity(
        kernel_name="k",
        source=src,
        target_arch="sm_90",
        include_paths=[str(inc)],
        toolchain_id="nvcc:fake",
    )
    assert any("mode.h" in n for n, _ in first.dependencies)

    (inc / "mode.h").write_text("#define MODE 2\n")
    second = compute_build_identity(
        kernel_name="k",
        source=src,
        target_arch="sm_90",
        include_paths=[str(inc)],
        toolchain_id="nvcc:fake",
    )
    assert first.cache_key != second.cache_key


def test_option_include_dirs_forced_includes_and_response_files(tmp_path):
    inc = tmp_path / "inc"
    inc.mkdir()
    (inc / "forced.h").write_text("#define FORCED 1\n")
    src = "__global__ void k() {}\n"

    # -I carried inside options is honored for user angle headers.
    (inc / "opt.h").write_text("// opt v1\n")
    src_inc = "#include <opt.h>\n" + src
    with_i = compute_build_identity(
        kernel_name="k",
        source=src_inc,
        target_arch="sm_90",
        options=["-I", str(inc)],
        toolchain_id="nvcc:fake",
    )
    assert any("opt.h" in n for n, _ in with_i.dependencies)
    (inc / "opt.h").write_text("// opt v2\n")
    with_i_changed = compute_build_identity(
        kernel_name="k",
        source=src_inc,
        target_arch="sm_90",
        options=["-I", str(inc)],
        toolchain_id="nvcc:fake",
    )
    assert with_i.cache_key != with_i_changed.cache_key

    # Forced include content enters identity and changing it changes the key.
    forced_v1 = compute_build_identity(
        kernel_name="k",
        source=src,
        target_arch="sm_90",
        options=["-I", str(inc), "-include", "forced.h"],
        toolchain_id="nvcc:fake",
    )
    (inc / "forced.h").write_text("#define FORCED 2\n")
    forced_v2 = compute_build_identity(
        kernel_name="k",
        source=src,
        target_arch="sm_90",
        options=["-I", str(inc), "-include", "forced.h"],
        toolchain_id="nvcc:fake",
    )
    assert forced_v1.cache_key != forced_v2.cache_key

    # An unresolvable forced include cannot silently reuse an entry.
    from little_kernel.backends.nv.common.runtime.cache import UncacheableBuildError
    with pytest.raises(UncacheableBuildError):
        compute_build_identity(
            kernel_name="k",
            source=src,
            target_arch="sm_90",
            options=["-include", "missing.h"],
            toolchain_id="nvcc:fake",
        )

    # A response file's own content is fingerprinted.
    rsp = tmp_path / "args.rsp"
    rsp.write_text(f"-I {inc}\n")
    rsp_v1 = compute_build_identity(
        kernel_name="k",
        source=src,
        target_arch="sm_90",
        options=[f"@{rsp}"],
        toolchain_id="nvcc:fake",
    )
    rsp.write_text(f"-I {inc} -DEXTRA=1\n")
    rsp_v2 = compute_build_identity(
        kernel_name="k",
        source=src,
        target_arch="sm_90",
        options=[f"@{rsp}"],
        toolchain_id="nvcc:fake",
    )
    assert rsp_v1.cache_key != rsp_v2.cache_key

    missing_rsp = tmp_path / "nope.rsp"
    with pytest.raises(UncacheableBuildError):
        compute_build_identity(
            kernel_name="k",
            source=src,
            target_arch="sm_90",
            options=[f"@{missing_rsp}"],
            toolchain_id="nvcc:fake",
        )


def test_nonliteral_computed_include_fails_closed(tmp_path):
    from little_kernel.backends.nv.common.runtime.cache import UncacheableBuildError
    inc = tmp_path / "inc"
    inc.mkdir()
    (inc / "hdr.h").write_text("// hdr\n")
    # Macro-expanded / computed include cannot be closed statically.
    src = '#define HEADER "hdr.h"\n#include HEADER\n__global__ void k() {}\n'
    with pytest.raises(UncacheableBuildError, match="non-literal"):
        compute_build_identity(
            kernel_name="k",
            source=src,
            target_arch="sm_90",
            include_paths=[str(inc)],
            toolchain_id="nvcc:fake",
        )

    # Same rule applies transitively inside a user header.
    (inc / "outer.h").write_text('#include "inner.h"\n')
    (inc / "inner.h").write_text("#include CHOICE\n")
    with pytest.raises(UncacheableBuildError, match="non-literal"):
        compute_build_identity(
            kernel_name="k",
            source='#include "outer.h"\n',
            target_arch="sm_90",
            include_paths=[str(inc)],
            toolchain_id="nvcc:fake",
        )


def test_options_file_flags_all_forms_and_depth_bound(tmp_path):
    from little_kernel.backends.nv.common.runtime.cache import expand_option_dependencies, UncacheableBuildError
    inc = tmp_path / "inc"
    inc.mkdir()
    (inc / "f.h").write_text("#define F 1\n")
    f1 = tmp_path / "a.optf"
    f1.write_text(f"-I {inc}\n")

    src = "#include <f.h>\n__global__ void k() {}\n"

    # Paired, attached and comma-separated forms all expand and track content.
    paired = compute_build_identity(
        kernel_name="k",
        source=src,
        target_arch="sm_90",
        options=["-optf", str(f1)],
        toolchain_id="nvcc:fake",
    )
    attached = compute_build_identity(
        kernel_name="k",
        source=src,
        target_arch="sm_90",
        options=[f"--options-file={f1}"],
        toolchain_id="nvcc:fake",
    )
    assert any("f.h" in n for n, _ in paired.dependencies)
    assert any("f.h" in n for n, _ in attached.dependencies)

    f1.write_text(f"-I {inc} -DCHANGED=1\n")
    changed = compute_build_identity(
        kernel_name="k",
        source=src,
        target_arch="sm_90",
        options=["-optf", str(f1)],
        toolchain_id="nvcc:fake",
    )
    assert paired.cache_key != changed.cache_key

    # Comma-separated multiple files.
    f2 = tmp_path / "b.optf"
    f2.write_text("-DB=1\n")
    multi = expand_option_dependencies([f"-optf={f1},{f2}"])
    assert len(multi[3]) == 2

    # Missing options file argument / unreadable file fail closed.
    with pytest.raises(UncacheableBuildError):
        expand_option_dependencies(["--options-file"])
    with pytest.raises(UncacheableBuildError):
        expand_option_dependencies([f"-optf={tmp_path / 'missing.optf'}"])

    # A recursive cycle is detected without recursion overflow.
    cyc = tmp_path / "cyc.optf"
    cyc.write_text(f"-optf {cyc}\n")
    with pytest.raises(UncacheableBuildError):
        expand_option_dependencies([f"@{cyc}"])


def test_dependency_bearing_env_fails_closed_and_disabled_cache_compiles(monkeypatch, tmp_path):
    from little_kernel.backends.nv.common.runtime.cache import UncacheableBuildError
    monkeypatch.setenv("LITTLE_KERNEL_TOOLCHAIN_ID", "nvcc:fake")
    monkeypatch.setenv("NVCC_PREPEND_FLAGS", "-DSECRET=1")
    with pytest.raises(UncacheableBuildError, match="NVCC_PREPEND_FLAGS"):
        compute_build_identity(
            kernel_name="k",
            source="x",
            target_arch="sm_90",
            toolchain_id="nvcc:fake",
        )

    monkeypatch.setattr(cache_mod, "default_toolchain_identity", lambda: pytest.fail("no identity when caching off"))
    counter = {"n": 0}

    def fake_compile(*a, **k):
        counter["n"] += 1
        return b"cubin"

    monkeypatch.setattr(kernel_mod, "compile_cuda", fake_compile)
    # Cache disabled: untracked env must not block a normal compilation.
    k = CompiledKernel("x", "k", (1, 1, 1), (1, 1, 1), arch="sm_90", cache_dir=None, runtime=_FakeRuntimeNoDriver())
    assert k._compile() == b"cubin" and counter["n"] == 1


class _FakeRuntimeNoDriver:

    def synchronize(self, stream=None):
        pass


def test_effective_include_paths_shared_by_compiler_and_identity(monkeypatch, tmp_path):
    import little_kernel.backends.nv.common.runtime.compiler as compiler_module

    # Explicit paths preserve compiler search order (missing ones skipped).
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    package_root = str(Path(__import__("little_kernel").__file__).resolve().parent.parent)
    assert compiler_module.effective_include_paths([str(a), str(b), str(tmp_path / "x")]) == [str(a), package_root]
    assert compiler_module.effective_include_paths(None) == compiler_module.default_include_paths()

    # The implicit default include path (packaged cutlass) is seen by the
    # identity scanner too, so a default-path header enters the closure.
    default_dir = tmp_path / "default_inc"
    default_dir.mkdir()
    monkeypatch.setattr(compiler_module, "default_include_paths", lambda: [str(default_dir)])
    (default_dir / "builtin.h").write_text("// v1\n")
    src = '#include <builtin.h>\n__global__ void k() {}\n'
    first = compute_build_identity(
        kernel_name="k",
        source=src,
        target_arch="sm_90",
        include_paths=None,
        toolchain_id="nvcc:fake",
    )
    assert any("builtin.h" in n for n, _ in first.dependencies)
    (default_dir / "builtin.h").write_text("// v2\n")
    second = compute_build_identity(
        kernel_name="k",
        source=src,
        target_arch="sm_90",
        include_paths=None,
        toolchain_id="nvcc:fake",
    )
    assert first.cache_key != second.cache_key


def test_user_header_contents_participate_and_mtime_does_not(tmp_path):
    header = tmp_path / "kernel_header.h"
    header.write_text("#define TAG 1\n")

    first = _identity(include_paths=[str(tmp_path)])
    deps = scan_user_dependencies(SOURCE, [str(tmp_path)])
    assert any(n.endswith("kernel_header.h") for n, _ in deps)

    # Touching only the mtime must not change the identity.
    future = time.time() + 1000
    os.utime(header, (future, future))
    assert _identity(include_paths=[str(tmp_path)]).cache_key == first.cache_key

    # Changing header contents must.
    header.write_text("#define TAG 2\n")
    second = _identity(include_paths=[str(tmp_path)])
    assert second.cache_key != first.cache_key


def test_transitive_and_explicit_dependency_contents(tmp_path):
    (tmp_path / "a.h").write_text('#include "b.h"\n')
    (tmp_path / "b.h").write_text("struct X { int v; };\n")
    deps = scan_user_dependencies('#include "a.h"\n', [str(tmp_path)])
    names = {os.path.basename(n) for n, _ in deps}
    assert names == {"a.h", "b.h"}

    explicit = _identity(headers=[("generated.inl", "int x = 1;")])
    changed = _identity(headers=[("generated.inl", "int x = 2;")])
    assert explicit.cache_key != changed.cache_key

    # A bytes-valued dependency and a path dependency are both supported.
    with_bytes = _identity(headers=[("generated.inl", b"int x = 1;")])
    assert with_bytes.cache_key == explicit.cache_key


def test_source_content_not_mtime_identity():
    a = _identity(source="__global__ void k(){}")
    b = _identity(source="__global__ void k(){ }")
    assert a.cache_key != b.cache_key
    assert len(a.source_hash) == 64


# -----------------------------------------------------------------------------
# Atomic publication
# -----------------------------------------------------------------------------


def test_atomic_publish_readers_never_see_partial_file(tmp_path, monkeypatch):
    target = tmp_path / "out.cubin"
    started = threading.Event()
    release = threading.Event()

    real_fsync = os.fsync

    def blocking_fsync(fd):
        started.set()
        assert release.wait(timeout=5)
        return real_fsync(fd)

    monkeypatch.setattr(cache_mod.os, "fsync", blocking_fsync)

    result = {}

    def writer():
        result["path"] = atomic_publish(tmp_path, "out.cubin", b"X" * 4096)

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        assert started.wait(timeout=5)
        # While the temp file is being written/fsynced the final name is absent.
        assert not target.exists()
        temps = list(tmp_path.glob(".out.cubin.*.tmp"))
        assert len(temps) == 1
        release.set()
    finally:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert target.read_bytes() == b"X" * 4096
    # No temp files are left behind.
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_publish_cleans_temp_on_failure(tmp_path, monkeypatch):

    def boom(fd):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(cache_mod.os, "fsync", boom)
    with pytest.raises(OSError):
        atomic_publish(tmp_path, "out.cubin", b"data")
    assert not (tmp_path / "out.cubin").exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_legacy_empty_cubin_is_treated_as_miss(tmp_path):
    identity = _identity()
    cache = AtomicCubinCache(tmp_path)
    cache.cubin_path(identity).write_bytes(b"")
    assert cache.load(identity) is None

    built = cache.get_or_build(identity, lambda: b"cubin", source=SOURCE)
    assert built == b"cubin"
    assert cache.cubin_path(identity).read_bytes() == b"cubin"
    # Manifest + debug source are published alongside.
    assert cache.manifest_path(identity).exists()
    assert cache.source_path(identity).read_text() == SOURCE


def test_concurrent_builders_compile_once(tmp_path):
    identity = _identity()
    cache = AtomicCubinCache(tmp_path)
    counter = {"n": 0}
    start = threading.Event()

    def builder():
        # Hold the serialized section briefly to expose any duplicate build.
        time.sleep(0.05)
        counter["n"] += 1
        return b"cubin"

    def run(_):
        assert start.wait(timeout=5)
        return cache.get_or_build(identity, builder)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(run, i) for i in range(6)]
        start.set()
        results = [f.result(timeout=15) for f in futures]

    assert counter["n"] == 1
    assert all(r == b"cubin" for r in results)


def test_second_build_hits_cache_and_does_not_compile(tmp_path):
    identity = _identity()
    cache = AtomicCubinCache(tmp_path)
    calls = []

    def builder():
        calls.append(1)
        return b"cubin"

    assert cache.get_or_build(identity, builder) == b"cubin"
    # New cache instance (simulating a new process) loads the artifact.
    cache2 = AtomicCubinCache(tmp_path)
    assert cache2.get_or_build(identity, builder) == b"cubin"
    assert len(calls) == 1


def test_empty_builder_output_is_rejected(tmp_path):
    cache = AtomicCubinCache(tmp_path)
    with pytest.raises(RuntimeError):
        cache.get_or_build(_identity(), lambda: b"")
    assert not list(tmp_path.glob("*.cubin"))


# -----------------------------------------------------------------------------
# CompiledKernel integration with the real compile_cuda hook
# -----------------------------------------------------------------------------


class _ModuleBusyError(RuntimeError):
    """Mirrors the Driver refusing to unload a module with live users."""


class _FakeRuntime:
    """Integration-shaped runtime: modules carry live-user counts.

    ``unload_module`` *refuses* while a launcher/user is attached, exactly as
    the merged RF04 runtime does -- it never blindly accepts an unload.
    """

    def __init__(self):
        self.loaded = []
        self.unloaded = []
        self.closed = 0
        self.users = {}

    def load_module(self, cubin):
        module = ("module", cubin)
        self.loaded.append(module)
        self.users[module] = 0
        return module

    def attach(self, module):
        self.users[module] = self.users.get(module, 0) + 1

    def detach(self, module):
        self.users[module] -= 1

    def unload_module(self, module):
        if self.users.get(module, 0) != 0:
            raise _ModuleBusyError(f"refusing to unload {module!r}: {self.users[module]} live user(s)")
        self.unloaded.append(module)
        self.users.pop(module, None)

    def close(self):
        self.closed += 1

    def synchronize(self, stream=None):
        pass


class _ModuleUserLauncher:
    """RF04-shaped launcher: registers a module user, detaches on close."""

    def __init__(self, runtime, module, kernel_name, shared_mem_bytes=0, cubin_data=None, cluster_shape=None,
                 allow_non_portable_cluster=False, signature=None):
        self.runtime = runtime
        self.module = module
        self.kernel_name = kernel_name
        self.signature = signature
        self.detached = False
        self.closed_calls = 0
        runtime.attach(module)

    def close(self):
        if self.detached:
            return
        self.runtime.detach(self.module)
        self.detached = True
        self.closed_calls += 1


def _patch_compile(monkeypatch, counter, *, by_option=False):

    def fake_compile_cuda(code, kernel_name, arch=None, target_format="cubin", options=None, verbose=False,
                          include_paths=None, cache_dir=None, device=None):
        counter["n"] += 1
        counter["last_arch"] = arch
        token = b"-".join((options or ["plain"])[0].encode().split()) or b"plain"
        return b"CUBIN-" + token

    monkeypatch.setattr(kernel_mod, "compile_cuda", fake_compile_cuda)


def test_disabled_cache_compiles_without_target_or_toolchain_probe(monkeypatch):
    # Cache disabled must not identify the build: neither target resolution
    # nor toolchain probing should run, and no Driver is required.
    monkeypatch.setattr(
        kernel_mod,
        "resolve_target",
        lambda *a, **k: pytest.fail("target must not be resolved when caching is off"),
        raising=False,
    )
    monkeypatch.setattr(
        cache_mod,
        "default_toolchain_identity",
        lambda: pytest.fail("toolchain must not be identified when caching is off"),
    )

    def boom(*a, **k):
        raise AssertionError("CUDARuntime/Driver must not be constructed to compile")

    monkeypatch.setattr(kernel_mod, "CUDARuntime", boom)
    counter = {"n": 0}
    _patch_compile(monkeypatch, counter)

    kernel = CompiledKernel(SOURCE, "k", (1, 1, 1), (1, 1, 1), arch=None, cache_dir=None)
    cubin = kernel._compile()
    assert cubin == b"CUBIN-plain"
    assert counter["n"] == 1


def test_mode_collision_fixed_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("LITTLE_KERNEL_TOOLCHAIN_ID", "nvcc:fake")
    counter = {"n": 0}
    _patch_compile(monkeypatch, counter, by_option=True)
    rt = _FakeRuntime()

    k1 = CompiledKernel(
        SOURCE,
        "k",
        (1, 1, 1),
        (1, 1, 1),
        arch="sm_90",
        cache_dir=str(tmp_path),
        extra_nvcc_options=["-DMODE=1"],
        runtime=rt,
    )
    k2 = CompiledKernel(
        SOURCE,
        "k",
        (1, 1, 1),
        (1, 1, 1),
        arch="sm_90",
        cache_dir=str(tmp_path),
        extra_nvcc_options=["-DMODE=2"],
        runtime=rt,
    )
    assert k1.build_identity().cache_key != k2.build_identity().cache_key

    assert k1._compile() == b"CUBIN--DMODE=1"
    assert k2._compile() == b"CUBIN--DMODE=2"
    assert counter["n"] == 2

    # Rebuilding both from a fresh cache instance hits the artifacts.
    k1b = CompiledKernel(
        SOURCE,
        "k",
        (1, 1, 1),
        (1, 1, 1),
        arch="sm_90",
        cache_dir=str(tmp_path),
        extra_nvcc_options=["-DMODE=1"],
        runtime=rt,
    )
    k2b = CompiledKernel(
        SOURCE,
        "k",
        (1, 1, 1),
        (1, 1, 1),
        arch="sm_90",
        cache_dir=str(tmp_path),
        extra_nvcc_options=["-DMODE=2"],
        runtime=rt,
    )
    assert k1b._compile() == b"CUBIN--DMODE=1"
    assert k2b._compile() == b"CUBIN--DMODE=2"
    assert counter["n"] == 2  # no recompilation


def test_auto_arch_is_resolved_before_keying(tmp_path, monkeypatch):
    monkeypatch.setenv("LITTLE_KERNEL_TOOLCHAIN_ID", "nvcc:fake")

    # arch=None must be resolved (never keyed as 'auto'): resolve_target gives sm_86.
    import little_kernel.backends.nv.common.runtime.target as target_mod

    monkeypatch.setattr(
        target_mod,
        "resolve_target",
        lambda *a, **k: CudaTarget.from_arch("sm_86"),
    )
    counter = {"n": 0}
    _patch_compile(monkeypatch, counter)

    kernel = CompiledKernel(
        SOURCE,
        "k",
        (1, 1, 1),
        (1, 1, 1),
        arch=None,
        cache_dir=str(tmp_path),
        runtime=_FakeRuntime(),
    )
    identity = kernel.build_identity()
    assert identity.target_arch == "sm_86"
    kernel._compile()
    assert counter["last_arch"] == "sm_86"


def test_legacy_positional_keyword_compatibility(monkeypatch):
    monkeypatch.setenv("LITTLE_KERNEL_TOOLCHAIN_ID", "nvcc:fake")
    # Exactly the pre-refactor positional order still works.
    kernel = CompiledKernel(SOURCE, "k", (1, 1, 1), (1, 1, 1), 128, "sm_90", None, False, None)
    assert kernel.shared_mem_bytes == 128
    assert kernel.arch == "sm_90"
    assert kernel.signature is None

    sig = KernelSignature([KernelParameter.pointer("out")])
    typed = CompiledKernel(SOURCE, "k", (1, 1, 1), (1, 1, 1), signature=sig)
    assert typed.signature is sig

    with pytest.raises(TypeError):
        CompiledKernel(SOURCE, "k", (1, ), (1, ), signature="not-a-signature")


def test_from_artifact_factory(monkeypatch):
    monkeypatch.setenv("LITTLE_KERNEL_TOOLCHAIN_ID", "nvcc:fake")
    sig = KernelSignature([KernelParameter.pointer("out")])
    artifact = CompilationArtifact(
        source=SOURCE,
        kernel_name="entry_kernel",
        signature=sig,
        target=CudaTarget.from_arch("sm_90a"),
        metadata={"options": ("--use_fast_math", )},
    )
    kernel = CompiledKernel.from_artifact(
        artifact,
        grid=(1, 1, 1),
        block=(128, 1, 1),
        extra_nvcc_options=["-DEXTRA=1"],
    )
    assert kernel.cuda_code == SOURCE
    assert kernel.kernel_name == "entry_kernel"
    assert kernel.arch == "sm_90a"
    assert kernel.block == (128, 1, 1)
    assert kernel.signature is sig
    # Artifact options and explicit options both survive; order normalized.
    assert set(kernel.extra_nvcc_options) == {"--use_fast_math", "-DEXTRA=1"}

    identity = kernel.build_identity()
    assert identity.target_arch == "sm_90a"

    # Untyped artifact remains an explicit untyped path.
    untyped = CompilationArtifact(source=SOURCE, kernel_name="k", target="sm_80")
    ku = CompiledKernel.from_artifact(untyped, (1, 1, 1), (1, 1, 1))
    assert ku.signature is None and ku.arch == "sm_80"


# -----------------------------------------------------------------------------
# Signature propagation and runtime ownership/lifecycle
# -----------------------------------------------------------------------------


class _RecordingLauncher:
    last = None

    def __init__(self, runtime, module, kernel_name, shared_mem_bytes=0, cubin_data=None, cluster_shape=None,
                 allow_non_portable_cluster=False, signature=None):
        self.runtime = runtime
        self.module = module
        self.kernel_name = kernel_name
        self.signature = signature
        _RecordingLauncher.last = self


def test_signature_propagated_to_launcher(monkeypatch):
    monkeypatch.setattr(kernel_mod, "_LAUNCHER_ACCEPTS_SIGNATURE", True)
    monkeypatch.setattr(kernel_mod, "KernelLauncher", _RecordingLauncher)
    sig = KernelSignature([KernelParameter.pointer("out")])
    rt = _FakeRuntime()
    kernel = CompiledKernel(b"x", "k", (1, 1, 1), (1, 1, 1), runtime=rt, signature=sig)
    # Avoid compilation: seed the module directly.
    kernel._module = rt.load_module(b"cubin")
    launcher = kernel.launcher
    assert launcher is _RecordingLauncher.last
    assert launcher.signature is sig
    # Cached on repeat access.
    assert kernel.launcher is launcher


def test_known_signature_rejected_when_launcher_cannot_accept_it(monkeypatch):
    # An old/mock launcher lacking the merged trailing-signature contract must
    # not silently drop a known signature into value-guessing fallback.
    monkeypatch.setattr(kernel_mod, "_LAUNCHER_ACCEPTS_SIGNATURE", False)
    monkeypatch.setattr(kernel_mod, "KernelLauncher", _RecordingLauncher)
    sig = KernelSignature([KernelParameter.pointer("out")])
    rt = _FakeRuntime()
    kernel = CompiledKernel(b"x", "k", (1, 1, 1), (1, 1, 1), runtime=rt, signature=sig)
    kernel._module = rt.load_module(b"cubin")
    with pytest.raises(RuntimeError, match="signature"):
        _ = kernel.launcher

    # The explicit untyped path still works against an old launcher.
    untyped = CompiledKernel(b"x", "j", (1, 1, 1), (1, 1, 1), runtime=rt)
    untyped._module = rt.load_module(b"cubin2")
    assert untyped.launcher is _RecordingLauncher.last


def test_close_detaches_launcher_before_unloading(monkeypatch):
    monkeypatch.setattr(kernel_mod, "_LAUNCHER_ACCEPTS_SIGNATURE", True)
    monkeypatch.setattr(kernel_mod, "KernelLauncher", _ModuleUserLauncher)
    rt = _FakeRuntime()
    kernel = CompiledKernel(b"x", "k", (1, 1, 1), (1, 1, 1), runtime=rt)
    kernel._module = rt.load_module(b"cubin")
    module = kernel._module
    launcher = kernel.launcher  # registers a live module user
    assert rt.users[module] == 1

    kernel.close()
    assert launcher.detached and launcher.closed_calls == 1
    assert rt.unloaded == [module]
    assert rt.users.get(module, 0) == 0
    assert rt.closed == 0  # injected runtime is borrowed/shared

    kernel.close()  # idempotent: no second detach/unload
    assert launcher.closed_calls == 1
    assert rt.unloaded == [module]
    with pytest.raises(RuntimeError):
        _ = kernel.module


def test_close_refused_when_module_user_remains_but_state_preserved(monkeypatch):
    monkeypatch.setattr(kernel_mod, "_LAUNCHER_ACCEPTS_SIGNATURE", True)
    monkeypatch.setattr(kernel_mod, "KernelLauncher", _ModuleUserLauncher)
    rt = _FakeRuntime()
    kernel = CompiledKernel(b"x", "k", (1, 1, 1), (1, 1, 1), runtime=rt)
    kernel._module = rt.load_module(b"cubin")
    module = kernel._module
    launcher = kernel.launcher
    # Simulate a release failure: the launcher did NOT detach despite close.
    monkeypatch.setattr(launcher, "close", lambda: None)
    launcher.detached = False

    with pytest.raises(_ModuleBusyError):
        kernel.close()
    # State preserved for retry: not marked closed, references retained.
    assert not kernel._closed
    assert kernel._module is module and kernel._launcher is launcher
    assert rt.unloaded == []

    # Recover the detach, then close succeeds.
    monkeypatch.undo()
    kernel.close()
    assert rt.unloaded == [module]
    assert kernel._closed


def test_owned_runtime_is_closed_and_shared_runtime_is_not(monkeypatch):
    monkeypatch.setattr(kernel_mod, "_LAUNCHER_ACCEPTS_SIGNATURE", True)
    monkeypatch.setattr(kernel_mod, "KernelLauncher", _ModuleUserLauncher)
    shared = _FakeRuntime()

    k_a = CompiledKernel(b"a", "a", (1, 1, 1), (1, 1, 1), runtime=shared)
    k_b = CompiledKernel(b"b", "b", (1, 1, 1), (1, 1, 1), runtime=shared)
    k_a._module = shared.load_module(b"a")
    k_b._module = shared.load_module(b"b")
    mod_a = k_a._module

    k_a.close()
    assert shared.unloaded == [mod_a]
    assert shared.closed == 0  # shared runtime survives the first close
    # The other kernel can still use its borrowed runtime.
    assert k_b.runtime is shared
    mod_b = k_b._module
    k_b.close()
    assert len(shared.unloaded) == 2 and mod_b in shared.unloaded
    assert shared.closed == 0


def test_context_manager_closes_only_owned(monkeypatch):
    monkeypatch.setattr(kernel_mod, "_LAUNCHER_ACCEPTS_SIGNATURE", True)
    monkeypatch.setattr(kernel_mod, "KernelLauncher", _ModuleUserLauncher)
    rt = _FakeRuntime()
    sig = KernelSignature([KernelParameter.pointer("out")])
    with CompiledKernel(b"x", "k", (1, 1, 1), (1, 1, 1), runtime=rt, signature=sig) as kernel:
        assert kernel.signature is sig
        kernel._module = rt.load_module(b"cubin")
        module = kernel._module
        _ = kernel.launcher
    assert rt.unloaded == [module]
    assert rt.closed == 0  # injected runtime is borrowed

    class _HooklessRuntime:

        def load_module(self, cubin):
            return "mod"

    # Runtime without ownership hooks must not break close().
    hookless = _HooklessRuntime()
    k = CompiledKernel(b"x", "k", (1, 1, 1), (1, 1, 1), runtime=hookless)
    k._module = "mod"
    k.close()  # no exception


def test_close_is_idempotent_under_concurrency(monkeypatch):
    monkeypatch.setattr(kernel_mod, "_LAUNCHER_ACCEPTS_SIGNATURE", True)
    monkeypatch.setattr(kernel_mod, "KernelLauncher", _ModuleUserLauncher)
    rt = _FakeRuntime()
    kernel = CompiledKernel(b"x", "k", (1, 1, 1), (1, 1, 1), runtime=rt)
    kernel._module = rt.load_module(b"cubin")
    module = kernel._module
    kernel.launcher  # attach a user
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: kernel.close(), range(8)))
    assert rt.unloaded == [module]
    assert rt.users.get(module, 0) == 0


def test_absolute_include_changes_identity_without_search_directories(tmp_path):
    header = tmp_path / "absolute.h"
    header.write_text("#define MODE 1\n")
    source = f'#include "{header}"\n__global__ void k() {{}}'
    first = _identity(source=source, include_paths=[])
    assert str(header.resolve()) in dict(first.dependencies)
    header.write_text("#define MODE 2\n")
    assert _identity(source=source, include_paths=[]).cache_key != first.cache_key


def test_injected_runtime_device_controls_auto_target_and_uncached_compile(tmp_path, monkeypatch):
    import little_kernel.backends.nv.common.runtime.target as target

    monkeypatch.setenv("LITTLE_KERNEL_TOOLCHAIN_ID", "nvcc:fake")
    chosen = []

    def resolve_target(*, device=None):
        chosen.append(device)
        return CudaTarget.from_arch("sm_100a" if device == 1 else "sm_90a")

    monkeypatch.setattr(target, "resolve_target", resolve_target)
    runtime = _FakeRuntime()
    runtime.device = 1
    cached = CompiledKernel("source", "k", (1, 1, 1), (1, 1, 1), runtime=runtime, cache_dir=str(tmp_path))
    assert cached.build_identity().target_arch == "sm_100a"
    assert chosen == [1]
    received = []

    def compile_cuda(*args, **kwargs):
        received.append(kwargs)
        return b"cubin"

    monkeypatch.setattr(kernel_mod, "compile_cuda", compile_cuda)
    uncached = CompiledKernel("source", "k", (1, 1, 1), (1, 1, 1), runtime=runtime)
    assert uncached._compile() == b"cubin"
    assert received[0]["device"] == 1


def test_include_next_is_explicitly_uncacheable(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    (first / "layer.h").write_text("#include_next <layer.h>\n")
    (second / "layer.h").write_text("#define MODE 1\n")
    with pytest.raises(cache_mod.UncacheableBuildError, match="include_next"):
        _identity(source="#include <layer.h>\n", include_paths=[first, second])


def test_attached_forced_header_and_list_paths_are_fingerprinted(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    forced = tmp_path / "forced.h"
    forced.write_text("#define MODE 1\n")
    (b / "value.h").write_text("#define VALUE 2\n")
    options = [f"--pre-include={forced}", f"--include-path={a},{b}"]
    first = _identity(source="#include <value.h>\n", include_paths=[], options=options)
    assert {str(forced), str(b / "value.h")} <= set(dict(first.dependencies))
    forced.write_text("#define MODE 3\n")
    assert _identity(source="#include <value.h>\n", include_paths=[], options=options).cache_key != first.cache_key


def test_explicit_dependency_paths_do_not_collide_by_basename(tmp_path):
    paths = [tmp_path / "a" / "config.json", tmp_path / "b" / "config.json"]
    for path in paths:
        path.parent.mkdir()
        path.write_text("1")
    first = _identity(headers=paths)
    paths[0].write_text("2")
    assert _identity(headers=paths).cache_key != first.cache_key
    with pytest.raises(cache_mod.UncacheableBuildError, match="duplicate explicit"):
        _identity(headers=[("config", "1"), ("config", "2")])


def test_close_retry_does_not_repeat_successful_module_unload():

    class FaultOnceRuntime(_FakeRuntime):

        def close(self):
            self.closed += 1
            if self.closed == 1:
                raise RuntimeError("context release failed")

        def unload_module(self, module):
            assert module not in self.unloaded, "module was already released"
            return super().unload_module(module)

    runtime = FaultOnceRuntime()
    kernel = CompiledKernel("source", "k", (1, 1, 1), (1, 1, 1), runtime=runtime, owns_runtime=True)
    kernel._module = runtime.load_module(b"cubin")
    with pytest.raises(RuntimeError, match="context release failed"):
        kernel.close()
    assert len(runtime.unloaded) == 1
    assert kernel._module is None
    with pytest.raises(RuntimeError, match="closing"):
        _ = kernel.module
    kernel.close()
    kernel.close()
    assert runtime.closed == 2 and len(runtime.unloaded) == 1


def test_lazy_runtime_creation_is_serialized_with_close(monkeypatch):
    entered, release, close_started = (threading.Event() for _ in range(3))
    runtime = _FakeRuntime()

    def create_runtime():
        entered.set()
        assert release.wait(5)
        return runtime

    def close_kernel(kernel):
        close_started.set()
        kernel.close()

    monkeypatch.setattr(kernel_mod, "CUDARuntime", create_runtime)
    kernel = CompiledKernel("source", "k", (1, 1, 1), (1, 1, 1))
    with ThreadPoolExecutor(max_workers=2) as pool:
        initializing = pool.submit(lambda: kernel.runtime)
        assert entered.wait(5)
        closing = pool.submit(close_kernel, kernel)
        assert close_started.wait(5)
        try:
            assert not closing.done()
        finally:
            release.set()
        assert initializing.result(timeout=5) is runtime
        closing.result(timeout=5)
    assert kernel._closed and runtime.closed == 1
    kernel.close()
    assert runtime.closed == 1


def test_relative_include_paths_survive_private_compile_directory(tmp_path, monkeypatch):
    import shutil
    from little_kernel.backends.nv.common.runtime.compiler import compile_cuda, effective_include_paths

    if shutil.which("nvcc") is None:
        pytest.skip("NVCC is required for the user-header compilation check")
    include = tmp_path / "headers"
    include.mkdir()
    (include / "value.h").write_text("#define VALUE 7\n")
    monkeypatch.chdir(tmp_path)
    assert effective_include_paths(["headers"])[0] == str(include)
    source = '#include "value.h"\nextern "C" __global__ void k(int* out) { out[0] = VALUE; }'
    assert compile_cuda(source, "k", arch="sm_80", include_paths=["headers"])
