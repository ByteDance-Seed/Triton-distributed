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

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from flash_comm.utils import (
    CuTeDSLKernelCache,
    KernelArtifactStore,
    KernelHandle,
    KernelSpec,
    cached_cutedsl_kernel,
    get_global_cutedsl_kernel_cache,
)


@contextmanager
def patched_attr(owner, name, value):
    old_value = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old_value)


@contextmanager
def patched_env(name, value):
    old_value = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_value


def fail(message):
    raise AssertionError(message)


def test_unified_cache_memory_hit():
    calls = {"count": 0}

    def build():
        calls["count"] += 1
        return lambda x: x + 1

    cache = CuTeDSLKernelCache(mode="jit_only")
    spec = KernelSpec(op_name="unit", variant_args=("bf16", 128), builder=build)

    first = cache.get_or_compile(spec)
    second = cache.get_or_compile(spec)

    assert first is second
    assert first(1) == 2
    assert calls["count"] == 1
    assert cache.stats()["memory_hits"] == 1
    assert cache.stats()["ops"] == {"unit": 1}


def test_kernel_key_is_used_for_memory_and_jit_paths():
    calls = {"count": 0}
    cache = CuTeDSLKernelCache(mode="jit_only")

    def build():
        calls["count"] += 1
        return lambda x: x + 1

    spec = KernelSpec(op_name="keyed", variant_args=("bf16", 128, False), builder=build)
    first = cache.get_or_compile(spec)
    second = cache.get_or_compile(
        KernelSpec(
            op_name="keyed",
            variant_args=("bf16", 128, False),
            builder=lambda: fail("memory hit should not build"),
        ))

    assert first is second
    assert first(2) == 3
    assert calls["count"] == 1
    assert cache.stats()["memory_hits"] == 1
    assert first.key == spec.key()


def test_global_cache_singleton():
    first = get_global_cutedsl_kernel_cache()
    second = get_global_cutedsl_kernel_cache()
    assert first is second


def test_cache_strict_mode_requires_aot():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CuTeDSLKernelCache(aot_dir=tmpdir, mode="strict")
        spec = KernelSpec(op_name="unit", variant_args=("missing", ), builder=lambda: None)

        try:
            cache.get_or_compile(spec)
        except FileNotFoundError:
            pass
        else:
            fail("strict mode should fail on AOT miss")

        assert cache.stats()["strict_misses"] == 1


def test_cached_kernel_decorator_uses_owner_cache():

    class Op:

        def __init__(self):
            self._cache = CuTeDSLKernelCache(mode="jit_only")

        @cached_cutedsl_kernel
        def compile(self, value):
            return KernelSpec(op_name="decorated", variant_args=(value, ), builder=lambda: lambda x: x * value)

    op = Op()
    assert op.compile(3)(4) == 12
    assert op.compile(3)(5) == 15
    assert op._cache.stats()["entries"] == 1


def test_record_keys_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        record = Path(tmpdir) / "keys.jsonl"
        with patched_env("FLASH_COMM_CUTEDSL_RECORD_KEYS", str(record)):
            cache = CuTeDSLKernelCache(mode="jit_only")
            spec = KernelSpec(op_name="recorded", variant_args={"hidden": 128}, builder=lambda: lambda: None)
            cache.get_or_compile(spec)

        lines = record.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        item = json.loads(lines[0])
        assert item["key"]["op_name"] == "recorded"
        assert item["key_hash"]


def test_artifact_store_paths_and_lock():
    with tempfile.TemporaryDirectory() as tmpdir:
        spec = KernelSpec(op_name="some_op", variant_args=("x", ), builder=lambda: lambda: None)
        key = spec.key()
        store = KernelArtifactStore(tmpdir)
        paths = store.paths(key, spec.symbol_name(key))

        assert paths.directory.parent.name == "some_op"
        assert paths.manifest.name == "manifest.json"
        with store.lock(key):
            assert paths.lock_file.exists()


def test_aot_load_hit_is_cached():
    from flash_comm.utils.cutedsl_aot import KernelArtifactStore as Store

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = KernelSpec(op_name="loadable", variant_args=("x", ), builder=lambda: fail("should not jit"))
        key = spec.key()
        loaded = KernelHandle(key=key, function=lambda x: x, source="aot", symbol_name=spec.symbol_name(key))

        def fake_load(self, load_spec, load_key):
            assert load_spec.op_name == spec.op_name
            assert load_key.hash == key.hash
            return loaded

        with patched_attr(Store, "load", fake_load):
            cache = CuTeDSLKernelCache(aot_dir=tmpdir, mode="auto")
            handle = cache.get_or_compile(spec)

            assert handle is loaded
            assert cache.get_or_compile(spec) is loaded
            assert cache.stats()["aot_hits"] == 1
            assert cache.stats()["memory_hits"] == 1


def test_aot_export_writes_manifest_and_outputs():
    from flash_comm.utils.cutedsl_aot import KernelArtifactStore as Store
    import flash_comm.utils.cutedsl_aot as aot

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = KernelSpec(op_name="exportable", variant_args=("x", ), builder=lambda: None)
        key = spec.key()
        symbol = spec.symbol_name(key)
        store = KernelArtifactStore(tmpdir)

        class FakeCompiled:

            def export_to_c(self, path, function_name):
                assert function_name == symbol
                Path(path).write_text("object", encoding="utf-8")

        def fake_run(cmd, check):
            assert check is True
            assert cmd[:3] == ["gcc", "-shared", "-o"]
            Path(cmd[3]).write_text("shared", encoding="utf-8")

        with patched_attr(Store, "_runtime_libraries", staticmethod(lambda: [])):
            with patched_attr(aot.subprocess, "run", fake_run):
                exported = store.export(
                    spec,
                    KernelHandle(key=key, function=FakeCompiled(), source="jit", symbol_name=symbol),
                )
        paths = store.paths(key, symbol)
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))

        assert exported.source == "jit+aot_export"
        assert paths.object_file.read_text(encoding="utf-8") == "object"
        assert paths.shared_object.read_text(encoding="utf-8") == "shared"
        assert manifest["key_hash"] == key.hash
        assert manifest["symbol"] == symbol
        assert manifest["key"]["op_name"] == key.op_name
        assert manifest["key"]["backend"] == key.backend


def test_aot_load_ignores_manifest_mismatch():
    with tempfile.TemporaryDirectory() as tmpdir:
        spec = KernelSpec(op_name="loadable", variant_args=("x", ), builder=lambda: None)
        key = spec.key()
        store = KernelArtifactStore(tmpdir)
        paths = store.paths(key, spec.symbol_name(key))
        paths.directory.mkdir(parents=True)
        paths.shared_object.write_text("shared", encoding="utf-8")
        paths.manifest.write_text(json.dumps({"key_hash": "wrong", "symbol": spec.symbol_name(key)}), encoding="utf-8")

        assert store.load(spec, key) is None


def test_aot_paths_are_reconstructed_from_runtime_key():
    with tempfile.TemporaryDirectory() as tmpdir:
        build_spec = KernelSpec(
            op_name="loadable",
            variant_args=("bf16", 128, False),
            builder=lambda: None,
            compile_options=("--enable-tvm-ffi", ),
        )
        runtime_spec = KernelSpec(
            op_name="loadable",
            variant_args=("bf16", 128, False),
            builder=lambda: fail("strict AOT hit should not jit"),
            compile_options=("--enable-tvm-ffi", ),
        )
        assert build_spec.key() == runtime_spec.key()
        assert build_spec.key().hash == runtime_spec.key().hash

        store = KernelArtifactStore(tmpdir)
        build_paths = store.paths(build_spec.key(), build_spec.symbol_name(build_spec.key()))
        runtime_paths = store.paths(runtime_spec.key(), runtime_spec.symbol_name(runtime_spec.key()))
        assert build_paths.directory == runtime_paths.directory
        assert build_paths.shared_object == runtime_paths.shared_object


def test_export_on_jit_exports_and_caches():
    from flash_comm.utils.cutedsl_aot import KernelArtifactStore as Store

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CuTeDSLKernelCache(aot_dir=tmpdir, mode="auto", export_on_jit=True)
        calls = {"build": 0, "export": 0}

        def build():
            calls["build"] += 1
            return lambda x: x

        def fake_export(self, spec, handle):
            calls["export"] += 1
            return KernelHandle(
                key=handle.key,
                function=handle.function,
                source="jit+aot_export",
                symbol_name=handle.symbol_name,
                artifact_path=str(Path(tmpdir) / "fake.so"),
            )

        with patched_attr(Store, "export", fake_export):
            spec = KernelSpec(op_name="export-on-jit", variant_args=("x", ), builder=build)

            first = cache.get_or_compile(spec)
            second = cache.get_or_compile(spec)

        assert first is second
        assert first.source == "jit+aot_export"
        assert calls == {"build": 1, "export": 1}
        assert cache.stats()["exports"] == 1
        assert cache.stats()["memory_hits"] == 1


def test_aot_tool_manifest_helpers():
    from flash_comm.tools.cutedsl_aot import _op_configs, _selected_ops

    manifest = {
        "ops": ["group_gemm"],
        "group_gemm": [{"n_out": 5120}],
    }

    assert _selected_ops(manifest) == {"group_gemm"}
    assert _selected_ops(manifest, ["dispatch"]) == {"dispatch"}
    assert tuple(_op_configs(manifest, "group_gemm", {})) == ({"n_out": 5120}, )
    assert tuple(_op_configs(manifest, "combine_push", {"hidden": 8192})) == ({"hidden": 8192}, )


def main():
    tests = [
        test_unified_cache_memory_hit,
        test_kernel_key_is_used_for_memory_and_jit_paths,
        test_global_cache_singleton,
        test_cache_strict_mode_requires_aot,
        test_cached_kernel_decorator_uses_owner_cache,
        test_record_keys_jsonl,
        test_artifact_store_paths_and_lock,
        test_aot_load_hit_is_cached,
        test_aot_export_writes_manifest_and_outputs,
        test_aot_load_ignores_manifest_mismatch,
        test_aot_paths_are_reconstructed_from_runtime_key,
        test_export_on_jit_exports_and_caches,
        test_aot_tool_manifest_helpers,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} tests")


if __name__ == "__main__":
    main()
