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
"""AOT artifact helpers for TVM FFI compiled CuTeDSL kernels."""

import contextlib
import fcntl
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from .cutedsl_cache import KernelHandle, KernelKey, KernelSpec

_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class ArtifactPaths:
    directory: Path
    manifest: Path
    object_file: Path
    shared_object: Path
    lock_file: Path


class KernelArtifactStore:
    """Filesystem store for TVM FFI object/shared-library artifacts."""

    def __init__(self, root: str) -> None:
        self.root = Path(root).expanduser().resolve()

    def paths(self, key: KernelKey, symbol_name: str) -> ArtifactPaths:
        key_dir = self.root / key.op_name / key.hash
        lock_dir = self.root / ".locks"
        return ArtifactPaths(
            directory=key_dir,
            manifest=key_dir / _MANIFEST_NAME,
            object_file=key_dir / f"{symbol_name}.o",
            shared_object=key_dir / f"{symbol_name}.so",
            lock_file=lock_dir / f"{key.op_name}_{key.hash}.lock",
        )

    @contextlib.contextmanager
    def lock(self, key: KernelKey) -> Iterator[None]:
        paths = self.paths(key, "lock")
        paths.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(paths.lock_file, "w", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def load(self, spec: KernelSpec, key: KernelKey) -> Optional[KernelHandle]:
        symbol = spec.symbol_name(key)
        paths = self.paths(key, symbol)
        if not paths.manifest.exists() or not paths.shared_object.exists():
            return None
        manifest = self._read_manifest(paths.manifest)
        if manifest.get("key_hash") != key.hash or manifest.get("symbol") != symbol:
            return None
        import cutlass.cute as cute

        module = cute.runtime.load_module(str(paths.shared_object), enable_tvm_ffi=True)
        function = getattr(module, symbol)
        return KernelHandle(
            key=key,
            function=function,
            source="aot",
            symbol_name=symbol,
            artifact_path=str(paths.shared_object),
        )

    def export(self, spec: KernelSpec, handle: KernelHandle) -> KernelHandle:
        symbol = spec.symbol_name(handle.key)
        paths = self.paths(handle.key, symbol)
        paths.directory.mkdir(parents=True, exist_ok=True)

        tmp_suffix = f".tmp.{os.getpid()}"
        tmp_obj = paths.object_file.with_suffix(paths.object_file.suffix + tmp_suffix)
        tmp_so = paths.shared_object.with_suffix(paths.shared_object.suffix + tmp_suffix)
        tmp_manifest = paths.manifest.with_suffix(paths.manifest.suffix + tmp_suffix)
        for path in (tmp_obj, tmp_so, tmp_manifest):
            if path.exists():
                path.unlink()

        if not hasattr(handle.function, "export_to_c"):
            raise TypeError(f"Compiled function for {spec.op_name} does not support export_to_c")

        handle.function.export_to_c(str(tmp_obj), function_name=symbol)
        runtime_libs = self._runtime_libraries()
        cmd = ["gcc", "-shared", "-o", str(tmp_so), str(tmp_obj), *runtime_libs]
        subprocess.run(cmd, check=True)

        manifest = {
            "op_name": handle.key.op_name,
            "backend": handle.key.backend,
            "key": handle.key.payload(),
            "key_hash": handle.key.hash,
            "symbol": symbol,
            "object_file": paths.object_file.name,
            "shared_object": paths.shared_object.name,
            "created_at": time.time(),
        }
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")

        os.replace(tmp_obj, paths.object_file)
        os.replace(tmp_so, paths.shared_object)
        os.replace(tmp_manifest, paths.manifest)
        return KernelHandle(
            key=handle.key,
            function=handle.function,
            source="jit+aot_export",
            symbol_name=symbol,
            artifact_path=str(paths.shared_object),
        )

    @staticmethod
    def _runtime_libraries() -> list[str]:
        import cutlass.cute as cute

        return list(cute.runtime.find_runtime_libraries(enable_tvm_ffi=True))

    @staticmethod
    def _read_manifest(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
