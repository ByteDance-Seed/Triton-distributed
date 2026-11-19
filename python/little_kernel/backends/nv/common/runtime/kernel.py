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
"""Compiled CUDA kernel handle with complete build identity (RF02/RF04).

This module wires three contracts together:

* :mod:`little_kernel.backends.nv.common.runtime.cache` computes a *complete* build identity
  (resolved target, source, options, user header contents, toolchain) and
  publishes cubins atomically;
* an optional :class:`little_kernel.runtime.abi.KernelSignature` is accepted
  as a trailing constructor argument and propagated to
  :class:`~little_kernel.backends.nv.common.runtime.cuda_runtime.KernelLauncher`;
* the loaded Driver module/runtime has an explicit, idempotent close
  lifecycle that never destroys a borrowed/shared runtime.
"""

from __future__ import annotations

import inspect
import threading
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

from little_kernel.backends.nv.common.runtime.compiler import compile_cuda
from little_kernel.backends.nv.common.runtime.cuda_runtime import CUDARuntime, KernelLauncher
from little_kernel.backends.nv.common.runtime.cache import AtomicCubinCache, BuildIdentity, compute_build_identity

__all__ = ["CompiledKernel", "compile_and_load_kernel"]


# Whether the merged KernelLauncher accepts the trailing typed ``signature``
# (RF04). Detected once rather than assumed, so this module keeps working with
# both the pre- and post-RF04 launcher signatures.
def _launcher_accepts_signature() -> bool:
    try:
        params = inspect.signature(KernelLauncher.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False
    return "signature" in params


_LAUNCHER_ACCEPTS_SIGNATURE = _launcher_accepts_signature()


class CompiledKernel:
    """Compiled CUDA kernel that can be launched from Python.

    Parameters
    ----------
    signature:
        Optional trailing :class:`~little_kernel.runtime.abi.KernelSignature`.
        It is propagated to :class:`KernelLauncher`; a raw-source caller may
        supply one, and an untyped (``None``) path can never override a known
        signature.
    runtime:
        Optional existing :class:`CUDARuntime`. When supplied the runtime is
        *borrowed/shared* and :meth:`close` will never destroy it. When
        omitted, the kernel lazily creates and owns its runtime.
    dependencies:
        Optional extra build dependencies (paths or ``(name, content)``
        pairs) whose *contents* participate in cache identity, in addition to
        quoted user headers reachable through ``include_paths``.
    """

    def __init__(
            self,
            cuda_code: str,
            kernel_name: str,
            grid: Tuple[int, int, int],
            block: Tuple[int, int, int],
            shared_mem_bytes: int = 0,
            arch: Optional[str] = None,
            cache_dir: Optional[str] = None,
            verbose: bool = False,
            include_paths: Optional[List[Union[str, Path]]] = None,
            cluster_shape: Optional[Tuple[int, int, int]] = None,
            cta_group_shape: Optional[Tuple[int, int, int]] = None,
            extra_nvcc_options: Optional[List[str]] = None,
            allow_non_portable_cluster: bool = False,
            signature: Any = None,
            *,
            runtime: Optional[CUDARuntime] = None,
            owns_runtime: Optional[bool] = None,
            dependencies: Sequence[Any] = (),
    ):
        if cluster_shape is not None:
            cluster_shape = tuple(int(dim) for dim in cluster_shape)
            if len(cluster_shape) != 3 or any(dim <= 0 for dim in cluster_shape):
                raise ValueError("cluster_shape must contain three positive dimensions, got {cluster_shape}")
            if cluster_shape[0] * cluster_shape[1] * cluster_shape[2] > 16:
                raise ValueError(f"cluster_shape={cluster_shape} exceeds the supported 16-CTA cluster limit")
        if cta_group_shape is not None:
            cta_group_shape = tuple(int(dim) for dim in cta_group_shape)
            if cta_group_shape not in {(1, 1, 1), (2, 1, 1)}:
                raise ValueError(f"cta_group currently supports only (1, 1, 1) and (2, 1, 1), got {cta_group_shape}")
            if cluster_shape is None:
                raise ValueError("cta_group_shape requires cluster_shape")
            if any(c % g for c, g in zip(cluster_shape, cta_group_shape)):
                raise ValueError(f"cta_group_shape {cta_group_shape} must divide cluster_shape {cluster_shape}")

        if signature is not None:
            # Imported lazily to keep construction light and avoid cycles.
            from little_kernel.runtime.abi import KernelSignature

            if not isinstance(signature, KernelSignature):
                raise TypeError("signature must be a little_kernel.runtime.abi.KernelSignature "
                                "or None")

        self.cuda_code = cuda_code
        self.kernel_name = kernel_name
        self.grid = grid
        self.block = block
        self.shared_mem_bytes = shared_mem_bytes
        self.arch = arch
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.include_paths = include_paths
        self.cluster_shape = cluster_shape
        self.cta_group_shape = cta_group_shape
        self.extra_nvcc_options = list(extra_nvcc_options) if extra_nvcc_options else []
        self.allow_non_portable_cluster = allow_non_portable_cluster
        self.signature = signature
        self.dependencies = tuple(dependencies)

        # Runtime ownership: an injected runtime is borrowed/shared unless the
        # caller explicitly declares ownership.
        self._runtime = runtime
        if owns_runtime is None:
            self._owns_runtime = runtime is None
        else:
            self._owns_runtime = bool(owns_runtime)

        self._module = None
        self._launcher = None
        self._cubin_data = None
        self._build_identity: Optional[BuildIdentity] = None
        self._closed = False
        self._closing = False
        self._lifecycle_lock = threading.RLock()

    # -- factories ----------------------------------------------------------

    @classmethod
    def from_artifact(
            cls,
            artifact: Any,
            grid: Tuple[int, int, int],
            block: Tuple[int, int, int],
            shared_mem_bytes: int = 0,
            cache_dir: Optional[str] = None,
            verbose: bool = False,
            include_paths: Optional[List[Union[str, Path]]] = None,
            extra_nvcc_options: Optional[List[str]] = None,
            *,
            runtime: Optional[CUDARuntime] = None,
            dependencies: Sequence[Any] = (),
    ) -> "CompiledKernel":
        """Build a launchable kernel from a :class:`CompilationArtifact`.

        Uses ``artifact.source``, its entry-point name (``kernel_name`` or
        ``name``), typed ``signature`` and resolved ``target``. Options stored
        in ``artifact.metadata['options']`` are merged with
        ``extra_nvcc_options`` (explicit caller options win and cannot be
        silently dropped).
        """
        source = artifact.source
        name = getattr(artifact, "kernel_name", None) or getattr(artifact, "name", None)
        if not name:
            raise ValueError("artifact has no kernel_name/name")
        target = getattr(artifact, "target", None)
        arch = getattr(target, "arch", target)

        artifact_options = []
        metadata = getattr(artifact, "metadata", None)
        if metadata:
            raw_options = metadata.get("options", ())
            if raw_options:
                artifact_options = list(raw_options)
        options = artifact_options + list(extra_nvcc_options or [])

        return cls(
            source,
            name,
            grid,
            block,
            shared_mem_bytes=shared_mem_bytes,
            arch=arch,
            cache_dir=cache_dir,
            verbose=verbose,
            include_paths=include_paths,
            extra_nvcc_options=options or None,
            signature=getattr(artifact, "signature", None),
            runtime=runtime,
            dependencies=dependencies,
        )

    # -- runtime ownership --------------------------------------------------

    @property
    def runtime(self) -> CUDARuntime:
        with self._lifecycle_lock:
            self._check_open()
            if self._runtime is None:
                self._runtime = CUDARuntime()
                self._owns_runtime = True
            return self._runtime

    @property
    def owns_runtime(self) -> bool:
        """True when this kernel created (and may close) its runtime."""
        return self._owns_runtime

    # -- build identity / cache --------------------------------------------

    def _resolve_arch(self) -> str:
        """Return the resolved target arch; never the literal ``auto``."""
        if self.arch is None:
            from little_kernel.backends.nv.common.runtime.target import resolve_target

            device = getattr(self._runtime, "device", None)
            return resolve_target(device=device).arch
        target = self.arch
        arch = getattr(target, "arch", target)
        return str(arch)

    def build_identity(self) -> BuildIdentity:
        """Compute (and cache) the complete immutable identity of this build."""
        if self._build_identity is None:
            self._build_identity = compute_build_identity(
                kernel_name=self.kernel_name,
                source=self.cuda_code,
                target_arch=self._resolve_arch(),
                options=self.extra_nvcc_options,
                include_paths=self.include_paths,
                extra_dependencies=self.dependencies,
            )
        return self._build_identity

    def _compile(self) -> bytes:
        # Cache disabled: never resolve a target, probe a Driver or identify a
        # toolchain just for keying. compile_cuda owns target resolution.
        if not self.cache_dir:
            selected_device = getattr(self._runtime, "device", None)
            device_options = {"device": selected_device} if selected_device is not None else {}
            return compile_cuda(
                self.cuda_code,
                self.kernel_name,
                arch=self.arch,
                target_format="cubin",
                options=self.extra_nvcc_options,
                verbose=self.verbose,
                include_paths=self.include_paths,
                **device_options,
            )

        identity = self.build_identity()
        cache = AtomicCubinCache(self.cache_dir)

        def _build() -> bytes:
            if self.verbose:
                print(f"Compiling kernel: {self.kernel_name} ({identity.cache_key})")
            cubin = compile_cuda(
                self.cuda_code,
                self.kernel_name,
                arch=identity.target_arch,
                target_format="cubin",
                options=self.extra_nvcc_options,
                verbose=self.verbose,
                include_paths=self.include_paths,
            )
            if not isinstance(cubin, bytes) or not cubin:
                raise RuntimeError("compile_cuda produced empty cubin")
            return cubin

        cubin = cache.get_or_build(identity, _build, source=self.cuda_code)
        if self.verbose:
            print(f"Cached/loaded binary: {cache.cubin_path(identity)}")
        return cubin

    # -- module/launcher ----------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"CompiledKernel {self.kernel_name!r} has been closed")
        if self._closing:
            raise RuntimeError(
                f"CompiledKernel {self.kernel_name!r} is closing; retry close() to release remaining resources")

    @property
    def module(self):
        with self._lifecycle_lock:
            self._check_open()
            if self._module is None:
                cubin = self._compile()
                self._cubin_data = cubin
                self._module = self.runtime.load_module(cubin)
            return self._module

    @property
    def launcher(self) -> KernelLauncher:
        with self._lifecycle_lock:
            return self._get_launcher()

    def _get_launcher(self) -> KernelLauncher:
        self._check_open()
        if self._launcher is None:
            _ = self.module
            # A known signature must reach the merged RF04 launcher. If the
            # installed launcher cannot accept it, that is an integration
            # incompatibility: fail loudly instead of silently launching an
            # explicitly typed kernel through value-guessing fallback.
            if self.signature is not None and not _LAUNCHER_ACCEPTS_SIGNATURE:
                raise RuntimeError("this KernelLauncher does not accept the trailing typed "
                                   "'signature' argument required by RF01/RF04; the runtime "
                                   "worker's merged KernelLauncher(runtime, module, name, "
                                   "shared_mem, cubin, ..., signature=None) contract is "
                                   "required to launch an explicitly signed kernel (got "
                                   f"kernel {self.kernel_name!r}). Pass no signature for the "
                                   "explicit untyped path, or use the merged runtime.")
            kwargs = dict(
                cluster_shape=self.cluster_shape,
                allow_non_portable_cluster=self.allow_non_portable_cluster,
            )
            if _LAUNCHER_ACCEPTS_SIGNATURE:
                kwargs["signature"] = self.signature
            self._launcher = KernelLauncher(
                self.runtime,
                self.module,
                self.kernel_name,
                self.shared_mem_bytes,
                self._cubin_data,
                **kwargs,
            )
        return self._launcher

    def __call__(self, *args, stream: Optional[Any] = None, grid: Optional[Tuple[int, int, int]] = None):
        self._check_open()
        g = grid if grid is not None else self.grid
        if self.cluster_shape is not None:
            if any(grid_dim % cluster_dim for grid_dim, cluster_dim in zip(g, self.cluster_shape)):
                raise ValueError(f"grid {g} must be divisible by cluster_shape {self.cluster_shape}")
        self.launcher(g, self.block, *args, stream=stream, cluster_shape=self.cluster_shape)

    def synchronize(self, stream: Optional[Any] = None):
        self._check_open()
        self.runtime.synchronize(stream)

    def max_active_clusters(self) -> int:
        """Return the maximum resident clusters for this compiled kernel."""
        return self.launcher.max_active_clusters(self.block)

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Release this kernel's loaded module and, if owned, its runtime.

        Release order mirrors resource ownership:

        1. the :class:`KernelLauncher` is closed first -- the typed
           launcher registers a live module user at construction and only its
           ``close()`` detaches it;
        2. only then is the module unloaded through the runtime's
           ``unload_module`` hook (which may refuse while a user is attached);
        3. the runtime is closed only when this kernel owns it -- a borrowed or
           shared runtime is never destroyed here.

        The operation is idempotent and thread-safe. If any release step fails,
        remaining owned resources stay tracked so the caller can retry
        ``close()``. Successfully released resources are never released twice;
        launching is disabled once shutdown begins.
        """
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closing = True

            launcher = self._launcher
            module = self._module
            runtime = self._runtime
            owns_runtime = self._owns_runtime

            # 1. Detach the launcher's live module user first.
            if launcher is not None:
                launcher_close = getattr(launcher, "close", None)
                if callable(launcher_close):
                    launcher_close()

            # 2. Unload the module now that no launcher user remains.
            if module is not None and runtime is not None:
                unload = getattr(runtime, "unload_module", None)
                if callable(unload):
                    unload(module)
            # These releases succeeded. If context shutdown fails later,
            # retry only the remaining context rather than unloading twice.
            self._launcher = None
            self._module = None

            # 3. Close only a runtime we own; borrowed/shared survives.
            if owns_runtime and runtime is not None:
                close = getattr(runtime, "close", None)
                if callable(close):
                    close()

            # Mark fully closed only after the remaining context is released.
            self._closed = True
            self._closing = False

    def __enter__(self) -> "CompiledKernel":
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False


def compile_and_load_kernel(
    cuda_code: str,
    kernel_name: str,
    grid: Tuple[int, int, int],
    block: Tuple[int, int, int],
    shared_mem_bytes: int = 0,
    arch: Optional[str] = None,
    cache_dir: Optional[str] = None,
    verbose: bool = False,
    include_paths: Optional[List[Union[str, Path]]] = None,
) -> CompiledKernel:
    return CompiledKernel(
        cuda_code,
        kernel_name,
        grid,
        block,
        shared_mem_bytes,
        arch,
        cache_dir,
        verbose,
        include_paths,
    )
