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
"""Hardware-specific source toolchain implementation."""
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union
from little_kernel.backends.nv.common.runtime.target import resolve_target


def get_nvcc_path() -> str:
    """Get the path to nvcc compiler."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc not found. Please ensure CUDA toolkit is installed and nvcc is in PATH.")
    return nvcc


def get_cuda_arch(device: Any = None) -> str:
    """Return the canonical CUDA architecture string for one selected device.

    Compatibility wrapper around :func:`little_kernel.backends.nv.common.runtime.target.resolve_target`.
    Discovery inspects the current Torch device (or ``device``) and parses
    major/minor compute capability as integers. It deliberately raises
    :class:`~little_kernel.backends.nv.common.runtime.target.TargetResolutionError` instead of
    silently guessing an architecture when no capability can be discovered;
    callers that need off-host compilation must pass an explicit architecture
    via ``resolve_target(arch=...)`` or ``compile_cuda(arch=...)``.
    """
    return resolve_target(device=device).arch


def default_include_paths() -> List[str]:
    """User include directories implicitly added when none are supplied.

    This is the single owner of compile_cuda's implicit search policy (the
    packaged CUTLAS headers): both compilation and the RF02 build identity
    call it, so a change to a header the compiler searches *implicitly*
    cannot evade the cache key.
    """
    compiler_dir = Path(__file__).parent.resolve()  # noqa: F841
    # Wheel installation keeps little_kernel/ under this directory. Internal
    # generated includes must resolve without a repository checkout or an
    # extra caller-supplied -Ipython flag.
    package_root = Path(__import__("little_kernel").__file__).resolve().parent.parent
    project_root = package_root.parent
    default_cute_path = project_root / "3rdparty" / "cutlass" / "include"
    paths = [str(package_root)]
    if default_cute_path.exists():
        paths.append(str(default_cute_path))
    return paths


def effective_include_paths(include_paths: Optional[List[Union[str, Path]]] = None, ) -> List[str]:
    """Return the ordered include directories compile_cuda emits as ``-I``.

    Explicit paths are resolved relative to the caller before NVCC enters its
    private build directory. They keep caller order (missing paths are skipped,
    matching nvcc's behavior); ``None`` selects :func:`default_include_paths`.
    The order is exactly the order the compiler searches and emits flags, so
    build identity must not sort it.
    """
    if include_paths is None:
        return default_include_paths()
    effective: List[str] = []
    for path in include_paths:
        path_str = os.path.realpath(os.path.abspath(os.fspath(path)))
        if os.path.exists(path_str):
            effective.append(path_str)
    package_root = str(Path(__import__("little_kernel").__file__).resolve().parent.parent)
    if package_root not in effective:
        effective.append(package_root)
    return effective


def compile_cuda(
    code: str,
    kernel_name: str,
    arch: Optional[str] = None,
    target_format: Literal["ptx", "cubin"] = "cubin",
    options: Optional[List[str]] = None,
    verbose: bool = False,
    cache_dir: Optional[str] = None,
    include_paths: Optional[List[Union[str, Path]]] = None,
    device: Any = None,
) -> bytes:
    """
    Compile CUDA code to PTX or CUBIN using nvcc.

    Parameters
    ----------
    code : str
        The CUDA source code.
    kernel_name : str
        Name of the kernel function.
    arch : Optional[str]
        Target architecture (e.g., "sm_90a"). If None, auto-detects.
    target_format : Literal["ptx", "cubin"]
        Output format: PTX or CUBIN.
    options : Optional[List[str]]
        Additional nvcc options.
    verbose : bool
        Print compilation output.
    cache_dir : Optional[str]
        Directory to cache compiled binaries.
    include_paths : Optional[List[Union[str, Path]]]
        List of include paths to add to nvcc command. Each element can be
        a string or pathlib.Path. If None, uses default paths.
    device : Any
        Selected device for automatic discovery when ``arch`` is None (the
        current Torch device by default; an int or ``'cuda:N'`` may select a
        specific visible device). An explicit ``arch`` is a cross-compilation
        request and never probes a Driver.

    Returns
    -------
    bytes
        Compiled binary (PTX or CUBIN).
    """
    if target_format not in ("ptx", "cubin"):
        raise ValueError(f"target_format must be 'ptx' or 'cubin', got {target_format!r}")
    target = resolve_target(arch=arch, device=device)

    # Create temporary directory for compilation
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        temp_dir = cache_dir
        cleanup = False
    else:
        temp_dir = tempfile.mkdtemp()
        cleanup = True

    try:
        # Write source code
        source_file = os.path.join(temp_dir, f"{kernel_name}.cu")
        with open(source_file, "w") as f:
            f.write(code)

        # Determine output file
        if target_format == "cubin":
            output_file = os.path.join(temp_dir, f"{kernel_name}.cubin")
            nvcc_format = "--cubin"
        else:
            output_file = os.path.join(temp_dir, f"{kernel_name}.ptx")
            nvcc_format = "--ptx"

        # Build nvcc command
        nvcc = get_nvcc_path()

        # Build include path list through the single shared policy so that
        # compilation and cache identity search exactly the same directories.
        include_args = [f"-I{path}" for path in effective_include_paths(include_paths)]
        if verbose and include_paths is not None:
            for path in include_paths:
                path_str = str(path) if isinstance(path, Path) else path
                if not os.path.exists(path_str):
                    print(f"Warning: Include path does not exist: {path_str}")

        # Build arch args from the resolved target (feature suffix deliberate).
        arch_args = target.nvcc_arch_args()

        cmd = [nvcc, nvcc_format, "-O3", "-std=c++17"]
        cmd.extend(arch_args)
        cmd.extend(include_args)
        if options:
            cmd.extend(options)
        cmd.extend(["-o", output_file, source_file])

        if verbose:
            print(f"Compiling with command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir)
        if result.returncode != 0:
            error_msg = f"Compilation failed:\n{result.stderr}\n\nSource code:\n{code}"
            raise RuntimeError(error_msg)

        if verbose and result.stderr:
            print(result.stderr)
        if verbose and result.stdout:
            print(result.stdout)

        with open(output_file, "rb") as f:
            binary = f.read()

        if not binary:
            raise RuntimeError("Compilation produced empty output")

        return binary

    finally:
        if cleanup:
            shutil.rmtree(temp_dir, ignore_errors=True)
