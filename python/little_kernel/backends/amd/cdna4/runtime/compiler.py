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


def get_hipcc_path() -> str:
    """Find hipcc through HIPCC, PATH or an explicitly configured SDK root."""
    env = os.environ.get("HIPCC")
    if env:
        return env
    found = shutil.which("hipcc")
    if found:
        return found
    for root in (os.environ.get("ROCM_PATH"), os.environ.get("HIP_PATH")):
        if root and os.path.exists(os.path.join(root, "bin", "hipcc")):
            return os.path.join(root, "bin", "hipcc")
    raise RuntimeError("hipcc not found: set HIPCC or ROCM_PATH, or add hipcc to PATH")


def get_hip_arch() -> str:
    """AMDGPU target of device 0, e.g. "gfx950" ($LK_HIP_ARCH overrides).

    The ``ARCH`` variable that ROCm images export is deliberately not consulted: it often names a
    different GPU than the one present.
    """
    env = os.environ.get("LK_HIP_ARCH")
    if env:
        return env
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except (ImportError, AttributeError, RuntimeError):
        pass
    try:
        out = subprocess.run(["rocm_agent_enumerator", "-name"], capture_output=True, text=True, timeout=10)
        for name in out.stdout.split():
            if name.startswith("gfx") and name != "gfx000":
                return name
    except (OSError, subprocess.SubprocessError):
        pass
    raise RuntimeError("Cannot detect the AMDGPU architecture: set LK_HIP_ARCH (e.g. gfx950)")


def compile_hip(
    code: str,
    kernel_name: str,
    arch: Optional[str] = None,
    options: Optional[List[str]] = None,
    verbose: bool = False,
    include_paths: Optional[List[Union[str, Path]]] = None,
) -> bytes:
    """Compile HIP source to a device code object with ``hipcc --genco`` (loadable by hipModuleLoadData).

    Parameters
    ----------
    arch : Optional[str]
        AMDGPU target (e.g. "gfx950"). If None, the GPU of device 0.
    options : Optional[List[str]]
        Extra hipcc options.
    """
    if arch is None:
        arch = get_hip_arch()
    temp_dir = tempfile.mkdtemp()
    try:
        source_file = os.path.join(temp_dir, f"{kernel_name}.hip")
        output_file = os.path.join(temp_dir, f"{kernel_name}.hsaco")
        with open(source_file, "w") as f:
            f.write(code)
        # the emitter parenthesizes every comparison: `if ((x == 0))` would warn
        cmd = [
            get_hipcc_path(), "--genco", "-O3", "-std=c++17", f"--offload-arch={arch}", "-Wno-unused-result",
            "-Wno-parentheses-equality"
        ]
        for path in include_paths or []:
            cmd.append(f"-I{path}")
        cmd.extend(options or [])
        cmd.extend(["-o", output_file, source_file])
        if verbose:
            print(f"Compiling with command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir)
        if result.returncode != 0:
            raise RuntimeError(f"HIP compilation failed:\n{result.stderr}\n\nSource code:\n{code}")
        if verbose and (result.stderr or result.stdout):
            print(result.stderr + result.stdout)
        with open(output_file, "rb") as f:
            binary = f.read()
        if not binary:
            raise RuntimeError("HIP compilation produced an empty code object")
        return binary
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
