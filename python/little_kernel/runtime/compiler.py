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

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Literal, Union


def get_nvcc_path() -> str:
    """Get the path to nvcc compiler."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc not found. Please ensure CUDA toolkit is installed and nvcc is in PATH.")
    return nvcc


def get_cuda_arch() -> str:
    """Auto-detect CUDA architecture from current GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            # Get compute capability
            props = torch.cuda.get_device_properties(0)
            major, minor = props.major, props.minor
            arch = f"sm_{major}{minor}"
            if major >= 9:
                arch += "a"  # For sm_90 and above
            return arch
    except ImportError:
        pass

    # Fallback: try nvidia-smi
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True,
                                text=True, check=True)
        cap = result.stdout.strip().split(".")[0]
        major = int(cap[0])
        minor = int(cap[1]) if len(cap) > 1 else 0
        arch = f"sm_{major}{minor}"
        if major >= 9:
            arch += "a"
        return arch
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass

    # Default to sm_90a if detection fails
    return "sm_90a"


def compile_cuda(
    code: str,
    kernel_name: str,
    arch: Optional[str] = None,
    target_format: Literal["ptx", "cubin"] = "cubin",
    options: Optional[List[str]] = None,
    verbose: bool = False,
    cache_dir: Optional[str] = None,
    include_paths: Optional[List[Union[str, Path]]] = None,
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
    
    Returns
    -------
    bytes
        Compiled binary (PTX or CUBIN).
    """
    if arch is None:
        arch = get_cuda_arch()

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

        # Build include path list
        include_args = []
        if include_paths is not None:
            for path in include_paths:
                path_str = str(path) if isinstance(path, Path) else path
                if os.path.exists(path_str):
                    include_args.append(f"-I{path_str}")
                elif verbose:
                    print(f"Warning: Include path does not exist: {path_str}")
        else:
            compiler_dir = Path(__file__).parent.resolve()
            project_root = compiler_dir.parent.parent.parent
            default_cute_path = project_root / "3rdparty" / "cutlass" / "include"
            possible_cute_paths = [str(default_cute_path)]
            for path in possible_cute_paths:
                if os.path.exists(path):
                    include_args.append(f"-I{path}")
                    break

        # Build arch args
        arch_args = []
        if arch.endswith("a"):
            # For sm_90a, sm_100a, etc. - need -gencode with compute_XXa
            arch_args.extend(["-gencode", f"arch=compute_{arch.replace('sm_', '')},code={arch}"])
        else:
            arch_args.append(f"-arch={arch}")

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
