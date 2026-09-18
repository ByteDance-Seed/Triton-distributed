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
import platform
import re
import subprocess
from pathlib import Path

import setuptools
from setuptools import setup


def _split_cuda_arch_list(cuda_arch):
    return [token for token in cuda_arch.replace(',', ';').replace(' ', ';').split(';') if token]


def _cuda_arch_major(token):
    arch = token.strip().lower().replace('+ptx', '')
    for prefix in ('compute_', 'sm_', 'sm'):
        if arch.startswith(prefix):
            arch = arch[len(prefix):]
            break
    arch = arch.rstrip('a')
    if arch.isdigit():
        value = int(arch)
        return value // 10 if value >= 10 else value
    return int(float(arch))


def _normalize_cuda_arch_list(cuda_arch):
    arch_tokens = _split_cuda_arch_list(cuda_arch)
    arch_majors = []
    for token in arch_tokens:
        try:
            arch_majors.append(_cuda_arch_major(token))
        except ValueError:
            pass
    if 10 in arch_majors and 9 not in arch_majors:
        arch_tokens.append('9.0')
    return ';'.join(arch_tokens)


def _nccl_pip_dist_names():
    for major in ("13", "12"):
        yield f"nvidia-nccl-cu{major}"
        yield f"nvidia_nccl_cu{major}"


def _is_nccl_root(path: Path) -> bool:
    return ((path / "include" / "nccl.h").is_file() and (path / "include" / "nccl_device.h").is_file()
            and ((path / "lib" / "libnccl.so").is_file() or (path / "lib" / "libnccl.so.2").is_file()))


def _available_cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _nvcc_supports_option(cuda_home: Path, option: str):
    nvcc = cuda_home / "bin" / "nvcc"
    try:
        result = subprocess.run([str(nvcc), "--help"], capture_output=True, text=True, check=False)
    except OSError as e:
        print(f"Unable to query {nvcc} for {option} support: {e}")
        return False
    if result.returncode != 0:
        print(f"Unable to query {nvcc} for {option} support; NVCC --help returned {result.returncode}")
        return False
    return option in result.stdout


def _nvcc_parallel_flag(cuda_home: Path, env_name: str, option: str, default_threads: int):
    value = os.getenv(env_name, str(default_threads)).strip().lower()
    if value in ("", "off", "false", "no"):
        return None
    try:
        threads = int(value)
    except ValueError as e:
        raise RuntimeError(f"{env_name} must be a non-negative integer or one of off/false/no") from e
    if threads < 0:
        raise RuntimeError(f"{env_name} must be non-negative")
    if threads == 1:
        return None

    if not _nvcc_supports_option(cuda_home, option):
        print(f"NVCC does not support {option}; using serial compilation for that dimension")
        return None
    return f"{option}={threads}"


def _nvcc_compress_flag(cuda_home: Path):
    mode = os.getenv("FLASH_COMM_NVCC_COMPRESS_MODE", "size").strip().lower()
    if mode in ("", "off", "false", "no"):
        return None
    valid_modes = {"default", "size", "speed", "balance", "none"}
    if mode not in valid_modes:
        raise RuntimeError(f"FLASH_COMM_NVCC_COMPRESS_MODE must be one of {sorted(valid_modes)}, got {mode!r}")
    option = "--compress-mode"
    if not _nvcc_supports_option(cuda_home, option):
        print(f"NVCC does not support {option}; using its default fatbinary compression")
        return None
    return f"{option}={mode}"


def _internode_hidden_sizes(project_root: Path):
    hidden_sizes_header = project_root / "include" / "flash_comm" / "ep" / "hidden_sizes.h"
    lines = hidden_sizes_header.read_text(encoding="utf-8").splitlines()
    definition = "#define FLASH_COMM_SUPPORTED_HIDDEN_SIZES(OP, ...)"
    definition_pattern = re.compile(r"#define\s+FLASH_COMM_SUPPORTED_HIDDEN_SIZES\(OP, \.\.\.\)\s*\\")
    definition_lines = [index for index, line in enumerate(lines) if definition_pattern.fullmatch(line.strip())]
    if len(definition_lines) != 1:
        raise RuntimeError(f"Expected exactly one {definition} in {hidden_sizes_header}")

    hidden_sizes = []
    terminator_found = False
    for line in lines[definition_lines[0] + 1:]:
        entry = line.strip()
        if entry == "/* X-macro terminator: keep this line last. */":
            terminator_found = True
            break
        match = re.fullmatch(r"OP\((\d+), __VA_ARGS__\)\s*\\", entry)
        if match is None:
            raise RuntimeError(f"Invalid hidden-size entry in {hidden_sizes_header}: {entry!r}")
        hidden_sizes.append(int(match.group(1)))
    if not terminator_found:
        raise RuntimeError(f"Missing X-macro terminator in {hidden_sizes_header}")
    if (not hidden_sizes or any(value <= 0 for value in hidden_sizes) or hidden_sizes != sorted(set(hidden_sizes))):
        raise RuntimeError(
            f"FLASH_COMM_SUPPORTED_HIDDEN_SIZES must contain positive, unique, sorted values, got {hidden_sizes}")
    return hidden_sizes


def _generated_internode_sources(project_root: Path, generated_dir: Path):
    return [(hidden_size, generated_dir / f"internode_cuda_{hidden_size}.cu")
            for hidden_size in _internode_hidden_sizes(project_root)]


def _write_generated_internode_sources(project_root: Path, generated_dir: Path):
    source_specs = _generated_internode_sources(project_root, generated_dir)
    generated_dir.mkdir(parents=True, exist_ok=True)
    expected_paths = {path for _, path in source_specs}
    for stale_path in generated_dir.glob("internode_cuda_*.cu"):
        if stale_path not in expected_paths:
            stale_path.unlink()

    generated_sources = []
    for hidden_size, path in source_specs:
        generated_sources.append(str(path))
        content = ("// Generated by setup.py. Do not edit.\n"
                   "#define FLASH_COMM_INTERNODE_INSTANTIATION\n"
                   "#include \"internode_cuda_impl.cuh\"\n"
                   f"FLASH_COMM_INSTANTIATE_INTERNODE_HIDDEN_SIZE({hidden_size})\n")
        if path.is_file() and path.read_text(encoding="utf-8") == content:
            continue
        temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary_path.write_text(content, encoding="utf-8")
        os.replace(temporary_path, path)
    return generated_sources


def get_extension():
    try:
        from torch.utils.cpp_extension import CUDAExtension
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Building flash_comm requires PyTorch with CUDA available. "
                           "Install torch first, then run `python setup.py bdist_wheel`.") from e

    custom_arch = os.getenv('CUSTOM_CUDA_ARCH')
    if custom_arch:
        cuda_arch = custom_arch
    else:
        cuda_arch = os.getenv('TORCH_CUDA_ARCH_LIST', '9.0')
    cuda_arch = _normalize_cuda_arch_list(cuda_arch)
    os.environ['TORCH_CUDA_ARCH_LIST'] = cuda_arch
    print(f"Using CUDA architecture: {cuda_arch}")

    # Derive max dynamic shared memory per block (bytes) from target arch.
    arch_majors = [_cuda_arch_major(token) for token in _split_cuda_arch_list(cuda_arch)]
    arch_majors = [major for major in arch_majors if major is not None]
    if arch_majors and min(arch_majors) < 9:
        raise RuntimeError("FlashComm requires CUDA arch 9.0 or newer")
    max_major = max(arch_majors) if arch_majors else 9
    max_smem_kb = {9: 227, 10: 227}.get(max_major, 227)
    max_smem_bytes = max_smem_kb * 1024
    print(f"Max dynamic shared memory per block: {max_smem_kb} KB "
          f"({max_smem_bytes} bytes) for SM major={max_major}")

    cur_dir = Path(__file__).resolve().parent
    sources = [
        os.path.join("csrc", "quantization", "kernels", "mxfp8_cuda.cu"),
        os.path.join("csrc", "quantization", "mxfp8.cpp"),
        os.path.join("csrc", "ep", "kernels", "intranode_cuda.cu"),
        os.path.join("csrc", "ep", "kernels", "internode_cuda.cu"),
        os.path.join("csrc", "ep", "kernels", "chunk_plan_cuda.cu"),
        os.path.join("csrc", "ep", "intranode.cpp"),
        os.path.join("csrc", "ep", "internode.cpp"),
        os.path.join("csrc", "ep", "chunk_plan.cpp"),
        os.path.join("csrc", "bindings.cpp"),
        os.path.join("csrc", "buffer", "pybind.cpp"),
        os.path.join("csrc", "buffer", "nccl_gin.cpp"),
        os.path.join("csrc", "buffer", "shareable_block.cpp"),
        os.path.join("csrc", "buffer", "symmetric_memory.cpp"),
        os.path.join("csrc", "buffer", "nccl_symmetric_memory.cpp"),
    ]
    include_dirs = [str(cur_dir / "include"), str(cur_dir / "csrc" / "ep" / "kernels")]
    libraries = []

    def _nccl_pip_root():
        import site
        for dist_name in _nccl_pip_dist_names():
            try:
                from importlib.metadata import distribution
                dist = distribution(dist_name)
            except Exception:
                continue
            for entry in dist.files or []:
                rel = str(entry).replace("\\", "/")
                if rel.endswith("lib/libnccl.so.2") or rel.endswith("include/nccl_device.h"):
                    return Path(dist.locate_file(entry)).parent.parent
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            root = Path(sp) / "nvidia" / "nccl"
            if (root / "lib" / "libnccl.so.2").is_file():
                return root
        return None

    custom_nccl_home = os.environ.get("CUSTOM_NCCL_HOME", "")
    nccl_home = os.environ.get("NCCL_HOME", "")
    nccl_candidates = []
    for source, explicit_home in (("CUSTOM_NCCL_HOME", custom_nccl_home), ("NCCL_HOME", nccl_home)):
        if explicit_home:
            nccl_candidates.append((source, Path(explicit_home).expanduser()))
    pip_root = _nccl_pip_root()
    if pip_root is not None:
        nccl_candidates.append(("nvidia-nccl", pip_root))
    nccl_root = None
    nccl_source = None
    nccl_link_args = []
    for source, cand in nccl_candidates:
        if cand is not None and _is_nccl_root(cand):
            nccl_root = cand
            nccl_source = source
            break
    if nccl_root is not None:
        include_dirs.append(str(nccl_root / "include"))
        library_dirs_extra = [str(nccl_root / "lib")]
        if (nccl_root / "lib" / "libnccl.so").is_file():
            libraries.append("nccl")
        else:
            nccl_link_args.append("-l:libnccl.so.2")
        print(f"Using NCCL from {nccl_root} ({nccl_source})")
    else:
        library_dirs_extra = []
        nccl_version = os.environ.get("NCCL_VERSION", os.environ.get("CUSTOM_NCCL_VERSION", "2.30.7"))
        raise RuntimeError("NCCL with device API (nccl_device.h) not found. "
                           f"Install nvidia-nccl-cu13=={nccl_version} or nvidia-nccl-cu12=={nccl_version}, "
                           "or set NCCL_HOME/CUSTOM_NCCL_HOME to headers+lib.")
    # Make `-lcuda` resolvable on both x86_64 and aarch64.
    # On many systems only the CUDA stub has `libcuda.so`, while the driver ships `libcuda.so.1`.
    cuda_home = Path(os.getenv("CUDA_HOME", "/usr/local/cuda"))
    arch = platform.machine()
    # CUDA targets naming differs across distros:
    # - x86_64:   targets/x86_64-linux
    # - aarch64:  targets/sbsa-linux (common on ARM servers) or targets/aarch64-linux
    if arch in ("aarch64", "arm64"):
        if (cuda_home / "targets" / "sbsa-linux").exists():
            target = "sbsa-linux"
        else:
            target = "aarch64-linux"
    else:
        target = "x86_64-linux"

    candidate_lib_dirs = [
        cuda_home / "lib64",
        cuda_home / "compat",
        cuda_home / "targets" / target / "lib" / "stubs",
        cuda_home / "targets" / target / "lib",
        Path("/usr/lib/aarch64-linux-gnu"),
        Path("/usr/lib/x86_64-linux-gnu"),
    ]
    library_dirs = library_dirs_extra + [str(p) for p in candidate_lib_dirs if p and p.exists()]
    extra_compile_args = {}
    extra_link_args = nccl_link_args + ['-lcuda', '-lcudart']
    if nccl_root is not None:
        extra_compile_args.setdefault("cxx", []).append("-DNCCL_OS_LINUX")
        extra_compile_args.setdefault("nvcc", []).append("-DNCCL_OS_LINUX")
    smem_define = f"-DFLASH_COMM_MAX_SMEM_BYTES={max_smem_bytes}"
    nvcc_flags = [
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-Xcompiler",
        "-fPIC,-fvisibility=hidden",
        "-O3",
        smem_define,
    ]
    compress_flag = _nvcc_compress_flag(cuda_home)
    if compress_flag is not None:
        nvcc_flags.append(compress_flag)
        print(f"Using NVCC fatbinary compression: {compress_flag}")
    num_archs = len(_split_cuda_arch_list(cuda_arch))
    available_cpus = _available_cpu_count()
    # Generated internode instances plus internode, intranode, and standalone
    # quantization CUDA translation units.
    cuda_tu_count = len(_internode_hidden_sizes(cur_dir)) + 3
    threads_per_tu = max(1, (available_cpus + cuda_tu_count - 1) // cuda_tu_count)
    arch_threads = min(num_archs, threads_per_tu)
    arch_compile_flag = _nvcc_parallel_flag(cuda_home, "FLASH_COMM_NVCC_THREADS", "--threads", arch_threads)
    if arch_compile_flag is not None:
        nvcc_flags.append(arch_compile_flag)
        print(f"Using NVCC architecture parallelism: {arch_compile_flag}")
    cxx_flags = [
        "-std=c++17",
        "-fvisibility=hidden",
        smem_define,
    ]
    extra_compile_args["nvcc"] = nvcc_flags
    extra_compile_args["cxx"] = cxx_flags

    print(f"include_dirs={include_dirs}")
    print(f"library_dirs={library_dirs}")
    print(f"extra_link_args={extra_link_args}")
    print(f"sources={sources}")
    print(f"extra_compile_args={extra_compile_args}")

    extension = CUDAExtension(name="flash_comm._C", include_dirs=include_dirs, library_dirs=library_dirs,
                              libraries=libraries, sources=sources, extra_compile_args=extra_compile_args,
                              extra_link_args=extra_link_args)

    return extension


def _lazy_build_ext():
    from torch.utils.cpp_extension import BuildExtension

    class FlashCommBuildExtension(BuildExtension):

        def build_extensions(self):
            project_root = Path(__file__).resolve().parent
            build_command = self.get_finalized_command("build")
            build_base = Path(build_command.build_base)
            if not build_base.is_absolute():
                build_base = project_root / build_base
            generated_dir = build_base / "generated" / "internode"
            generated_sources = _write_generated_internode_sources(project_root, generated_dir)
            print(f"Generated {len(generated_sources)} internode CUDA instantiation sources in {generated_dir}")
            try:
                generated_sources = [str(Path(source).relative_to(project_root)) for source in generated_sources]
            except ValueError:
                pass
            target_extensions = [extension for extension in self.extensions if extension.name == "flash_comm._C"]
            if len(target_extensions) != 1:
                raise RuntimeError(f"Expected exactly one flash_comm._C extension, got {len(target_extensions)}")
            extension = target_extensions[0]
            extension.sources.extend(source for source in generated_sources if source not in extension.sources)
            super().build_extensions()

    return FlashCommBuildExtension


def _get_ext_modules():
    return [get_extension()]


def _get_cmdclass():
    return {"build_ext": _lazy_build_ext()}


setup(
    name="flash_comm",
    version="0.0.1",
    description="FlashComm: CUDA communication library",
    package_dir={"": "python"},
    packages=setuptools.find_packages(where="python"),
    include_package_data=True,
    zip_safe=False,
    ext_modules=_get_ext_modules(),
    cmdclass=_get_cmdclass(),
)
