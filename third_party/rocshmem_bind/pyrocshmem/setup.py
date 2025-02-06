import os
from pathlib import Path
from typing import Tuple

import setuptools
from torch.utils.cpp_extension import BuildExtension

# Project directory root
root_path: Path = Path(__file__).resolve().parent
PACKAGE_NAME = "pyrocshmem"


def hip_version() -> Tuple[int, ...]:
    return tuple(1, 9)


def get_package_version():
    return "0.0.1"


def pathlib_wrapper(func):

    def wrapper(*kargs, **kwargs):
        include_dirs, library_dirs, libraries = func(*kargs, **kwargs)
        return map(str, include_dirs), map(str, library_dirs), map(str, libraries)

    return wrapper


@pathlib_wrapper
def rocshmem_deps():
    rocshmem_home = Path(os.environ.get("ROCSHMEM_HOME", root_path / "../rocshmem_build/install"))
    include_dirs = [rocshmem_home / "include"]
    library_dirs = [rocshmem_home / "lib"]
    libraries = []
    return include_dirs, library_dirs, libraries


@pathlib_wrapper
def hip_deps():
    """
    hip VERSION: 6.3.42131
    hsa-runtime64 VERSION: 1.14.60300
    amd_comgr VERSION: 2.8.0
    rocrand VERSION: 3.2.0
    hiprand VERSION: 2.11.0
    rocblas VERSION: 4.5.0
    hipblas VERSION: 2.3.0
    hipblaslt VERSION: 0.12.0
    miopen VERSION: 3.3.0
    hipfft VERSION: 1.0.17
    hipsparse VERSION: 3.1.2
    rccl VERSION: 2.21.5
    rocprim VERSION: 3.3.0
    hipcub VERSION: 3.3.0
    rocthrust VERSION: 3.3.0
    hipsolver VERSION: 2.3.0
    hiprtc VERSION: 6.3.42131
    """
    hip_home = Path(os.environ.get("HIP_HOME", "/opt/rocm/"))
    include_dirs = [hip_home / "include"]
    library_dirs = [hip_home / "lib"]
    libraries = [
        "amdhip64", "hiprtc", "hsa-runtime64", "hipfft", "amd_comgr", "rocrand", "hiprand", "rocblas", "hipblaslt",
        "hipfft", "hipsparse", "rccl", "hipsolver"
    ]
    return include_dirs, library_dirs, libraries


@pathlib_wrapper
def mpi_deps():
    mpi_home = root_path / "../ompi_build/install/ompi/"
    include_dirs = [mpi_home / "include"]
    library_dirs = [mpi_home / "lib"]
    libraries = ["mpi"]
    return include_dirs, library_dirs, libraries


def setup_pytorch_extension() -> setuptools.Extension:
    """Setup CppExtension for PyTorch support"""
    include_dirs, library_dirs, libraries = [], [], []

    deps = [hip_deps(), mpi_deps(), rocshmem_deps()]

    for include_dir, library_dir, library in deps:
        include_dirs += include_dir
        library_dirs += library_dir
        libraries += library

    print(f"include_dirs={include_dirs}")
    print(f"libraries={libraries}")

    # Compiler flags
    cxx_flags = [
        "-O3",
        "-DTORCH_ROCM=1",
        "-fvisibility=hidden",
        "-Wno-deprecated-declarations",
        "-fdiagnostics-color=always",
    ]
    ld_flags = ["--hip-link", "-fgpu-rdc", "-lrocshmem"]

    from torch.utils.cpp_extension import CppExtension

    return CppExtension(
        name="_pyrocshmem",
        sources=["src/pyrocshmem.cc"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        dlink=True,
        dlink_libraries=[],
        extra_compile_args={"cxx": cxx_flags, "hipcc": ["-fgpu-rdc"]},
        extra_link_args=ld_flags,
    )


"""
FIXME: hipcc not support `-v` option (requires `--version`, which is not compitable with torch cpp_extension)
TORCH_DONT_CHECK_COMPILER_ABI=1
"""


class HipBuildExtension(BuildExtension):

    def build_extensions(self):
        self.compiler.compiler_so = [os.environ.get('CXX', 'hipcc')]
        super().build_extensions()


def main():
    packages = setuptools.find_packages(
        where="python",
        include=[
            "pyrocshmem",
            "_pyrocshmem",
        ],
    )
    # Configure package
    setuptools.setup(
        name=PACKAGE_NAME,
        version=get_package_version(),
        package_dir={"": "python"},
        packages=packages,
        description="Dist-Triton-Pyrocshmem library",
        ext_modules=[setup_pytorch_extension()],
        cmdclass={"build_ext": HipBuildExtension},
        #setup_requires=["torch", "cmake", "packaging"],
        setup_requires=["cmake", "packaging"],
        #install_requires=["torch"],
        install_requires=[],
        #extras_require={"test": ["torch", "numpy"]},
        extras_require={"test": ["numpy"]},
        license_files=("LICENSE", ),
        package_data={
            "python/pyrocshmem/lib": ["*.so"],
        },  # only works for bdist_wheel under package
        python_requires=">=3.8",
        include_package_data=True,
    )


if __name__ == "__main__":
    main()
