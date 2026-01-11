import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import setuptools
import torch
from torch.utils.cpp_extension import BuildExtension
from glob import glob
import sysconfig
USE_MACA = True if os.getenv('MACA_PATH') else False

# Project directory root
root_path: Path = Path(__file__).resolve().parent
MXSHMEM_HOME = os.environ.get("MXSHMEM_HOME")
PACKAGE_NAME = "pymxshmem"

site_packages_dir = sysconfig.get_path("purelib")
virtual_env = sysconfig.get_config_var('base')
install_so_path = os.path.relpath(site_packages_dir, virtual_env)
install_so_path = os.path.join(install_so_path, PACKAGE_NAME)

def get_package_version():
    return "0.0.1"


def pathlib_wrapper(func):

    def wrapper(*kargs, **kwargs):
        include_dirs, library_dirs, libraries = func(*kargs, **kwargs)
        return map(str, include_dirs), map(str, library_dirs), map(str, libraries)

    return wrapper


@pathlib_wrapper
def mxshmem_deps():
    mxshmem_home = Path(os.environ.get("MXSHMEM_HOME", root_path / "../../../3rdparty/mxshmem/build/src"))
    # include_dirs = [mxshmem_home / "include", "/home/bxiong/mcTriton/third_party/triton_dist/3rdparty/mxshmem/src/include", "/home/bxiong/mcTriton/third_party/triton_dist/3rdparty/mxshmem/src/include_internal"]
    # library_dirs = ["/home/bxiong/mcTriton/third_party/triton_dist/3rdparty/mxshmem/build/src"]
    include_dirs = [mxshmem_home / "include", root_path / "../../../3rdparty/mxshmem/src/include", root_path / "../../../3rdparty/mxshmem/src/include_internal"]
    library_dirs = [mxshmem_home]
    print("include_dirs:", include_dirs)
    libraries = ["mxshmem_host", "mxshmem_device"]
    return include_dirs, library_dirs, libraries


@pathlib_wrapper
def deps():
    maca_path = os.getenv("MACA_PATH")
    include_dirs = [os.path.join(maca_path, "include")]
    library_dirs = [os.path.join(maca_path, "lib"), os.path.join(maca_path, "lib/stubs")]
    libraries = ["mcruntime"]
    return include_dirs, library_dirs, libraries


def setup_pytorch_extension() -> setuptools.Extension:
    """Setup CppExtension for PyTorch support"""
    include_dirs, library_dirs, libraries = [], [], []

    deps = [mxshmem_deps(), deps()]

    for include_dir, library_dir, library in deps:
        include_dirs += include_dir
        library_dirs += library_dir
        libraries += library

    cxx_flags = [
        "-O3",
        "-DTORCH_CUDA=1",
        "-fvisibility=hidden",
        "-Wno-deprecated-declarations",
        "-fdiagnostics-color=always",
    ]
    ld_flags = ["-Wl,--exclude-libs=libnccl_static"]

    from torch.utils.cpp_extension import CUDAExtension

    return CUDAExtension(
        name="_pymxshmem",
        sources=["src/pymxshmem.cc", "src/flush_l2c.cu"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        dlink=True,
        dlink_libraries=["mxshmem_device", "cudart_static"],
        extra_compile_args={"cxx": cxx_flags},
        extra_link_args=ld_flags,
    )


def main():
    packages = setuptools.find_packages(
        where="python",
        include=[
            "pymxshmem",
            "_pymxshmem",
        ],
    )
    print("packages are:", packages,)
    # Configure package
    setuptools.setup(
        name=PACKAGE_NAME,
        version=get_package_version(),
        package_dir={"": "python"},
        packages=packages,
        description="Triton-distributed pymxshmem",
        ext_modules=[setup_pytorch_extension()],
        cmdclass={"build_ext": BuildExtension.with_options(verbose=True)},
        setup_requires=["cmake", "packaging"],
        install_requires=[],
        extras_require={"test": ["numpy"]},
        license_files=("LICENSE", ),
        package_data={
            "python/pymxshmem/lib": ["*.so"],
        },  # only works for bdist_wheel under package
        data_files=[
            (install_so_path, glob(f"{MXSHMEM_HOME}/*.so*")),
        ],
        python_requires=">=3.8",
        include_package_data=True,
    )


if __name__ == "__main__":
    main()
