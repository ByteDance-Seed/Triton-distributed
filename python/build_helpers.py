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
import sysconfig
import sys
import shutil
from pathlib import Path


def get_base_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def get_cmake_dir():
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    cmake_dir = Path(get_base_dir()) / "python" / "build" / dir_name
    cmake_dir.mkdir(parents=True, exist_ok=True)
    return cmake_dir


def create_symlink_rel(target: Path, source: Path, base_dir: Path, dryrun: bool = False):
    """Make source path a symlink pointing to the target path.
    if both source/target under base_dir, create link with relative path.

    why it's tricky for that?

    source maybe file or directory. maybe relative or absolute.
    target maybe file or directory. maybe relative or absolute.
    base_dir maybe relative or absolute.
    """
    # should relative
    base_dir = base_dir.resolve()
    source.parent.mkdir(exist_ok=True, parents=True)
    if source.is_symlink() or source.exists():
        source.unlink()
    if target.resolve().is_relative_to(base_dir) and source.resolve().is_relative_to(base_dir):
        target = target.resolve()
        source = source.resolve()
        target = Path(os.path.relpath(target, source.parent))

    if not dryrun:
        source.symlink_to(target, target_is_directory=target.is_dir())


def copy_file(target: Path, source: Path, base_dir: Path):
    """Make source path a copy of the target path.
    if both source/target under base_dir, create link with relative path.

    why it's tricky for that?

    source maybe file or directory. maybe relative or absolute.
    target maybe file or directory. maybe relative or absolute.
    base_dir maybe relative or absolute.
    """
    # should relative
    base_dir = base_dir.resolve()
    source.parent.mkdir(exist_ok=True, parents=True)
    if source.is_file() or source.is_link():
        source.unlink()
    elif source.is_dir():
        shutil.rmtree(source)
    else:
        raise NotADirectoryError(f"Source {source} is not a file or directory.")

    if target.resolve().is_relative_to(base_dir) and source.resolve().is_relative_to(base_dir):
        target = target.resolve()
        source = source.resolve()
        target = Path(os.path.relpath(target, source.parent))

    if target.is_file() or target.is_symlink():
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, source)
        print(f"Copy file in build helper: {target} -> {source}")
    elif target.is_dir():
        shutil.copytree(target, source, symlinks=True, dirs_exist_ok=False)
        print(f"Copy directory in build helper: {target} -> {source}")

    else:
        raise NotADirectoryError(f"Source {source} is not a file or directory.")


def softlink_apply_patches():
    # Get the directory where the current script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path of the patches/triton directory
    patches_triton_dir = os.path.join(script_dir, '..', 'patches', 'triton')
    # Construct the path of the 3rdparty/triton directory
    _3rdparty_triton_dir = os.path.join(script_dir, '..', '3rdparty', 'triton')

    # Check if the patches/triton directory exists
    if not os.path.exists(patches_triton_dir):
        print(f"Patches directory {patches_triton_dir} does not exist.")
        return

    # Traverse all files and folders in the patches/triton directory
    for root, dirs, files in os.walk(patches_triton_dir):
        # Calculate the relative path
        relative_path = os.path.relpath(root, patches_triton_dir)
        # Construct the path of the target directory
        triton_dir = os.path.join(_3rdparty_triton_dir, relative_path)

        # Create the target directory if it does not exist
        if not os.path.exists(triton_dir):
            os.makedirs(triton_dir)

        # Copy files
        for file in files:
            patch_file = os.path.join(root, file)
            origin_file = os.path.join(triton_dir, file)
            # Check if the source file is a relative path
            base_dir = Path(__file__).parent.parent
            # make origin_file a softlink to pach_file
            create_symlink_rel(Path(patch_file), Path(origin_file), base_dir)


# ------ MACA extension ------
def get_pymaca_project():
    pymaca_dir = os.path.join(get_base_dir(), "3rdparty", "pymaca")
    return pymaca_dir


def get_pymxshmem_project():
    return os.path.join(get_base_dir(), "shmem", "mxshmem_bind", "pymxshmem")


def get_pymaca_cmake_dir(project_name):
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    cmake_dir = Path(get_base_dir()) / "python" / "build" / dir_name / project_name
    cmake_dir.mkdir(parents=True, exist_ok=True)
    return cmake_dir


def get_pymxshmem_cmake_dir(project_name):
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    cmake_dir = Path(get_base_dir()) / "python" / "build" / dir_name / project_name
    cmake_dir.mkdir(parents=True, exist_ok=True)
    return cmake_dir


def maca_patch_jit(triton_jit_path):
    if not os.path.exists(triton_jit_path):
        print(f"Can't find {triton_jit_path}")
        return
    with open(triton_jit_path, "r") as f:
        content = f.read()
    # replace jit.is_dist with always return True
    old_code = "def is_dist(self):\n        return False\n"
    new_code = "def is_dist(self):\n        return True\n"
    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(triton_jit_path, "w") as f:
            f.write(content)
        print(f"Patched {triton_jit_path}: is_dist always returns True")
    else:
        print(f"Can't find is_dist function in {triton_jit_path}")
