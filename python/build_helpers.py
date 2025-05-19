import os
import sysconfig
import sys
from pathlib import Path
import shutil


def get_base_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def get_cmake_dir():
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    cmake_dir = Path(get_base_dir()) / "python" / "build" / dir_name
    cmake_dir.mkdir(parents=True, exist_ok=True)
    return cmake_dir


def copy_apply_patches():
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
        target_dir = os.path.join(_3rdparty_triton_dir, relative_path)

        # Create the target directory if it does not exist
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Copy files
        for file in files:
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_dir, file)
            try:
                shutil.copy2(source_file, target_file)
            except Exception:
                shutil.copyfile(source_file, target_file)
            print(f"Copied {source_file} to {target_file}")
