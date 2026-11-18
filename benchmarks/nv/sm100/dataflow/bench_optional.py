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
"""Optional benchmark references; paths contain import roots, separated by os.pathsep."""

import importlib
import importlib.util
import os
import sys


def add_reference_argument(parser):
    parser.add_argument(
        "--references",
        choices=("auto", "required", "none"),
        default="auto",
        help="auto: use available external references; required: fail if missing; none: PyTorch checks only",
    )


def optional_import(module, path_env, default_paths=(), *, mode="auto"):
    if mode == "none":
        return None
    paths = os.environ.get(path_env)
    if paths is not None:
        paths = paths.split(os.pathsep)
    else:
        paths = () if importlib.util.find_spec(module.split(".")[0]) is not None else default_paths
    for path in reversed(paths):
        path = os.path.expanduser(str(path))
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    try:
        return importlib.import_module(module)
    except (ImportError, OSError) as exc:
        if mode == "required":
            raise RuntimeError(f"{module} is required; configure {path_env}: {exc}") from exc
        print(f"REFERENCE UNAVAILABLE: {module}; configure {path_env}: {exc}", flush=True)
        return None
