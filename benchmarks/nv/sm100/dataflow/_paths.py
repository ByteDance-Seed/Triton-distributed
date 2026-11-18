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
"""Canonical repository paths for dataflow benchmark input programs."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PYTHON_INCLUDE_ROOT = REPO_ROOT / "python"
EXAMPLES_DATAFLOW_DIR = REPO_ROOT / "examples/nv/sm100"

# performance_id -> (canonical examples/nv/sm100 name, historical filename)
ROLE_EXAMPLE_SOURCES: dict[str, tuple[str, str]] = {
    "gemm": ("gemm_sm100.py", "v2_gemm_sm100.py"),
    "gemm_a2a": ("gemm_a2a_sm100.py", "v2_gemm_a2a_sm100.py"),
    "a2a_gemm": ("a2a_gemm_sm100.py", "v2_a2a_gemm_sm100.py"),
    "nvfp4": ("nvfp4_gemm.py", "v2_nvfp4_gemm.py"),
    "mxfp4": ("mxfp4_gemm.py", "v2_mxfp4_gemm.py"),
    "mxfp8": ("mxfp8_gemm.py", "v2_mxfp8_gemm.py"),
    "w4a8_groupgemm": ("mxfp_w4a8_groupgemm.py", "v2_mxfp_w4a8_groupgemm.py"),
    "fa4": ("fa4_attention.py", "v2_fa4.py"),
    "gdn": ("gated_delta_net.py", "v2_gdn.py"),
    "groupgemm": ("grouped_gemm.py", "v2_groupgemm.py"),
    "ag_gemm": ("ag_gemm_sm100.py", "v2_ag_gemm_sm100.py"),
    "gemm_rs": ("gemm_rs_sm100.py", "v2_gemm_rs_sm100.py"),
    "dispatch_swiglu": (
        "dispatch_group_gemm_swiglu_fused_sm100.py",
        "v2_dispatch_group_gemm_swiglu_fuse_sm100.py",
    ),
}


def role_example_source(performance_id: str) -> str:
    """Read the maintained input program, never an in-package copy."""
    return role_example_path(performance_id).read_text(encoding="utf-8")


def role_example_path(performance_id: str) -> Path:
    """Return the maintained input program for a tracked benchmark."""
    canonical_name, _historical_name = ROLE_EXAMPLE_SOURCES[performance_id]
    path = EXAMPLES_DATAFLOW_DIR / canonical_name
    if not path.is_file():
        raise FileNotFoundError(f"Missing benchmark input program: {path}")
    return path
