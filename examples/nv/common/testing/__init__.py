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
"""Executable, runnable example fixtures shared by the test stages.

Modules in this package are runnable programs/examples, not production
compiler code.  ``vector_fixtures`` is the single canonical home for the
minimal 16-byte global-vector DSL programs used by:

* the CPU codegen stage (generated CUDA source / PTX opcode invariants),
* the NVCC stage (explicit ``sm_90a`` PTX cross-compilation), and
* the GPU semantic stage (bit-exact round trip and lane-order canaries),

so the three stages can never again disagree about what "valid geometry"
means.  See ``docs/behavioral-fixtures.md`` for the vector-count/tail-policy
contract and the hardware requirements of each stage.
"""
