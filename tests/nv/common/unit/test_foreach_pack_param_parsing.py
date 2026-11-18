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
"""Regression tests for packed-foreach ("single") entry-kernel parameter parsing.

The packed dispatch path locates the generated ``__global__`` entry kernel and
splits its parameter list into (type, name) pairs.  CudaEmitter may emit the
multi-line signature form ``ret name(\\n    param,\\n ...\\n) {`` and parameter
types may carry commas inside template brackets (e.g. a
``cuda::std::tuple<float*, int32_t>`` device-param spill).  Splitting the list
on plain "," shatters such decls into bogus (type, name) pairs, which then flow
into ``_pack_classification``/``_sizeof_cpp_param`` and raise spurious
"cannot size kernel parameter type" errors or misclassify a group pointer.

These tests pin bracket-aware splitting inside ``_extract_entry_kernel``; no GPU
is required (CPU-only source-string contract).
"""

import unittest

from little_kernel.backends.nv.common.export import _foreach


def _cuda_source(params: str, name: str = "agent_vec_add") -> str:
    # Mirrors CudaEmitter.visit_FunctionDef's long-signature form exactly:
    # `extern "C" __global__ __launch_bounds__(N) void name(\n    p,\n ...\n) {`.
    return ("extern \"C\" __global__ __launch_bounds__(128) void " + name + "(\n    " + params + "\n) {\n"
            "    const unsigned lk_bx = blockIdx.x;\n"
            "    if (lk_bx < gridDim.x) { /* comment with brace { */ }\n"
            "}\n")


class TestExtractEntryKernelBracketAware(unittest.TestCase):

    def test_tuple_typed_param_splits_into_exact_two_decls(self):
        source = _cuda_source("cuda::std::tuple<float*, int32_t> extra, const float* x")
        bounds, params, body, span = _foreach._extract_entry_kernel(source, "agent_vec_add")
        self.assertEqual(params, [("cuda::std::tuple<float*, int32_t>", "extra"), ("const float*", "x")])
        self.assertEqual(bounds, "__launch_bounds__(128)")
        self.assertIn("lk_bx", body)
        self.assertEqual(source[span[0]:span[1]].rstrip()[-1], "}")

    def test_nested_templates_and_parenthesized_arrays(self):
        source = _cuda_source("cuda::std::tuple<cuda::std::tuple<int, float>, unsigned long> deep,"
                              " int32_t* __restrict__ out, unsigned n")
        _, params, _, _ = _foreach._extract_entry_kernel(source, "agent_vec_add")
        self.assertEqual(params, [("cuda::std::tuple<cuda::std::tuple<int, float>, unsigned long>", "deep"),
                                  ("int32_t* __restrict__", "out"), ("unsigned", "n")])

    def test_emitter_multiline_form_one_param_per_line(self):
        # Exact CudaEmitter long-signature rendering: one param per line with
        # newlines between top-level commas and `) {` on its own line.
        source = ("extern \"C\" __global__ void agent_pack(\n"
                  "    cuda::std::tuple<float*, int32_t> extra,\n"
                  "    float* out,\n"
                  "    int32_t n\n"
                  ") {\n"
                  "    // body\n"
                  "}\n")
        _, params, _, _ = _foreach._extract_entry_kernel(source, "agent_pack")
        self.assertEqual(params, [("cuda::std::tuple<float*, int32_t>", "extra"), ("float*", "out"), ("int32_t", "n")])

    def test_plain_params_still_parse(self):
        source = _cuda_source("float* out, const float* a, int32_t n")
        _, params, _, _ = _foreach._extract_entry_kernel(source, "agent_vec_add")
        self.assertEqual(params, [("float*", "out"), ("const float*", "a"), ("int32_t", "n")])

    def test_parsed_types_flow_through_sizeof(self):
        # A bogus split turns "cuda::std::tuple<float*, int32_t> extra" into
        # bogus segments; the pair produced by the bracket-aware split must be
        # sizeable (tuple contains a pointer) without raising.
        source = _cuda_source("cuda::std::tuple<float*, int32_t> extra, const float* x")
        _, params, _, _ = _foreach._extract_entry_kernel(source, "agent_vec_add")
        sizes = [_foreach._sizeof_cpp_param(type_text) for type_text, _ in params]
        self.assertEqual(sizes, [8, 8])

    def test_missing_entry_kernel_raises(self):
        with self.assertRaises(ValueError):
            _foreach._extract_entry_kernel("extern \"C\" __global__ void other(int n) { }", "agent_vec_add")

    def test_unparseable_parameter_decl_raises(self):
        # A decl with no trailing identifier cannot yield a (type, name) pair.
        with self.assertRaises(ValueError):
            _foreach._extract_entry_kernel("extern \"C\" __global__ void agent_k(int32_t 42) { }", "agent_k")


if __name__ == "__main__":
    unittest.main()
