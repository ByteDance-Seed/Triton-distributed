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
"""Derived constexpr constructor selectors (runtime-specialized kernels).

An export.json constructor entry may carry ``"constexpr": "<expr>"`` where the
expression is a small comparison/arithmetic over *preceding* ordinary
constructor args. Unlike a literal constexpr, the selector's value is bound
when the Model is constructed (``self._lk_cx_<name> = bool(<expr>)``) and the
export fans out into one compile-time-pruned CUDA kernel per bool domain value
(FALSE/TRUE), picked by the generated launcher. The source args stay ordinary
runtime kernel scalars; the selector name never reaches the device ABI.

This is the generic capability behind "specialize a fused kernel on
momentum == 0 at Model construction while momentum stays a runtime FP32
scalar": the FALSE variant must not even mention the momentum buffer outside
its (unused) parameter list.
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower

SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def sgd(g: ll.Tensor[ll.float32], p: ll.Tensor[ll.float32], m: ll.Tensor[ll.float32],
        out: ll.Tensor[ll.float32], n: ll.int32,
        momentum: ll.float32, has_mom: ll.bool_) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        if has_mom:
            mv: ll.float32 = m[i] * momentum + g[i]
            out[i] = p[i] - 0.01 * mv
        else:
            out[i] = p[i] - 0.01 * g[i]
'''

SPEC = {
    "kernel":
    "sgd",
    "function":
    "sgd",
    "inputs": [
        {"name": "g", "dtype": "float32", "rank": 1},
        {"name": "p", "dtype": "float32", "rank": 1},
        {"name": "m", "dtype": "float32", "rank": 1},
    ],
    "output": {"name": "out", "dtype": "float32", "shape": ["g[0]"]},
    "constructor": [
        {"name": "momentum", "dtype": "float32"},
        {"name": "has_mom", "dtype": "bool", "constexpr": "momentum != 0.0"},
    ],
    "scalars": [{"name": "n", "expr": "g[0]"}],
    "kernel_args": ["g", "p", "m", "out", "n", "momentum", "has_mom"],
    "launch": {"grid": ["ceil_div(g[0], 256)", 1, 1], "block": [256, 1, 1]},
}


def _build(tmp_root, source=SOURCE, spec=SPEC):
    root = Path(tmp_root)
    task = root / "task"
    task.mkdir()
    (task / "model.py").write_text("class Model: pass\n")
    (task / "test_cases.py").write_text("test_cases = []\n")
    program = root / "program"
    program.mkdir()
    (program / "draft.py").write_text(source)
    (program / "export.json").write_text(json.dumps(spec))
    output = root / "out"
    lower(task, program, output, 0)
    return {
        "cuda": (output / "kernels/agent_lk_sgd.cu").read_text(),
        "binding": (output / "kernels/agent_lk_sgd_binding.cpp").read_text(),
        "model": (output / "model_new.py").read_text(),
    }


def _variant_body(cuda: str, index: int) -> str:
    start = cuda.index(f"namespace lk_cx{index} {{")
    end = cuda.index(f"}} // namespace lk_cx{index}", start)
    return cuda[start:end]


class DerivedConstexprTests(unittest.TestCase):

    def test_fans_out_into_two_pruned_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = _build(directory)
            cuda = generated["cuda"]
            off = _variant_body(cuda, 0)
            on = _variant_body(cuda, 1)
            # Both specializations are baked into one translation unit.
            self.assertIn("namespace lk_cx0 {", cuda)
            self.assertIn("namespace lk_cx1 {", cuda)
            # The FALSE variant keeps the momentum-buffer pointer in its
            # signature but never dereferences it (bit-exact no-touch path).
            self.assertIn("const float* __restrict__ m", off)
            self.assertNotIn("m[", off)
            self.assertNotIn("if (false)", off)
            self.assertIn("0.01f * g[", off)
            # The TRUE variant runs the fused momentum update.
            self.assertIn("* momentum + g[", on)
            self.assertNotIn("if (true)", on)
            # The originating arg stays a runtime scalar in BOTH variants.
            self.assertEqual(off.count("float momentum"), 1)
            self.assertEqual(on.count("float momentum"), 1)
            # The selector name never reaches device code.
            self.assertNotIn("has_mom", cuda.replace("lk_cx_has_mom", ""))
            # Includes stay at global scope, outside the variant namespaces.
            self.assertLess(cuda.index("#include <cstdint>"), cuda.index("namespace lk_cx0 {"))

    def test_launcher_switches_on_init_bound_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda = _build(directory)["cuda"]
            launcher = cuda[cuda.index("extern \"C\" cudaError_t agent_lk_launch_sgd"):]
            self.assertIn("bool lk_cx_has_mom", launcher)
            self.assertIn("const unsigned lk_cx_index = "
                          "((unsigned)(lk_cx_has_mom ? 1 : 0) << 0);", launcher)
            self.assertIn("case 0:", launcher)
            self.assertIn("case 1:", launcher)
            self.assertIn("lk_cx0::agent_sgd<<<", launcher)
            self.assertIn("lk_cx1::agent_sgd<<<", launcher)
            self.assertIn("default: return cudaErrorInvalidValue;", launcher)
            # Runtime momentum scalar reaches both launches.
            self.assertEqual(launcher.count("static_cast<float>(momentum)"), 2)

    def test_binding_threads_selector_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            binding = _build(directory)["binding"]
            # The selector is not a Model constructor parameter of the binding;
            # momentum is, and the selector flag trails the launcher ABI.
            self.assertIn(
                "static torch::Tensor agent_sgd(torch::Tensor g, "
                "torch::Tensor p, torch::Tensor m, float momentum, "
                "bool lk_cx_has_mom)", binding)
            self.assertIn("cudaStream_t stream, bool lk_cx_has_mom)", binding)
            self.assertIn(", 0, lk_stream, lk_cx_has_mom)", binding)
            self.assertIn('pybind11::arg("lk_cx_has_mom")', binding)
            # No host-side literal alias is invented for a derived selector.
            self.assertNotIn("const bool has_mom =", binding)

    def test_model_evaluates_selector_at_init(self):
        with tempfile.TemporaryDirectory() as directory:
            model = _build(directory)["model"]
            compile(model, "model_new.py", "exec")
            # The selector is computed from the actual constructor value, not
            # declared as a parameter or checked against a frozen literal.
            self.assertIn("def __init__(self, momentum):", model)
            self.assertIn("self._lk_cx_has_mom = bool(momentum != 0.0)", model)
            self.assertNotIn("if has_mom !=", model)
            # The originating scalar still flows to the kernel at runtime.
            self.assertIn("self.momentum = momentum", model)
            self.assertIn("cuda_extension.agent_sgd(g, p, m, self.momentum, "
                          "lk_cx_has_mom=self._lk_cx_has_mom)", model)

    def test_spec_diagnostics(self):
        cases = [
            ([{"name": "momentum", "dtype": "float32"},
              {"name": "flag", "dtype": "int32", "constexpr":
               "momentum != 0.0"}], "derived constexpr expression requires bool dtype"),
            ([{"name": "flag", "dtype": "bool", "constexpr":
               "momentum != 0.0"}], "may only reference preceding constructor arguments"),
            ([{"name": "momentum", "dtype": "float32"},
              {"name": "flag", "dtype": "bool", "constexpr":
               "select(momentum, 1, 0)"}], "unsupported derived constexpr expression"),
            ([{"name": "momentum", "dtype": "float32"},
              {"name": "flag", "dtype": "bool", "constexpr": "momentum != 0.0", "default":
               False}], "constexpr constructor arguments must not declare a default"),
            ([{"name": "momentum", "dtype": "float32"},
              {"name": "flag", "dtype": "bool", "constexpr":
               "  "}], "derived constexpr expression must be a nonempty string"),
            ([{"name": "momentum", "dtype": "float32"},
              {"name": "flag", "dtype": "bool", "constexpr":
               "momentum and True"}], "unsupported derived constexpr expression"),
        ]
        for constructor, message in cases:
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, message):
                    _build(directory, spec={**SPEC, "constructor": constructor})

    def test_literal_constexpr_still_shares_the_same_entry(self):
        # A literal bool constexpr does not fan out: one kernel, no switch.
        literal = {
            **SPEC,
            "constructor": [{"name": "has_mom", "dtype": "bool", "constexpr": True}],
            "kernel_args": ["g", "p", "m", "out", "n", "has_mom"],
        }
        source = SOURCE.replace("momentum: ll.float32, has_mom", "has_mom")
        source = source.replace("m[i] * momentum + g[i]", "m[i] + g[i]")
        with tempfile.TemporaryDirectory() as directory:
            cuda = _build(directory, source=source, spec=literal)["cuda"]
            self.assertNotIn("namespace lk_cx", cuda)
            self.assertNotIn("lk_cx_index", cuda)
            # The literal-true arm is pruned the same way as fanout variants.
            self.assertNotIn("if (true)", cuda)
            self.assertIn("+ g[", cuda)


if __name__ == "__main__":
    unittest.main()
