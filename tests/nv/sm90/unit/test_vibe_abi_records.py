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
"""Launch-ABI records: rank-0 outputs, passthrough args, optional inputs, dtypes.

Covers the generic manifest/binding extensions requested across cycles:
- rank-0 (scalar) outputs via ``"shape": []`` (callee-allocated output, numel 1);
- ``passthrough_forward_args`` + ``forward_args`` positional ABI reordering;
- ``"optional": true`` inputs with the ``present(name)`` shape expression;
- byte-logical dtypes (bool) routed as one-byte storage;
- generic WGMMA SS dtype/trans/scale variants with PTX immediates baked in.

Legality contract (per cw.codegen.tilelang-runtime-abi::skill, pinned wiki
revision 2e523d63b315bedc2adb022b750d9af0f42f7428, LaunchABI ordered parameter
metadata): the manifest record fixes the full ordered forward ABI; generated
code never infers ABI slots from names, and manifests that use no extension
must render bindings without any extension text.
"""

import json
import re
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower

VEC_ADD = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def vec_add(a: ll.Tensor[ll.float32], b: ll.Tensor[ll.float32],
            out: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        out[i] = a[i] + b[i]
'''

VEC_ADD_SPEC = {
    "kernel": "vec_add",
    "function": "add",
    "inputs": [
        {"name": "a", "dtype": "float32", "rank": 1},
        {"name": "b", "dtype": "float32", "rank": 1},
    ],
    "output": {"name": "out", "dtype": "float32", "shape": ["a[0]"]},
    "scalars": [{"name": "n", "expr": "a[0]"}],
    "launch": {"grid": ["ceil_div(a[0], 256)", 1, 1], "block": [256, 1, 1]},
}

GATED = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def gated(x: ll.Tensor[ll.float32], bias: ll.Tensor[ll.float32],
          out: ll.Tensor[ll.float32], n: ll.int32, has_bias: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        value: ll.float32 = x[i]
        if has_bias == 1:
            value = value + bias[i]
        out[i] = value
'''

GATED_SPEC = {
    "kernel":
    "gated",
    "function":
    "gated",
    "inputs": [
        {"name": "x", "dtype": "float32", "rank": 1},
        {"name": "bias", "dtype": "float32", "rank": 1, "optional": True},
    ],
    "output": {"name": "out", "dtype": "float32", "shape": ["x[0]"]},
    "scalars": [{"name": "n", "expr": "x[0]"}, {"name": "has_bias", "expr": "present(bias)"}],
    "launch": {"grid": ["ceil_div(x[0], 256)", 1, 1], "block": [256, 1, 1]},
}

SCALE_WRITE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def row_less(logits: ll.Tensor[ll.float32], loss: ll.Tensor[ll.float32]) -> ll.void:
    if ll.blockIdx_x() == 0 and ll.threadIdx_x() == 0:
        loss[0] = logits[0]
'''

RANK0_SPEC = {
    "kernel": "row_less",
    "function": "row_less",
    "inputs": [{"name": "logits", "dtype": "float32", "rank": 2}],
    "output": {"name": "loss", "dtype": "float32", "shape": []},
    "launch": {"grid": [1, 1, 1], "block": [32, 1, 1]},
}

WGMMA_PROBE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def wgmma_probe(sink: ll.Tensor[ll.float32], desc_a: ll.int64,
                desc_b: ll.int64) -> ll.void:
    acc = ll.zeros([32], dtype=ll.float32, scope="local")
    ll.wgmma_m64n64k16_f16(acc, ll.uint64(desc_a), ll.uint64(desc_b), 0, 1, 1)
    sink[ll.threadIdx_x()] = acc[0]
'''

WGMMA_PROBE_BAD = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def wgmma_probe(sink: ll.Tensor[ll.float32], desc_a: ll.int64,
                desc_b: ll.int64) -> ll.void:
    acc = ll.zeros([32], dtype=ll.float32, scope="local")
    ll.wgmma_m64n64k16_f16(acc, ll.uint64(desc_a), ll.uint64(desc_b), 0, 2, 1)
    sink[ll.threadIdx_x()] = acc[0]
'''

WGMMA_LEGACY = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def wgmma_probe(sink: ll.Tensor[ll.float32], desc_a: ll.int64,
                desc_b: ll.int64) -> ll.void:
    acc = ll.zeros([128], dtype=ll.float32, scope="local")
    ll.wgmma_m64n256k16(acc, ll.uint64(desc_a), ll.uint64(desc_b))
    sink[ll.threadIdx_x()] = acc[0]
'''

WGMMA_PROBE_SPEC = {
    "kernel":
    "wgmma_probe",
    "function":
    "wgmma_probe",
    "inputs": [{"name": "sink", "dtype": "float32", "rank": 1}],
    "scalars": [
        {"name": "desc_a", "dtype": "int64", "runtime": True},
        {"name": "desc_b", "dtype": "int64", "runtime": True},
    ],
    "output": {"name": "sink_out", "dtype": "float32", "shape": ["sink[0]"]},
    "kernel_args": ["sink", "desc_a", "desc_b"],
    "launch": {"grid": [1, 1, 1], "block": [128, 1, 1]},
}


def _export(program_source, spec):
    """Run the generic compiler over one authored program; return its directory."""
    root = tempfile.TemporaryDirectory()
    base = Path(root.name)
    task = base / "task"
    task.mkdir()
    (task / "model.py").write_text("class Model: pass\n")
    (task / "test_cases.py").write_text("test_cases = []\n")
    program = base / "program"
    program.mkdir()
    (program / "draft.py").write_text(program_source)
    (program / "export.json").write_text(json.dumps(spec))
    output = base / "output"
    manifest = lower(task, program, output, 0)
    return root, output, manifest


def _read(output, name):
    return (output / name).read_text()


class RankZeroOutputTests(unittest.TestCase):

    def test_rank0_output_allocates_and_launches(self):
        root, output, _ = _export(SCALE_WRITE, RANK0_SPEC)
        self.addCleanup(root.cleanup)
        binding = _read(output, "kernels/agent_lk_row_less_binding.cpp")
        # Rank-0 output: at::empty({}) has numel()==1, so the all-outputs-empty
        # early return can never fire and the kernel always launches.
        self.assertIn("at::empty({}", binding)
        self.assertIn("if (loss.numel() == 0) return loss;", binding)
        model = _read(output, "model_new.py")
        self.assertIn("def forward(self, logits):", model)

    def test_list_shape_contract_unchanged(self):
        bad = dict(RANK0_SPEC, output={"name": "loss", "dtype": "float32"})
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            task = base / "task"
            task.mkdir()
            (task / "model.py").write_text("class Model: pass\n")
            (task / "test_cases.py").write_text("test_cases = []\n")
            program = base / "program"
            program.mkdir()
            (program / "draft.py").write_text(SCALE_WRITE)
            (program / "export.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                lower(task, program, base / "output", 0)


class PassthroughForwardArgTests(unittest.TestCase):

    def test_passthrough_slot_accepted_and_dropped(self):
        spec = dict(VEC_ADD_SPEC)
        spec["passthrough_forward_args"] = ["reduction"]
        spec["forward_args"] = ["b", "a", "reduction"]
        root, output, _ = _export(VEC_ADD, spec)
        self.addCleanup(root.cleanup)
        model = _read(output, "model_new.py")
        self.assertIn("def forward(self, b, a, reduction):", model)
        # The untyped ABI slot never reaches the extension.
        self.assertIn("cuda_extension.agent_add(a, b)", model)
        binding = _read(output, "kernels/agent_lk_add_binding.cpp")
        self.assertNotIn("reduction", binding)

    def test_passthrough_requires_forward_args(self):
        spec = dict(VEC_ADD_SPEC)
        spec["passthrough_forward_args"] = ["reduction"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "task").mkdir()
            (base / "task/model.py").write_text("class Model: pass\n")
            (base / "task/test_cases.py").write_text("test_cases = []\n")
            program = base / "program"
            program.mkdir()
            (program / "draft.py").write_text(VEC_ADD)
            (program / "export.json").write_text(json.dumps(spec))
            with self.assertRaises(ValueError):
                lower(base / "task", program, base / "output", 0)

    def test_forward_args_must_cover_all_slots(self):
        spec = dict(VEC_ADD_SPEC)
        spec["passthrough_forward_args"] = ["reduction"]
        spec["forward_args"] = ["a", "reduction"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "task").mkdir()
            (base / "task/model.py").write_text("class Model: pass\n")
            (base / "task/test_cases.py").write_text("test_cases = []\n")
            program = base / "program"
            program.mkdir()
            (program / "draft.py").write_text(VEC_ADD)
            (program / "export.json").write_text(json.dumps(spec))
            with self.assertRaises(ValueError):
                lower(base / "task", program, base / "output", 0)


class OptionalInputTests(unittest.TestCase):

    def test_optional_input_binds_nullable_and_present(self):
        root, output, _ = _export(GATED, GATED_SPEC)
        self.addCleanup(root.cleanup)
        binding = _read(output, "kernels/agent_lk_gated_binding.cpp")
        self.assertIn("#include <optional>", binding)
        self.assertIn("std::optional<torch::Tensor> bias_opt", binding)
        # Absent optional inputs materialize as zero-sized tensors of the
        # declared dtype/rank so every check and data_ptr path stays valid.
        self.assertRegex(binding, r"const torch::Tensor bias = bias_opt\.value_or\(at::empty\(\{0\}")
        self.assertIn("(bias_opt.has_value() ? 1 : 0)", binding)
        cuda = _read(output, "kernels/agent_lk_gated.cu")
        self.assertIn("has_bias", cuda)
        model = _read(output, "model_new.py")
        self.assertIn("def forward(self, x, bias):", model)

    def test_non_optional_manifest_has_no_optional_text(self):
        spec = {
            "kernel": "gated",
            "function": "gated",
            "inputs": [{"name": "x", "dtype": "float32", "rank": 1}, {"name": "bias", "dtype": "float32", "rank": 1}],
            "output": dict(GATED_SPEC["output"]),
            "scalars": [dict(GATED_SPEC["scalars"][0]), {"name": "has_bias", "expr": "1"}],
            "launch": dict(GATED_SPEC["launch"]),
        }
        root, output, _ = _export(GATED, spec)
        self.addCleanup(root.cleanup)
        binding = _read(output, "kernels/agent_lk_gated_binding.cpp")
        self.assertNotIn("std::optional", binding)
        self.assertNotIn("#include <optional>", binding)

    def test_present_requires_optional_input(self):
        bad = dict(GATED_SPEC)
        bad["inputs"] = [{"name": "x", "dtype": "float32", "rank": 1}, {"name": "bias", "dtype": "float32", "rank": 1}]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "task").mkdir()
            (base / "task/model.py").write_text("class Model: pass\n")
            (base / "task/test_cases.py").write_text("test_cases = []\n")
            program = base / "program"
            program.mkdir()
            (program / "draft.py").write_text(GATED)
            (program / "export.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                lower(base / "task", program, base / "output", 0)


BYTE_COPY = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def byte_copy(flags: ll.Tensor[ll.uint8], out: ll.Tensor[ll.uint8],
              n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        out[i] = flags[i]
'''

BYTE_COPY_SPEC = {
    "kernel": "byte_copy",
    "function": "byte_copy",
    "inputs": [{"name": "flags", "dtype": "bool", "rank": 1}],
    "output": {"name": "out", "dtype": "bool", "shape": ["flags[0]"]},
    "scalars": [{"name": "n", "expr": "flags[0]"}],
    "launch": {"grid": ["ceil_div(flags[0], 256)", 1, 1], "block": [256, 1, 1]},
}


class DtypeTests(unittest.TestCase):

    def test_bool_tensors_use_byte_storage(self):
        root, output, _ = _export(BYTE_COPY, BYTE_COPY_SPEC)
        self.addCleanup(root.cleanup)
        binding = _read(output, "kernels/agent_lk_byte_copy_binding.cpp")
        self.assertIn("flags.scalar_type() == at::kBool", binding)
        self.assertIn("out = at::empty({flags.size(0)}, flags.options().dtype(at::kBool))", binding)
        cuda = _read(output, "kernels/agent_lk_byte_copy.cu")
        self.assertIn("reinterpret_cast<uint8_t*>(flags)", cuda)


class WgmmaGenericVariantTests(unittest.TestCase):

    def test_f16_variant_bakes_trans_and_scale_immediates(self):
        root, output, _ = _export(WGMMA_PROBE, WGMMA_PROBE_SPEC)
        self.addCleanup(root.cleanup)
        cuda = _read(output, "kernels/agent_lk_wgmma_probe.cu")
        self.assertIn("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16", cuda)
        # Helper name carries the immediates so combinations never collide.
        self.assertIn("wgmma_m64n64k16_f16_a0b1d1_fn", cuda)
        self.assertRegex(cuda, r"%32,%33,p,1,1,0,1;")

    def test_non_binary_flag_is_a_compile_error(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "task").mkdir()
            (base / "task/model.py").write_text("class Model: pass\n")
            (base / "task/test_cases.py").write_text("test_cases = []\n")
            program = base / "program"
            program.mkdir()
            (program / "draft.py").write_text(WGMMA_PROBE_BAD)
            (program / "export.json").write_text(json.dumps(WGMMA_PROBE_SPEC))
            with self.assertRaises(ValueError):
                lower(base / "task", program, base / "output", 0)

    def test_legacy_bf16_entry_points_unchanged(self):
        root, output, _ = _export(WGMMA_LEGACY, WGMMA_PROBE_SPEC)
        self.addCleanup(root.cleanup)
        cuda = _read(output, "kernels/agent_lk_wgmma_probe.cu")
        self.assertIn("wgmma_m64n256k16_fn(", cuda)
        self.assertIn("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16", cuda)


class PipelinePassthroughTests(unittest.TestCase):

    def test_pipeline_forward_abi_carries_passthrough(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            task = base / "task"
            task.mkdir()
            (task / "model.py").write_text("class Model: pass\n")
            (task / "test_cases.py").write_text("test_cases = []\n")
            program = base / "program"
            stage = program / "stages" / "add"
            stage.mkdir(parents=True)
            (stage / "draft.py").write_text(VEC_ADD)
            (stage / "export.json").write_text(json.dumps(VEC_ADD_SPEC))
            (program / "pipeline.json").write_text(
                json.dumps({
                    "inputs": ["a", "b"],
                    "outputs": ["out"],
                    "passthrough_forward_args": ["reduction"],
                    "forward_args": ["a", "b", "reduction"],
                    "stages": [{"name": "add"}],
                }))
            output = base / "output"
            manifest = lower(task, program, output, 0)
            self.assertEqual(manifest["backend"], "cuda-extension")
            model = _read(output, "model_new.py")
            self.assertIn("def forward(self, a, b, reduction):", model)
            self.assertIn("cuda_extension.agent_add(a, b)", model)


if __name__ == "__main__":
    unittest.main()
