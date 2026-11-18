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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""CPU regressions for warp-specialized cluster-persistent launches.

Covers the shared surface behind a 2-CTA producer/consumer kernel:

  * ll.warpgroup_reg_dec / ll.warpgroup_reg_inc lower to
    setmaxnreg.dec/inc.sync.aligned.u32 with distinct helper symbols, so a
    producer dec<40> and consumer inc<224> coexist in one translation unit;
  * ll.mbarrier_arrive_remote lowers to mapa.shared::cluster +
    mbarrier.arrive.shared::cluster peer arrival;
  * launch.grid shape expressions gain the zero-argument device_sms()
    builtin so a persistent one-wave grid can size itself from the device
    (min(device_sms() // 2, tiles) * 2) while staying divisible by a
    declared cluster_shape.

No GPU is used: these are IR/codegen-text and emitted-binding checks only.
"""

import json
import unittest
from pathlib import Path

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR
import little_kernel.backends.nv.common.language.intrin.barrier as barrier_intrin


class TestWarpgroupRegIncDec(unittest.TestCase):

    def test_inc_dec_are_exported_builtins(self):
        for name in ("warpgroup_reg_inc", "warpgroup_reg_dec", "warpgroup_reg_alloc", "warpgroup_reg_dealloc"):
            self.assertTrue(hasattr(ll, name), name)
            self.assertTrue(getattr(getattr(ll, name), BUILTIN_ATTR, False), name)

    def test_inc_dec_codegen_modes_and_distinct_helpers(self):
        from little_kernel.backends.nv.common.language.intrin.simt import codegen_warpgroup_reg_inc, codegen_warpgroup_reg_dec
        inc = codegen_warpgroup_reg_inc(224)
        dec = codegen_warpgroup_reg_dec(40)
        # Mode maps to the setmaxnreg direction and the immediate is legal
        # (24..256 in steps of 8).
        self.assertIn("setmaxnreg.inc.sync.aligned.u32 224;", inc.body)
        self.assertIn("setmaxnreg.dec.sync.aligned.u32 40;", dec.body)
        self.assertIn("void warpgroup_reg_inc_fn(int value)", inc.body)
        self.assertIn("void warpgroup_reg_dec_fn(int value)", dec.body)
        self.assertEqual(inc.return_val, "warpgroup_reg_inc_fn(224)")
        self.assertEqual(dec.return_val, "warpgroup_reg_dec_fn(40)")

    def test_producer_consumer_split_compiles(self):
        """A 2-CTA cluster draft: the producer warpgroup drops to dec<40> and
        the consumer warpgroups raise to inc<224>, each under its warp-role
        branch."""
        from little_kernel.core.passes import PASSES
        from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def warp_specialized(num_tiles: ll.int32) -> ll.void:
            tid: ll.int32 = ll.threadIdx_x()
            wg: ll.int32 = tid // 128
            if wg == 0:
                # TMA producer: low register ceiling.
                ll.warpgroup_reg_dec(40)
            else:
                # WGMMA consumers: high register ceiling.
                ll.warpgroup_reg_inc(224)

        code = warp_specialized.compile(PASSES["cuda"], codegen_cuda, num_threads=256)
        self.assertIn("setmaxnreg.dec.sync.aligned.u32 40;", code)
        self.assertIn("setmaxnreg.inc.sync.aligned.u32 224;", code)
        self.assertIn("warpgroup_reg_dec_fn(40);", code)
        self.assertIn("warpgroup_reg_inc_fn(224);", code)


class TestMbarrierArriveRemote(unittest.TestCase):

    def test_remote_arrive_codegen(self):
        codegen_func = getattr(barrier_intrin.mbarrier_arrive_remote, CODEGEN_FUNC_ATTR)
        builtin = codegen_func("bar", "peer")
        self.assertIn("mapa.shared::cluster.u32", builtin.body)
        self.assertIn("mbarrier.arrive.shared::cluster.b64", builtin.body)
        self.assertEqual(builtin.return_val, "mbarrier_arrive_remote_fn(bar, peer)")
        self.assertTrue(getattr(ll.mbarrier_arrive_remote, BUILTIN_ATTR, False))


class TestDeviceSmsGridExpr(unittest.TestCase):

    def test_shape_expr_lowers_device_sms(self):
        from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _shape_expr
        # Persistent one-wave grid for a 2-CTA cluster:
        # grid_x = min(sms / 2, tiles) * 2 stays divisible by cluster width.
        cpp = _shape_expr("min(device_sms() // 2, num_tiles) * 2", ranks={}, scalars={"num_tiles"})
        self.assertEqual(cpp, "(lk_min((lk_device_sms() / 2), num_tiles) * 2)")

    def test_shape_expr_rejects_device_sms_args(self):
        from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _shape_expr
        with self.assertRaises(ValueError) as ctx:
            _shape_expr("device_sms(0)", ranks={}, scalars=set())
        self.assertIn("device_sms takes no arguments", str(ctx.exception))

    def _write_spec(self, tmp_path, spec):
        path = Path(tmp_path) / "export.json"
        path.write_text(json.dumps(spec))
        return path

    def test_persistent_cluster_manifest_binding(self):
        from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _load_spec, _render_binding
        import tempfile
        spec = {
            "kernel": "probe",
            "function": "probe",
            "inputs": [{"name": "x", "dtype": "float32", "rank": 1}],
            "output": {"name": "y", "dtype": "float32", "shape": ["x[0]"]},
            # tiles = x[0]; one cluster per pair of SMs, 2 CTAs per cluster.
            "launch": {
                "grid": ["min(device_sms() // 2, x[0]) * 2", 1, 1],
                "block": [256, 1, 1],
                "cluster_shape": [2, 1, 1],
            },
            "scalars": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            loaded = _load_spec(self._write_spec(tmp, spec))
        self.assertEqual(loaded["launch"]["_grid_cpp"][0], "(lk_min((lk_device_sms() / 2), x.size(0)) * 2)")
        binding = _render_binding(loaded)
        # Host helper queries the real device multiprocessor count.
        self.assertIn("int64_t lk_device_sms()", binding)
        self.assertIn("cudaDevAttrMultiProcessorCount", binding)
        self.assertIn("const int64_t lk_grid_x = "
                      "(lk_min((lk_device_sms() / 2), x.size(0)) * 2);", binding)
        # Persistent grid expression is always even; the binding still guards
        # cluster divisibility before launch (ragged-tail deadlock defence).
        self.assertIn("lk_grid_x % 2 == 0", binding)


if __name__ == "__main__":
    unittest.main()
