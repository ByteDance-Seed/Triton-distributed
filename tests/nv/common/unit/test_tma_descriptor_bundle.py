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
"""Per-batch TMA descriptor bundles for persistent cluster launches
(semantic key ir.launch.cluster_persistent_tma_multicast).

A warp-specialized batched GEMM keeps one launch: a cluster(2,1,1) grid
with a scheduler-strided persistent loop over cluster id, producer CTAs
multicasting B tiles (mask 0x3) through TMA, and rank-1 CTAs remote-
arriving on rank-0 arm barriers. The per-batch coordinate cannot live in
a single grid_constant descriptor: the host ABI must build a MapBundle --
one CUtensorMap per leading batch slice -- stage it to a device array and
hand the entry kernel a ``const CUtensorMap*`` it indexes as ``dB[b]``.

These CPU regressions pin the shared surface (no nvcc/GPU):

  * a ``batch_count`` manifest entry validates as a bundle and rejects
    unknown/typo'd fields, batch axes that do not exist, and stray
    ``batch_stride`` declarations;
  * the typed-IR entry asks for ``const[ptr[TmaDescriptor]]`` and may
    subscript it at runtime; the lowering ABI check rejects descriptor
    parameters whose transport (by-value grid_constant vs device-array
    pointer) disagrees with the manifest;
  * the compiled kernel carries the full one-launch pairing: persistent
    blockIdx/gridDim stride loop, %cluster_ctarank, 2d multicast TMA with
    mask 3, mbarrier parity wait and mapa-based remote arrive;
  * the launcher transports ``CUtensorMap*`` by pointer, and the binding
    host-encodes one map per batch slice at the slice offset, uploads the
    array with cudaMalloc/cudaMemcpy and releases it with cudaFree.
"""

import json
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _load_kernel, _load_spec, _render_binding, _render_cuda

DRAFT = """
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend='cuda', is_entry=True)
def bmm_persistent(dB: ll.const[ll.ptr[ll.TmaDescriptor]],
                   B: ll.Tensor[ll.float16]) -> ll.void:
    rank: ll.uint32 = ll.cluster_rank()
    b: ll.int32 = ll.blockIdx_y()
    sB = ll.alloc_shared_memory("sB", ll.float16, 64 * 8)
    bar = ll.alloc_shared_memory("s_bar", ll.int64, 1)
    ll.init_smem_barrier(bar, 2)
    # Persistent scheduler loop keyed on cluster id: stride is grid in
    # clusters (gridDim.x // 2 for cluster_shape (2,1,1)).
    for tile in range(ll.blockIdx_x() // 2, 910, ll.gridDim_x() // 2):
        desc = dB[b]
        if rank == 0:
            ll.mbarrier_arrive_and_expect_tx(bar, 64 * 8 * 2)
            # Multicast the B tile to both CTAs of the cluster (mask 0x3).
            ll.tma_load_2d_1sm_mcast(desc, bar, sB, 0, tile * 8, 0x3)
        ll.mbarrier_wait(bar, 0)
        if rank == 1:
            # Consumer CTA remote-arrives on the producer's arm barrier.
            ll.mbarrier_arrive_remote(bar, 0)
"""

GRID_CONST_DRAFT = DRAFT.replace("dB: ll.const[ll.ptr[ll.TmaDescriptor]]",
                                 "dB: ll.const[ll.grid_constant[ll.TmaDescriptor]]")


def _spec(*, batch_stride=None, extra_field=None, tensor_rank=3):
    descriptor = {
        "name": "dB", "tensor": "B", "rank": 2, "inner": "B[2]", "outer": "B[1]", "outer_stride": "B[2]", "box":
        [64, 8], "swizzle_bytes": 0, "batch_count": "B[0]"
    }
    if batch_stride is not None:
        descriptor["batch_stride"] = batch_stride
    if extra_field is not None:
        descriptor.update(extra_field)
    return {
        "function": "lk_forward",
        "kernel": "bmm_persistent",
        "inputs": [{"name": "B", "dtype": "float16", "rank": tensor_rank}],
        "outputs": [{"name": "out", "dtype": "float16", "shape": ["B[0]", "B[1]", "B[2]"]}],
        "scalars": [],
        "launch": {"grid": [2, "B[0]", 1], "block": [128, 1, 1], "cluster_shape": [2, 1, 1]},
        "tma_descriptors": [descriptor],
        "kernel_args": ["dB", "B"],
    }


def _write(tmp_path: Path, spec: dict, draft: str = DRAFT):
    (tmp_path / "draft.py").write_text(draft)
    (tmp_path / "export.json").write_text(json.dumps(spec))


def _lower(tmp_path: Path, spec: dict, draft: str = DRAFT):
    _write(tmp_path, spec, draft)
    spec = _load_spec(tmp_path / "export.json")
    kernel = _load_kernel(tmp_path / "draft.py", spec["_kernel"])
    return spec, _render_cuda(spec, kernel), _render_binding(spec)


class TestBundleManifestValidation(unittest.TestCase):

    def test_bundle_entry_annotated(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            _write(Path(td), _spec())
            spec = _load_spec(Path(td) / "export.json")
        item = spec["tma_descriptors"][0]
        self.assertTrue(item["_bundle"])
        self.assertEqual(item["_batch_count_cpp"], "B.size(0)")
        self.assertIn("dB", spec["_descriptor_bundles"])
        # Explicit element stride expression is host-evaluated.
        with tempfile.TemporaryDirectory() as td:
            _write(Path(td), _spec(batch_stride="B[1] * B[2]"))
            spec = _load_spec(Path(td) / "export.json")
        item = spec["tma_descriptors"][0]
        self.assertEqual(item["_batch_stride_cpp"], "(B.size(1) * B.size(2))")

    def test_default_contiguous_batch_stride(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            _write(Path(td), _spec())
            spec = _load_spec(Path(td) / "export.json")
        item = spec["tma_descriptors"][0]
        # Contiguous [B, outer, inner]: one slice spans outer*outer_stride.
        self.assertEqual(item["_batch_stride_cpp"], "((B.size(1)) * (B.size(2)))")

    def test_unknown_descriptor_field_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            _write(Path(td), _spec(extra_field={"batch_cnt": "B[0]"}))
            with self.assertRaisesRegex(ValueError, "unknown TMA descriptor fields"):
                _load_spec(Path(td) / "export.json")

    def test_bundle_requires_batch_axis(self):
        import tempfile
        # rank-2 tensor with a coherent rank-2 slice descriptor but no
        # leading batch axis.
        spec = _spec(tensor_rank=2)
        spec["outputs"] = [{"name": "out", "dtype": "float16", "shape": ["B[0]", "B[1]"]}]
        desc = spec["tma_descriptors"][0]
        desc.update(inner="B[1]", outer="B[0]", outer_stride="B[1]")
        with tempfile.TemporaryDirectory() as td:
            _write(Path(td), spec)
            with self.assertRaisesRegex(ValueError, "leading batch axis"):
                _load_spec(Path(td) / "export.json")

    def test_batch_stride_requires_count(self):
        import tempfile
        spec = _spec()
        del spec["tma_descriptors"][0]["batch_count"]
        spec["tma_descriptors"][0]["batch_stride"] = "B[1] * B[2]"
        with tempfile.TemporaryDirectory() as td:
            _write(Path(td), spec)
            with self.assertRaisesRegex(ValueError, "batch_stride requires batch_count"):
                _load_spec(Path(td) / "export.json")


class TestBundleDeviceIR(unittest.TestCase):

    def test_entry_signature_is_device_array_pointer(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            _, cuda, _ = _lower(Path(td), _spec())
        self.assertIn("using TmaDescriptor = CUtensorMap;", cuda)
        self.assertIn("extern \"C\" __global__ void agent_bmm_persistent("
                      "const TmaDescriptor* dB,", cuda)
        # No by-value grid_constant transport for a bundle.
        self.assertNotIn("__grid_constant__ TmaDescriptor dB", cuda)
        # Runtime per-batch subscript feeds the multicast TMA load.
        self.assertIn("desc = dB[b];", cuda)

    def test_one_launch_cluster_persistent_pairing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            _, cuda, _ = _lower(Path(td), _spec())
        # Scheduler-strided persistent loop over cluster id.
        self.assertIn("blockIdx.x / 2", cuda)
        self.assertIn("gridDim.x / 2", cuda)
        # Cluster rank and B-tile multicast mask 0x3.
        self.assertIn("%cluster_ctarank", cuda)
        self.assertIn("multicast::cluster", cuda)
        self.assertIn("tma_load_2d_1sm_mcast_fn(&desc, bar, sB, 0, tile * 8, 3, 1)", cuda)
        # mbarrier parity ring and sibling-CTA remote arrive (mapa).
        self.assertIn("mbarrier.try_wait.parity", cuda)
        self.assertIn("mbarrier.arrive.expect_tx", cuda)
        self.assertIn("mapa.shared::cluster", cuda)

    def test_bundle_with_grid_constant_param_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td)
            _write(tp, _spec(), GRID_CONST_DRAFT)
            spec = _load_spec(tp / "export.json")
            kernel = _load_kernel(tp / "draft.py", spec["_kernel"])
            with self.assertRaisesRegex(ValueError, "descriptor bundle dB"):
                _render_cuda(spec, kernel)

    def test_singleton_with_pointer_param_rejected(self):
        import tempfile
        spec = _spec()
        del spec["tma_descriptors"][0]["batch_count"]
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td)
            _write(tp, spec)
            spec = _load_spec(tp / "export.json")
            kernel = _load_kernel(tp / "draft.py", spec["_kernel"])
            with self.assertRaisesRegex(ValueError, "grid_constant\\[TmaDescriptor\\]"):
                _render_cuda(spec, kernel)


class TestBundleBindingABI(unittest.TestCase):

    def test_launcher_transports_pointer(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            _, cuda, _ = _lower(Path(td), _spec())
        self.assertIn("extern \"C\" cudaError_t agent_lk_launch_lk_forward(\n"
                      "    CUtensorMap* dB,", cuda)
        # Cluster launch forwards the bare device pointer.
        self.assertIn("cudaLaunchKernelEx(&config, agent_bmm_persistent, dB,", cuda)

    def test_binding_encodes_per_batch_and_uploads_array(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            _, _, binding = _lower(Path(td), _spec())
        # One encoded map per batch slice, at the per-slice offset.
        self.assertIn("int64_t lk_desc_count_dB = B.size(0);", binding)
        self.assertIn("std::vector<CUtensorMap> dB_host", binding)
        self.assertIn(
            "agent_lk_make_tma_2d(&dB_host[lk_b], "
            "static_cast<char*>((B).data_ptr()) + "
            "lk_b * (((B.size(1)) * (B.size(2)))) * 2,", binding)
        # Device staging and post-launch release.
        self.assertIn("cudaMalloc(reinterpret_cast<void**>(&dB)", binding)
        self.assertIn("cudaMemcpy(dB, dB_host.data()", binding)
        self.assertIn("agent_lk_launch_lk_forward(dB,", binding)
        self.assertIn("cudaFree(dB)", binding)
        # Cluster divisibility check accompanies the persistent grid.
        self.assertIn('TORCH_CHECK(lk_grid_x % 2 == 0', binding)


if __name__ == "__main__":
    unittest.main()
