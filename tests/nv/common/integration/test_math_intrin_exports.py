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
# WITHOUT WARRANTY OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
"""GPU regression test for the scalar math intrinsics newly exported from
`little_kernel.language` (cycle-008 kernel requests kernel-01 R1 / kernel-03):
`sqrt_f32`, `log2_f32`, `rcp_f32`, `sin_f32`, `cos_f32`, `fma_f32`.

Each processes one element per thread. `sqrt_f32` is IEEE RN; `log2_f32`
and `rcp_f32` are approximate SFU operations with bounded error, while
`sin_f32`/`cos_f32` also use PTX SFU error contracts; `fma_f32` matches a
widened float64 a*b+c calculation.
"""

try:  # Optional CUDA dependency; CPU collection must succeed without Torch.
    import torch
except ImportError:  # pragma: no cover - exercised only in Torch environments
    torch = None

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from tests.nv.conftest import requires_cuda

N = 65536


@lk.ll_kernel(backend="cuda", is_entry=True)
def math_kernel(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32], z: ll.Tensor[ll.float32],
                o_sqrt: ll.Tensor[ll.float32], o_log2: ll.Tensor[ll.float32], o_rcp: ll.Tensor[ll.float32],
                o_sin: ll.Tensor[ll.float32], o_cos: ll.Tensor[ll.float32], o_fma: ll.Tensor[ll.float32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < N:
        v: ll.float32 = x[i]
        o_sqrt[i] = ll.sqrt_f32(v)
        o_log2[i] = ll.log2_f32(v + 1.0)
        o_rcp[i] = ll.rcp_f32(v + 3.0)
        o_sin[i] = ll.sin_f32(v)
        o_cos[i] = ll.cos_f32(v)
        o_fma[i] = ll.fma_f32(v, y[i], z[i])


@requires_cuda
def test_math_intrin_exports():
    torch.manual_seed(0)
    x = torch.rand(N, dtype=torch.float32, device="cuda") * 8.0
    x[:3] = torch.tensor([0.0, 2.0**-23, 2.0**-22], device=x.device)
    y = torch.randn(N, dtype=torch.float32, device="cuda")
    z = torch.randn(N, dtype=torch.float32, device="cuda")
    outs = [torch.empty(N, dtype=torch.float32, device="cuda") for _ in range(6)]
    kern = math_kernel.build(PASSES["cuda"], codegen_cuda, grid=(N // 256, 1, 1), block=(256, 1, 1))
    kern(x, y, z, *outs)
    kern.synchronize()
    o_sqrt, o_log2, o_rcp, o_sin, o_cos, o_fma = outs

    assert torch.equal(o_sqrt, torch.sqrt(x)), "sqrt_f32 mismatch (IEEE RN)"
    assert torch.allclose(o_rcp, 1.0 / (x + 3.0), rtol=1e-6, atol=0.0), "rcp_f32 mismatch"
    # PTX lg2.approx.f32 specifies absolute error <=2^-22 for input in
    # (0.5, 2), and relative error <=2^-22 outside that interval. Use FP64
    # math for the oracle so Torch's own FP32 approximation is not the bound.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-lg2
    log2_reference = torch.log2((x + 1.0).double())
    log2_bound = 2.0**-22 * log2_reference.abs().clamp_min(1.0)
    assert ((o_log2.double() - log2_reference).abs() <= log2_bound).all(), "log2_f32 exceeds the PTX error bound"
    # PTX sin/cos.approx have absolute (not relative-to-libm) error bounds:
    # 2^-20.5 / 2^-20.4 in [-2*pi, 2*pi], and 2^-14.7 up to 100*pi.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-sin
    import math
    assert (x.abs() <= 100 * math.pi).all()
    near = x.double().abs() <= 2 * math.pi
    sin_bound = torch.where(near, 2.0**-20.5, 2.0**-14.7)
    cos_bound = torch.where(near, 2.0**-20.4, 2.0**-14.7)
    assert ((o_sin.double() - x.double().sin()).abs() <= sin_bound).all(), "sin_f32 exceeds the PTX error bound"
    assert ((o_cos.double() - x.double().cos()).abs() <= cos_bound).all(), "cos_f32 exceeds the PTX error bound"
    fma64 = (x.double() * y.double() + z.double()).float()
    assert torch.equal(o_fma, fma64), "fma_f32 mismatch (single rounding)"
