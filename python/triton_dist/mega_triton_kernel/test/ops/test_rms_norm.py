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
import os
import argparse
import torch
from triton_dist.mega_triton_kernel import ModelBuilder
from triton_dist.profiler_utils import get_torch_prof_ctx
from triton_dist.mega_triton_kernel.test.torch_impl_utils import rmsnorm_ref


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=False, action="store_true", help="enable profiling")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.cuda.set_device(0)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    profile = args.profile
    l2_cache = torch.randn((256, 1024, 1024)).cuda()

    builder = ModelBuilder()
    batch = 3
    seq_len = 1
    hidden_size = 5120
    dtype = torch.bfloat16
    rms_eps = 1e-6

    x = torch.randn((batch, seq_len, hidden_size), dtype=dtype, device=torch.cuda.current_device())
    rms_weight = torch.randn((hidden_size, ), dtype=dtype, device=torch.cuda.current_device())
    rms_out = torch.empty((batch, seq_len, hidden_size), dtype=dtype, device=torch.cuda.current_device())

    builder.make_rms_norm(x, rms_weight, rms_out, rms_eps)
    builder.compile()

    ctx = get_torch_prof_ctx(profile)
    with ctx:
        for i in range(30):
            l2_cache.zero_()
            tmp_input = torch.randn(x.shape, dtype=dtype).cuda()
            x.copy_(tmp_input)
            builder.run()

            # torch impl
            l2_cache.zero_()

            rms_out_ref = rmsnorm_ref(x, rms_weight, rms_eps)
            torch.testing.assert_close(rms_out_ref, rms_out, atol=0, rtol=0)

    if profile:
        import os
        prof_dir = "prof/"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/rms_norm.json.gz")
