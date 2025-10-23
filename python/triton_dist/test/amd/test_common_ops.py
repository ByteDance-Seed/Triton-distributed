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
import random

import torch

from triton_dist.kernels.amd.common_ops import barrier_on_this_grid, cooperative_barrier_on_this_grid
from triton_dist.utils import launch_cooperative_grid_options, support_launch_cooperative_grid


def test_barrier_on_this_grid_without_launch_cooperative_grid():
    # If launch_cooperative_grid is False, then the process just segment fault, like this:
    #  Memory access fault by GPU node-2 (Agent handle: 0x560c18b50270) on address (nil). Reason: Unknown.
    # GPU core dump created: gpucore.400609
    if support_launch_cooperative_grid():
        cooperative_barrier_on_this_grid[(random.randint(1, 128), )](launch_cooperative_grid=False)
        torch.cuda.synchronize()
        print("✅ cooperative_barrier_on_this_grid with launch_cooperative_grid=False passed")


def test_barrier_on_this_grid():
    print(">> barrier_on_this_grid start...")
    flag = torch.zeros((1, ), dtype=torch.int32, device="cuda")
    for _ in range(100):
        barrier_on_this_grid[(random.randint(1, 128), )](flag, use_cooperative=False,
                                                         **launch_cooperative_grid_options())
    print("✅ barrier_on_this_grid passed")

    for _ in range(100):
        cooperative_barrier_on_this_grid[(random.randint(1, 128), )](**launch_cooperative_grid_options())
    print("✅ cooperative_barrier_on_this_grid passed")

    # don't run this in CI. it will crash.
    # test_barrier_on_this_grid_without_launch_cooperative_grid()


test_barrier_on_this_grid()
