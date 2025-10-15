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
from typing import List
import torch
import torch.distributed as dist

from triton_dist.models.utils import init_model_cpu
from triton_dist.layers.nvidia.tp_mlp import TP_MLP
from triton_dist.layers.nvidia.pp_block import PPCommLayer
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import assert_allclose
from triton_dist.utils import (dist_print, finalize_distributed, initialize_distributed)

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
}

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def make_cuda_graph(mempool, func):
    """Create CUDA graph for performance benchmarking"""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(30):
            func()
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        func()
    return graph


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", default=8, type=int, help="batch size")
    parser.add_argument("--seq_len", default=128, type=int, help="sequence length")
    parser.add_argument("--num_blocks", default=2, type=int, help="number of transformer blocks to test")
    parser.add_argument("--pp_size", default=2, type=int, help="pipeline parallel size")
    parser.add_argument("--num_micro_batches", default=None, type=int,
                        help="number of micro-batches (default: pp_size * 2)")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", type=str, help="HuggingFace model name")
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=20, type=int, help="performance iterations")
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP.keys()), help="data type")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", action="store_true", help="enable profiling")
    parser.add_argument("--num_sms", default=8, type=int, help="number of SMs for PP communication")

    return parser.parse_args()


def setup_distributed_groups(pp_size: int):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    tp_size = world_size // pp_size
    all_ranks = list(range(world_size))
    pp_group, tp_group = None, None
    for i in range(tp_size):
        group = dist.new_group(all_ranks[i::tp_size])
        if rank in all_ranks[i::tp_size]: pp_group = group
    for i in range(0, world_size, tp_size):
        group = dist.new_group(all_ranks[i:i + tp_size])
        if rank in all_ranks[i:i + tp_size]: tp_group = group
    return tp_group, pp_group, dist.get_rank(group=tp_group), dist.get_rank(group=pp_group), tp_size


def rand_tensor(shape: list[int], dtype: torch.dtype):
    if dtype in [torch.int32, torch.int8]:
        return torch.randint(-127, 128, shape, dtype=dtype).cuda()
    else:
        return torch.rand(shape, dtype=dtype).cuda() / 10


class PipelinedStage(torch.nn.Module):

    def __init__(self, layers, pp_comm, tp_rank, tp_size):
        super().__init__()
        self.layers: List[TP_MLP] = layers
        self.pp_comm: PPCommLayer = pp_comm
        self.pp_rank = self.pp_comm.pp_rank
        self.pp_size = self.pp_comm.pp_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.backend = 'torch'

        self.recv_buf = None
        self.recv_buf_shape = None
        self.send_buf = None
        if self.pp_comm.is_last_stage():
            self.outputs_micro = []

    def set_backend(self, backend: str):
        self.backend = backend
        # If switching to triton_dist backend, ensure PP comm is initialized collectively
        if backend == 'triton_dist':
            self.pp_comm.init_triton_backend_collective()

    def init_dist_contexts(self, max_M, dtype):
        """Initialize distributed triton contexts."""
        # No initialization needed for torch mode
        pass

    def _setup_buffers(self, micro_shape, dtype):
        """Initializes buffers for communication."""
        if not self.pp_comm.is_first_stage():
            self.recv_buf_shape = micro_shape
        if not self.pp_comm.is_last_stage():
            self.send_buf = torch.empty(micro_shape, dtype=dtype, device="cuda")

    def _compute_step(self, hidden_states):
        """The pure computation logic for one micro-batch."""
        for layer in self.layers:
            hidden_states = layer.torch_fwd(x=hidden_states)

        return hidden_states

    def _step(self, micro_input=None):
        """Executes one fundamental step of the pipeline for the current stage."""
        if not self.pp_comm.is_first_stage():
            received_tensor_2d = self.pp_comm.recv(
                self.pp_comm.get_prev_rank(),
                expected_shape=(self.recv_buf_shape[0] * self.recv_buf_shape[1], self.recv_buf_shape[2]),
                backend=self.backend)
            self.recv_buf = received_tensor_2d.view(self.recv_buf_shape)

        current_input = micro_input if self.pp_comm.is_first_stage() else self.recv_buf
        compute_output = self._compute_step(current_input)

        if self.pp_comm.is_last_stage():
            self.outputs_micro.append(compute_output)

        if not self.pp_comm.is_last_stage():
            self.pp_comm.send(compute_output.view(-1, compute_output.size(-1)), self.pp_comm.get_next_rank(),
                              backend=self.backend)

    def forward(self, inputs_full=None, num_micro_batches=0):
        if self.pp_comm.is_last_stage():
            self.outputs_micro = []

        inputs_list = []
        if self.pp_comm.is_first_stage():
            assert inputs_full is not None, "inputs_full must be provided to the first stage"
            inputs_list = torch.chunk(inputs_full, num_micro_batches)

        # A pipeline of depth P processing M micro-batches takes M + P - 1 total steps.
        for i in range(num_micro_batches + self.pp_size - 1):
            micro_input = None
            if self.pp_comm.is_first_stage():
                if i < num_micro_batches:
                    micro_input = inputs_list[i]

            # A stage is active if a micro-batch is scheduled for it in the current step.
            # Stage `r` starts at time `r` and finishes at time `r + num_micro_batches`.
            if self.pp_rank <= i < (self.pp_rank + num_micro_batches):
                self._step(micro_input)

        if self.pp_comm.is_last_stage():
            # For ag_rs mode, output is split, need to gather
            return torch.cat(self.outputs_micro, dim=0)
        return None


if __name__ == "__main__":
    args = parse_args()

    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    GROUP = initialize_distributed(args.seed, initialize_shmem=True)

    DTYPE = DTYPE_MAP[args.dtype]
    ATOL = THRESHOLD_MAP[DTYPE]
    RTOL = THRESHOLD_MAP[DTYPE]
    tp_group, pp_group, tp_rank, pp_rank, tp_size = setup_distributed_groups(args.pp_size)

    hf_model = init_model_cpu(model_name=args.model, dtype=DTYPE)

    layers_per_stage = args.num_blocks // args.pp_size
    start_layer = pp_rank * layers_per_stage
    end_layer = (pp_rank + 1) * layers_per_stage

    stage_layers = []
    for idx in range(start_layer, end_layer):
        layer = TP_MLP(rank=tp_rank, world_size=tp_size, group=tp_group)
        layer._init_parameters(mlp=hf_model.model.layers[idx].mlp, verbose=False)
        stage_layers.append(layer)

    dist_print(f"RANK={RANK}, pp_rank={pp_rank}, tp_rank={tp_rank}, pp_size={args.pp_size}, tp_size={tp_size}")

    # NUM_MICRO_BATCHES can be configured independently from pp_size
    NUM_MICRO_BATCHES = args.num_micro_batches if args.num_micro_batches is not None else args.pp_size * 2
    assert args.bsz % NUM_MICRO_BATCHES == 0, f"Batch size {args.bsz} must be divisible by NUM_MICRO_BATCHES {NUM_MICRO_BATCHES}"
    micro_bsz = args.bsz // NUM_MICRO_BATCHES
    dist_print(f"Using NUM_MICRO_BATCHES={NUM_MICRO_BATCHES}, micro_bsz={micro_bsz}")
    micro_max_tokens = micro_bsz * args.seq_len

    pp_comm = PPCommLayer(
        pp_rank=pp_rank,
        pp_size=args.pp_size,
        pp_group=pp_group,
        max_tokens=micro_max_tokens,
        hidden_size=hf_model.config.hidden_size,
    )

    pipelined_stage = PipelinedStage(stage_layers, pp_comm, tp_rank, tp_size)

    micro_shape = (micro_bsz, args.seq_len, hf_model.config.hidden_size)
    pipelined_stage._setup_buffers(micro_shape, DTYPE)

    # Initialize distributed contexts
    max_M = micro_bsz * args.seq_len
    pipelined_stage.init_dist_contexts(max_M=max_M, dtype=DTYPE)

    full_shape = (args.bsz, args.seq_len, hf_model.config.hidden_size)
    initial_hidden_states = rand_tensor(full_shape, dtype=DTYPE) if pp_rank == 0 else torch.empty(
        full_shape, device="cuda", dtype=DTYPE)

    # Correctness Check
    dist.barrier()
    dist.broadcast(initial_hidden_states, src=0)

    golden_output = torch.empty_like(initial_hidden_states)
    if RANK == 0:
        golden_hidden_states = initial_hidden_states
        with torch.inference_mode():
            for i in range(args.num_blocks):
                layer = hf_model.model.layers[i].mlp.to("cuda")
                golden_hidden_states = layer(golden_hidden_states)
        golden_output.copy_(golden_hidden_states)

    dist.broadcast(golden_output, src=0)

    pipelined_stage.set_backend('triton_dist')
    pipelined_output_triton_dist_p2p = pipelined_stage.forward(inputs_full=initial_hidden_states,
                                                               num_micro_batches=NUM_MICRO_BATCHES)

    if pipelined_stage.pp_comm.is_last_stage():
        assert_allclose(pipelined_output_triton_dist_p2p, golden_output, atol=ATOL, rtol=RTOL)
    dist.barrier()

    # Efficiency: Benchmark full forward passes
    dist_print(f"Benchmarking full forward pass with {NUM_MICRO_BATCHES} micro-batches...")

    # Create input for benchmarking
    benchmark_input = None
    if pipelined_stage.pp_comm.is_first_stage():
        benchmark_input = rand_tensor(full_shape, dtype=DTYPE)

    dist.barrier()

    with group_profile("pp_block", args.profile, group=GROUP):

        # Benchmark PyTorch backend
        pipelined_stage.set_backend('torch')
        torch.cuda.synchronize()
        dist.barrier()

        def torch_forward():
            output = pipelined_stage.forward(inputs_full=benchmark_input, num_micro_batches=NUM_MICRO_BATCHES)
            # Clear outputs to avoid memory buildup
            if pipelined_stage.pp_comm.is_last_stage():
                pipelined_stage.outputs_micro = []
            return output

        _, torch_perf = perf_func(torch_forward, iters=args.iters, warmup_iters=args.warmup)

        # Benchmark Triton-dist backend
        pipelined_stage.set_backend('triton_dist')
        torch.cuda.synchronize()
        dist.barrier()

        def triton_forward():
            output = pipelined_stage.forward(inputs_full=benchmark_input, num_micro_batches=NUM_MICRO_BATCHES)
            # Clear outputs to avoid memory buildup
            if pipelined_stage.pp_comm.is_last_stage():
                pipelined_stage.outputs_micro = []
            return output

        mempool = torch.cuda.graph_pool_handle()
        triton_dist_graph = make_cuda_graph(mempool, triton_forward)

        _, dist_triton_perf = perf_func(triton_dist_graph.replay, iters=args.iters, warmup_iters=args.warmup)

    dist_print(f"torch pp forward #{RANK}", f"{torch_perf:.4f} ms", need_sync=True,
               allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"dist-triton pp forward #{RANK}", f"{dist_triton_perf:.4f} ms", f"{torch_perf/dist_triton_perf:.2f}x",
               need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    # Cleanup
    for layer in pipelined_stage.layers:
        layer.finalize()

    del triton_dist_graph, mempool

    finalize_distributed()
