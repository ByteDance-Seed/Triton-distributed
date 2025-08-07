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

import torch
import os
from argparse import ArgumentParser, Namespace

import pyrocshmem
from triton_dist.models import ModelConfig
from triton_dist.models.engine import Engine
from triton_dist.models.utils import seed_everything


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen3-32B")
    p.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    p.add_argument("--bsz", type=int, default=8, help="Batch size for inference")
    p.add_argument("--gen_len", type=int, default=256, help="Length of generated tokens")
    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--profile", action="store_true", help="Enable profiling")
    p.add_argument("--no_graph", action="store_true", help="Disable CUDA graph")
    p.add_argument("--triton_dist", action="store_true", help="Use triton_dist for distributed inference")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return p.parse_args()


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
    )
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    torch.distributed.barrier(TP_GROUP)
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.set_printoptions(precision=2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    pyrocshmem.init_rocshmem_by_uniqueid(TP_GROUP)
    current_stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    DTYPE = DTYPE_MAP[args.dtype]

    model_config = ModelConfig(model_name=args.model, max_length=args.max_length, dtype=DTYPE, rank=RANK,
                               world_size=WORLD_SIZE)
    bsz = args.bsz
    assert bsz % WORLD_SIZE == 0, "Batch size must be divisible by world size for distributed inference."
    engine = Engine(model_config, temperature=0.0, top_p=0.95, verbose=args.verbose, group=TP_GROUP)

    if args.profile:
        engine.enable_profile = True
    if args.no_graph:
        engine.no_graph = True
        engine.logger.log("❌ CUDA graph disabled!", "warning")

    prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n"
    input_ids = engine.tokenizer(prompt, return_tensors="pt").input_ids.cuda().repeat(bsz, 1)
    gen_len = args.gen_len

    if args.triton_dist:
        engine.logger.log("🔗 Using triton_dist for distributed inference.", "info")
        engine.backend = 'triton_dist'

    engine.serve(input_ids=input_ids, gen_len=gen_len)
    engine.logger.log("✅ Inference completed!", "success")

    pyrocshmem.rocshmem_finalize()
    torch.distributed.destroy_process_group()
