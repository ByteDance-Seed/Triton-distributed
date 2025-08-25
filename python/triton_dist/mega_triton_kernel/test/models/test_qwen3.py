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
import argparse
import os
import torch
import triton
from triton_dist.mega_triton_kernel import ModelBuilder
from triton_dist.mega_triton_kernel.models import Qwen3Model

from triton_dist.utils import get_torch_prof_ctx
from triton_dist.models import ModelConfig
from triton_dist.models.engine import Engine, AutoTokenizer
from triton_dist.models.utils import sample_token
from triton_dist.utils import (
    initialize_distributed,
    finalize_distributed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-32B", type=str, help="HuggingFace model name")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument("--backend", default="mega_kernel", type=str,
                        choices=["mega_kernel", "triton_dist_AR", "torch"], help="backend")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--profile", default=False, action="store_true", help="enable kernel level profiling")
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--intra_kernel_profile", default=False, action="store_true",
                        help="enable intra kernel profiling")

    return parser.parse_args()


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

if __name__ == "__main__":
    args = parse_args()
    TP_GROUP = initialize_distributed(seed=0)
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    assert args.dtype == "bfloat16"

    dtype = DTYPE_MAP[args.dtype]
    model_config = ModelConfig(model_name=args.model, max_length=args.max_length, dtype=dtype, rank=RANK,
                               world_size=WORLD_SIZE)

    builder = ModelBuilder(rank=RANK, world_size=WORLD_SIZE, local_world_size=LOCAL_WORLD_SIZE,
                           enable_profiling=args.intra_kernel_profile)
    batch_size = args.batch_size
    history = []
    ctx = get_torch_prof_ctx(args.profile)
    history = []
    user_input = "What is the capital of France?"
    history.append({"role": "user", "content": user_input})
    tokenizer = AutoTokenizer.from_pretrained(model_config)
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>\n' }}{% endif %}"
    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().repeat(batch_size, 1)
    gen_len = 512
    with ctx:
        if args.backend != "mega_kernel":
            engine = Engine(model_config, temperature=args.temperature, top_p=args.top_p, verbose=True)
            engine.backend = args.backend
            engine.serve(input_ids=input_ids, gen_len=gen_len)
        else:
            qwen3 = Qwen3Model(batch_size, model_config, builder, build_lm_head=True)

            input_seq_len = input_ids.shape[1]
            output_ids = []

            for idx in range(gen_len + input_seq_len):
                qwen3.kv_cache.inc_offset(1)
                if idx < input_seq_len:
                    next_token = input_ids[:, idx].contiguous().reshape(-1, 1)
                    logits = qwen3.mega_forwrad(next_token)
                    if idx == input_seq_len - 1:
                        next_token = sample_token(logits[:, -1, :], temperature=args.temperature, top_p=args.top_p)
                else:
                    logits = qwen3.mega_forwrad(next_token)
                    next_token = sample_token(logits[:, -1, :], temperature=args.temperature, top_p=args.top_p)
                if idx >= input_seq_len - 1:
                    output_ids.append(next_token)
                    next_token_cpu = next_token[0, -1].cpu()
                    if next_token_cpu.item() == qwen3.eos_token_id:
                        break

            output_ids = torch.cat(output_ids, dim=1).cpu().tolist()
            if RANK == 0:
                print(f"output = {tokenizer.batch_decode(output_ids, skip_special_tokens=True)}")
            torch.cuda.synchronize()
            if args.intra_kernel_profile:
                builder.dump_trace()
            builder.finalize()
    if args.profile:
        import os
        prof_dir = f"prof/qwen3_model_{args.model.split('-')[-1]}_bs_{batch_size}_tp{WORLD_SIZE}_backend{args.backend}/"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/rank_{RANK}.json.gz")
    torch.distributed.barrier()
    finalize_distributed()
