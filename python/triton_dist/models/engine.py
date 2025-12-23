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
import torch.distributed
from tqdm import tqdm
from datetime import datetime

from triton_dist.kernels.allreduce import AllReduceMethod
from triton_dist.models.kv_cache import KV_Cache
from triton_dist.models import AutoLLM, AutoTokenizer, ModelConfig
from triton_dist.models.utils import logger, sample_token


class Engine:

    def __init__(self, model_config: ModelConfig, temperature: float, top_p: float, verbose: bool = False, group=None):

        self.logger = logger
        self.logger.log("✅ Start Engine...", "success")
        self.model_config = model_config
        self.group = group

        self.temperature = temperature
        self.top_p = top_p
        self.verbose = verbose

        self._init_model()
        self.kv_cache = None
        self.no_graph = False
        self.backend = 'torch'

    def _init_model(self):
        self.logger.log(f"Initializing model {self.model_config}...", "info")
        self.model = AutoLLM.from_pretrained(self.model_config, self.group)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config)
        self.logger.log(f"Model {self.model_config} initialized!", "success")

    def _init_kv_cache(self, bsz: int):
        assert self.kv_cache is None
        self.logger.log("Initializing KV Cache...", "info")
        self.kv_cache = KV_Cache(
            num_layers=self.model.num_layers,
            kv_heads=self.model.num_key_value_heads,
            head_dim=self.model.head_dim,
            batch_size=bsz,
            dtype=self.model.dtype,
            max_length=self.model.max_length,
            world_size=self.model.world_size,
        )
        self.logger.log("KV Cache initialized!", "success")

    def _init_cuda_graph(self, bsz: int = 1):
        # we only init cuda graph for decoding, not for prefilling
        self.logger.log("Capturing CUDA Graph...", "info")
        self.mempool = torch.cuda.graphs.graph_pool_handle()
        static_input_ids = torch.full(
            (bsz, 1), 1225, dtype=torch.long).cuda() if self.backend != 'triton_dist' else torch.full(
                (bsz // self.model.world_size, 1), 1225, dtype=torch.long).cuda()
        static_position_ids = torch.full((bsz, 1), 1225, dtype=torch.long).cuda()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                logits = self.model.inference(input_ids=static_input_ids, position_ids=static_position_ids,
                                              kv_cache=self.kv_cache)
            s.synchronize()
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.mempool):
            logits = self.model.inference(input_ids=static_input_ids, position_ids=static_position_ids,
                                          kv_cache=self.kv_cache)

        def run(input_ids, position_ids):
            static_input_ids.copy_(input_ids)
            static_position_ids.copy_(position_ids)
            graph.replay()
            return logits.clone()

        self.logger.log("CUDA Graph Captured!", "success")
        return run

    def get_ctx(self, input_ids: torch.LongTensor):
        input_len = input_ids.size(1)
        past_len = self.kv_cache.get_kv_len()
        position_ids = past_len[:, None].long() + torch.arange(input_len, device="cuda", dtype=torch.long)
        return position_ids

    def serve(self, input_ids: torch.Tensor, gen_len: int):
        bsz = input_ids.shape[0]
        self.logger.log(f"Benchmarking {self.model.model_name} with prefill {input_ids.shape}, gen_len={gen_len}",
                        "info")
        self._init_kv_cache(bsz=bsz)

        # prefill with torch fwd
        self.kv_cache.clear()
        logits = self.model.inference(input_ids=input_ids.cuda(), position_ids=self.get_ctx(input_ids),
                                      kv_cache=self.kv_cache)
        next_token = sample_token(logits[:, -1, :], temperature=self.temperature, top_p=self.top_p)
        self.kv_cache.kv_offset.fill_(input_ids.shape[-1])

        if self.backend == 'triton_dist':
            next_token = next_token.split(bsz // self.model.world_size, dim=0)[self.model.rank]
            self.model.set_fwd(mode='triton_dist')
            self.model.init_triton_dist_ctx(max_M=bsz)
        elif self.backend == 'triton_dist_AR':
            self.model.set_fwd(mode='triton_dist_AR')
            self.model.init_triton_dist_AR_ctx(max_M=bsz, ar_method=AllReduceMethod.TwoShot_Multimem)
        elif self.backend == 'triton_dist_gemm_ar':
            self.model.set_fwd(mode='triton_dist_gemm_ar')
            self.model.init_triton_dist_gemm_ar_ctx(max_M=bsz)
        if self.no_graph:

            def run(input_ids, position_ids):
                return self.model.inference(input_ids=input_ids, position_ids=position_ids, kv_cache=self.kv_cache)

            self.model_launch = run
        else:
            self.model_launch = self._init_cuda_graph(bsz)

        output_ids = []

        # decode
        step_counter = 0
        profiler = None
        torch.cuda.synchronize()
        torch.distributed.barrier()
        start_time = datetime.now()
        if hasattr(self, "enable_profile") and self.enable_profile:
            self.logger.log("🔨 Profiling enabled for 64 decoding steps...", "info")
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                with_stack=True,
                record_shapes=True,
            )
            profiler.__enter__()

        for _ in tqdm(range(gen_len), desc="Decoding", disable=not hasattr(self, "enable_profile")):
            position_ids = self.get_ctx(next_token)
            logits = self.model_launch(next_token, position_ids)
            next_token = sample_token(logits[:, -1, :], temperature=self.temperature, top_p=self.top_p)
            self.kv_cache.inc_offset()
            output_ids.append(next_token)

            if hasattr(self, "enable_profile") and self.enable_profile:
                step_counter += 1
                if profiler and step_counter >= 64:
                    profiler.__exit__(None, None, None)
                    profiler.export_chrome_trace("trace_static.json")
                    profiler = None
                    self.logger.log("📦 Profiling done and trace saved as trace_static.json.", "success")

        torch.cuda.synchronize()
        torch.distributed.barrier()
        total_latency = (datetime.now() - start_time).total_seconds()
        self.logger.log(f"Decoding finished! Total latency: {total_latency:.2f} s")
        output_ids = torch.cat(output_ids, dim=1).cpu()
        if self.verbose:
            print(self.tokenizer.batch_decode(output_ids, skip_special_tokens=True))

        del self.model_launch
