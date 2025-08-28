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

from transformers import AutoTokenizer as HFTokenizer

from .config import ModelConfig
from .dense import DenseLLM
from .qwen_moe import Qwen3MoE


class AutoLLM:
    model_mapping = {
        "Qwen/Qwen3-0.6B": DenseLLM,
        "Qwen/Qwen3-8B": DenseLLM,
        "Qwen/Qwen3-14B": DenseLLM,
        "Qwen/Qwen3-32B": DenseLLM,
        "Qwen/Qwen3-30B-A3B": Qwen3MoE,
        "Qwen/Qwen3-235B-A22B": Qwen3MoE,
        "meta-llama/Meta-Llama-3-70B": DenseLLM,
        "ByteDance-Seed/Seed-OSS-36B-Instruct": DenseLLM,
    }

    @staticmethod
    def from_pretrained(config: ModelConfig, group=None):
        model_name = config.model_name

        for standard_name in AutoLLM.model_mapping.keys():
            if model_name.endswith(standard_name):
                return AutoLLM.model_mapping[standard_name](config, group)

        if model_name in AutoLLM.model_mapping:
            return AutoLLM.model_mapping[model_name](config, group)
        else:
            print(f"Model {model_name} not found in model mapping, "
                  f"Available models: {list(AutoLLM.model_mapping.keys())} "
                  f"Falling back to DenseLLM with default configuration.")
            return DenseLLM(config, group)


class AutoTokenizer:

    def __init__(self):
        self.tokenizer = None

    @staticmethod
    def from_pretrained(model_config):
        return HFTokenizer.from_pretrained(model_config.model_name, use_fast=True, legacy=False,
                                           local_files_only=model_config.local_only)
