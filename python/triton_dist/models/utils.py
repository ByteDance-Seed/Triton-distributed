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
from termcolor import colored
from datetime import datetime
import logging
import numpy as np
import random
import os
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights

if torch.version.cuda:
    PLATFORM = 'nvidia'
    import flashinfer
elif torch.version.hip:
    PLATFORM = 'amd'
else:
    raise RuntimeError("Unsupported platform: neither CUDA nor HIP is available.")


class MyLogger:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.propagate = False
        if os.getenv('TRITON_DIST_DEBUG', '').lower() in ('true', '1', 't'):
            self.logger.setLevel(logging.DEBUG)
            self.logger.debug("Debug logging enabled")

    def log(self, msg, level="info"):
        if level == "info":
            self.logger.info(colored(f"> [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", "cyan"))
        elif level == "warning":
            self.logger.warning(colored(f"> [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", "yellow"))
        elif level == "error":
            self.logger.error(colored(f"> [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", "red"))
        elif level == "success":
            self.logger.info(colored(f"> [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", "green"))
        elif level == "debug":
            self.logger.debug(colored(f"> [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", "magenta"))
        else:
            raise ValueError(
                colored(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Unknown log level: {level}", "red"))


logger = MyLogger()


def seed_everything(seed):
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@torch.inference_mode()
def sample_token(logits: torch.Tensor, temperature=0.6, top_p=0.95, top_k=-1):
    if PLATFORM == 'nvidia':
        if temperature == 0.0:
            token = logits.argmax(dim=-1, keepdim=True)
        else:
            if temperature != 1.0:
                logits = logits / temperature
            assert top_k == -1
            probs = logits.softmax(dim=-1)
            token = flashinfer.sampling.top_p_sampling_from_probs(probs=probs, top_p=top_p)
            token = token.unsqueeze(-1)
    elif PLATFORM == 'amd':
        if temperature == 0.0:
            token = logits.argmax(dim=-1, keepdim=True)
        else:
            raise NotImplementedError(
                "AMD platform does not support temperature sampling yet. Please use temperature=0.0 for argmax sampling."
            )
    return token


@torch.no_grad()
def init_model_cpu(model_name: str, dtype: torch.dtype):
    with torch.no_grad():
        random_params = os.environ.get("RANDOM_PARAMS", "0").lower() in ("1", "true", "yes")
        if random_params:
            print("Initializing model with random parameters. This may take a while...")
            config = AutoConfig.from_pretrained(model_name)
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype,
                                                         attn_implementation="flash_attention_2")
            model.to_empty(device="cuda")
            model.init_weights()
            return model

        else:
            print("Loading model from pre-trained weights. This may take a while...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
            )
            return model
