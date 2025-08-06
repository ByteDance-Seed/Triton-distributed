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

from typing import List, Iterable

import triton
from triton import TritonError
from triton.runtime.autotuner import Autotuner, Config


class _TuningContext:

    def __init__(self, configs: List[Config], bench_iter=None):
        self.configs = configs
        self.bench_iter = bench_iter
        self.finished = False
        self.okay_configs = []
        self.config_times = []


class ContextualAutoTuner:
    _INSTANCE = None

    class KernelError(Exception):
        pass

    def __init__(self, fn, is_dist=False, n_repeat=5, n_warmup=3):
        self.fn = fn
        self.n_repeat = n_repeat
        self.n_warmup = n_warmup
        self.is_dist = is_dist
        self._ctxs: List[_TuningContext] = []
        self._log_file = dict()

    def dist_print(self, *args, **kwargs):
        import torch
        import os

        rank = torch.distributed.get_rank()
        file = self._log_file.get(rank, None)
        if file is None:
            os.makedirs("./.autotune_logs", exist_ok=True)
            file = open(f"./.autotune_logs/rank-{rank}.log", "w")
            self._log_file[rank] = file
        print(f"[rank-{rank}]", *args, **kwargs, file=file, flush=True)

    def __call__(self, *args, **kwargs):
        f_run = lambda: self.fn(*args, **kwargs)

        assert ContextualAutoTuner._INSTANCE is None
        ContextualAutoTuner._INSTANCE = self
        self._ctxs = []
        try:
            while True:
                try:
                    ret = f_run()
                    break
                except self.KernelError:
                    continue
            if len(self._ctxs) <= 0:
                return ret
            while not all(ctx.finished for ctx in self._ctxs):
                # if self.dist:
                #     torch.distributed.barrier()
                try:
                    f_run()
                except self.KernelError:
                    continue
            return f_run()
        finally:
            ContextualAutoTuner._INSTANCE = None
            self._ctxs = []


def contextual_autotune(is_dist=False, n_repeat=5, n_warmup=3):

    def decor(fn):
        return ContextualAutoTuner(fn, is_dist=is_dist, n_repeat=n_repeat, n_warmup=n_warmup)

    return decor


def _do_bench_iterator(funcs, n_repeat=5, n_warmup=3, quantiles=None, return_mode="mean"):
    assert return_mode in ["min", "max", "mean", "median", "all"]
    if not isinstance(funcs, Iterable):
        funcs = [funcs]
    di = triton.runtime.driver.active.get_device_interface()
    device = triton.runtime.driver.active.get_active_torch_device()
    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark().to(device)
    for i, fn in enumerate(funcs):
        try:
            for j in range(n_warmup):
                ret = fn()
                yield ret, i, j, None
            start_event = [di.Event(enable_timing=True) for _ in range(n_repeat)]
            end_event = [di.Event(enable_timing=True) for _ in range(n_repeat)]
            for j in range(n_repeat):
                triton.runtime.driver.active.clear_cache(cache)
                stream = di.current_stream()
                start_event[j].record(stream)
                ret = fn()
                end_event[j].record(stream)
                if j < n_repeat - 1:
                    yield ret, i, j, None
                else:
                    times = [(e.synchronize(), s.elapsed_time(e))[-1] for s, e in zip(start_event, end_event)]
                    yield ret, i, j, triton.testing._summarize_statistics(times, quantiles, return_mode)
        except Exception as e:
            yield None, i, j, e


def _bench_fn(self: Autotuner, *args, config, **meta):
    full_nargs = dict({**self.nargs, **meta}, **config.all_kwargs())

    def kernel_call():
        if config.pre_hook:
            config.pre_hook(full_nargs)
        self.pre_hook(full_nargs)
        try:
            ret = self.fn.run(**full_nargs)
        except Exception as e:
            try:
                self.post_hook(full_nargs, exception=e)
            finally:
                # Throw exception raised by `self.fn.run`
                raise
        self.post_hook(full_nargs, exception=None)
        return ret

    return kernel_call


def _contextual_tuning_run(self: Autotuner, *args, **kwargs):
    self.nargs = dict(zip(self.arg_names, args))

    def f_run(config):
        self.best_config = config
        full_nargs = dict({**self.nargs, **kwargs}, **config.all_kwargs())
        if config.pre_hook is not None:
            # full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
            config.pre_hook(full_nargs)
        ret = self.fn.run(**full_nargs)
        self.nargs = None
        return ret

    def f_key():
        all_args = {**self.nargs, **kwargs}
        _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
        key = [_args[key] for key in self.keys if key in _args]
        ks = [key for key in self.keys if key in _args]
        kvs = dict(zip(ks, key))
        for _, arg in _args.items():
            if hasattr(arg, "dtype"):
                key.append(str(arg.dtype))
        return tuple(key), kvs

    if len(self.configs) <= 1:
        return f_run(self.configs[0])

    ctx: _TuningContext = getattr(self, "_tuning_context", None)
    ctx_tuner = ContextualAutoTuner._INSTANCE
    key, kvs = f_key()
    if ctx is None:
        config = self.cache.get(key, None)
        if config is not None:
            return f_run(config)

        pruned_configs = self.prune_configs(kwargs)
        bench_fns = [_bench_fn(self, *args, config=config, **kwargs) for config in pruned_configs]

        bench_iter = _do_bench_iterator(bench_fns, n_repeat=ctx_tuner.n_repeat, n_warmup=ctx_tuner.n_warmup,
                                        return_mode="mean")
        ctx = self._tuning_context = _TuningContext(pruned_configs, bench_iter)
        ctx_tuner._ctxs.append(ctx)

    while True:
        ret, cfg_i, iter_j, ms = next(ctx.bench_iter)
        msg = f"func: {self.fn.__name__} | key: {kvs} | config-id: {cfg_i} | config: {{{ctx.configs[cfg_i]}}} | measure-iter: {iter_j}"
        if isinstance(ms, Exception):
            msg += f" | error: {ms}"
        elif ms is not None:
            msg += f" | config-{cfg_i} average latency: {ms} ms"
        ctx_tuner.dist_print(msg)
        if not isinstance(ms, Exception):
            break
        if not isinstance(ms, TritonError):
            raise ctx_tuner.KernelError("kernel launch failed")
        if cfg_i >= len(ctx.configs) - 1:
            break

    if ms is not None:
        if not isinstance(ms, Exception):
            ctx.okay_configs.append((cfg_i, ctx.configs[cfg_i]))
            ctx.config_times.append(ms)
        if cfg_i >= len(ctx.configs) - 1:
            if len(ctx.okay_configs) <= 0:
                raise RuntimeError("cannot find valid config")
            if ctx_tuner.is_dist:
                import torch

                times_tensor = torch.tensor(ctx.config_times, device="cuda")
                torch.distributed.all_reduce(times_tensor, torch.distributed.ReduceOp.MAX)
                ctx.config_times = times_tensor.tolist()

            (self.best_config_id, self.best_config), self.best_time = min(zip(ctx.okay_configs, ctx.config_times),
                                                                          key=lambda t: t[-1])
            ctx_tuner.dist_print(
                f"func: {self.fn.__name__} | key: {kvs} | best-config-id: {self.best_config_id} | best-config: {{{self.best_config}}} | best-latency: {self.best_time} ms"
            )
            self.cache[key] = self.best_config
            self.configs_timings = ctx.config_times
            self._tuning_context = None
            ctx.finished = True

    self.nargs = None
    return ret


_old_autotuner_run = Autotuner.run


def _new_autotuner_run(self: Autotuner, *args, **kwargs):
    if ContextualAutoTuner._INSTANCE is not None:
        return _contextual_tuning_run(self, *args, **kwargs)
    return _old_autotuner_run(self, *args, **kwargs)


Autotuner.run = _new_autotuner_run
