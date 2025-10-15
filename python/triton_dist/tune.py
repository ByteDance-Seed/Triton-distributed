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

import datetime
import functools
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import subprocess
from contextlib import contextmanager
from typing import Callable
import warnings

from pathlib import Path
import torch

import triton
import triton_dist
from triton_dist.utils import (is_cuda, get_nvshmem_hash, get_nvshmem_version, get_rocshmem_hash, get_rocshmem_version,
                               get_triton_dist_world, get_cpu_info_linux, triton_dist_key, warn_if_cuda_launch_blocking,
                               get_bool_env, get_int_env, barrier_async, wait_until_max_gpu_clock_or_warning)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.WARN)
logger.addHandler(stream_handler)


@functools.lru_cache(maxsize=1)
def _autotune_always_tune():
    always_compile = os.getenv("TRITON_DIST_AUTOTUNE_ALWAYS_TUNE", "0") == "1"
    return always_compile


@contextmanager
def pretty_triton_config_repr():
    old_repr = triton.Config.__repr__
    triton.Config.__repr__ = triton.Config.__str__
    try:
        yield
    finally:
        triton.Config.__repr__ = old_repr


@contextmanager
def log_to_file(filename: str):
    handler = logging.FileHandler(filename)
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)


@contextmanager
def set_stream_handler_log_level(level: int):
    old_level = stream_handler.level
    stream_handler.setLevel(level)
    try:
        yield
    finally:
        stream_handler.setLevel(old_level)


def _check_autotune_version():
    return get_bool_env("TRITON_DIST_AUTOTUNE_VERSION_CHECK", False)


def get_git_info(pkg_name="triton_dist"):

    def _run(cmds):
        return subprocess.check_output(cmds, stderr=subprocess.DEVNULL).decode().strip()

    try:
        # locate package path
        spec = importlib.util.find_spec(pkg_name)
        if spec is None or not spec.origin:
            return f"Package {pkg_name} not found"

        pkg_path = os.path.dirname(spec.origin)

        # walk up until you find .git
        cur = pkg_path
        while cur != "/" and cur:
            if os.path.isdir(os.path.join(cur, ".git")):
                try:
                    commit = _run(["git", "-C", cur, "rev-parse", "HEAD"])
                    branch = _run(["git", "-C", cur, "rev-parse", "--abbrev-ref", "HEAD"])
                    return branch, commit
                except Exception as e:
                    print(f"Found .git under {cur}, but cannot read commit: {e}")
                    return None, None
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        print(f"{pkg_name} is installed from {pkg_path}, but no .git repo found")
        return None, None
    except Exception as e:
        print(f"Error checking {pkg_name}: {e}")
        return None, None


def to_hashable(t: torch.Tensor):
    return tuple(t.shape), t.stride(), t.dtype


class TuneRecordEncoder(json.JSONEncoder):

    def default(self, o):
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if isinstance(o, triton.Config):
            # print("encode triton.config: ", o)
            return {"__triton_config__": o.__dict__}
        if isinstance(o, torch.dtype):
            return {"__torch_dtype__": o.__str__()}
        return super().default(o)


def _torch_dtype_from_str(s: str):
    return {
        "torch.float16": torch.float16,
        "torch.float": torch.float,
        "torch.bfloat16": torch.bfloat16,
        "torch.float8_e4m3fn": torch.float8_e4m3fn,
        "torch.float8_e5m2": torch.float8_e5m2,
        "torch.int32": torch.int32,
        "torch.int64": torch.int64,
    }[s]


def from_json(obj):
    if isinstance(obj, dict):
        if v := obj.get("__triton_config__", None):
            return triton.Config(**v)
        if v := obj.get("__torch_dtype__", None):
            return _torch_dtype_from_str(v)
    return obj


def load_autotune_data(filename: str | Path):
    try:
        with open(filename, "r") as f:
            return json.load(f, object_hook=from_json)
    except FileNotFoundError:
        logger.warning(f"Autotune data file {filename} not found.")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error loading autotune data file {filename}: {e}")
        return None


def store_autotune_data(filename, func_name, key, timings):
    with open(filename, "w") as f:
        json.dump(
            {
                "config": timings,
                "deps": get_deps(),
                "key": key,
                "func": func_name,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"),
            },
            f,
            indent=2,
            cls=TuneRecordEncoder,
        )


def get_triton_dist_version():
    try:
        return triton_dist.__version__
    except Exception:
        return "0.2.0"


def get_cuda_extra_args():
    return {
        "CUDA_DEVICE_MAX_CONNECTIONS": get_int_env("CUDA_DEVICE_MAX_CONNECTIONS", None),
        "nvshmem.version": get_nvshmem_version(),
        "nvshmem.hash": get_nvshmem_hash(),
        "NVSHMEM_DISABLE_CUDA_VMM": get_bool_env("NVSHMEM_DISABLE_CUDA_VMM", None),
    }


def get_rocm_extra_args():
    return {
        "GPU_MAX_HW_QUEUES": get_int_env("GPU_MAX_HW_QUEUES", None),
        "rocshmem.version": get_rocshmem_version(),
        "rocshmem.hash": get_rocshmem_hash(),
    }


@functools.lru_cache()
def get_deps():
    args = {
        # hardware
        "cpu": get_cpu_info_linux(),
        # software version
        "torch.version": torch.__version__,
        "triton.version": triton.__version__,
        "triton.hash": triton.compiler.compiler.triton_key(),
        "triton_dist.version": get_triton_dist_version(),
        "triton_dist.hash": triton_dist_key(),
    }
    if is_cuda():
        args.update(get_cuda_extra_args())
    else:
        args.update(get_rocm_extra_args())
    return args


def _compare_deps(deps):
    current_deps = get_deps()
    for k, v in current_deps.items():
        if v != (vv := deps.get(k, None)):
            warnings.warn(f"{k} mismatch, expect {v}, got {vv}.")
            return False
    return True


def get_hardware_info(with_nic):
    hw_info = {
        "gpu.count": torch.cuda.device_count(),
        "gpu.name": torch.cuda.get_device_name(),
    }
    if is_cuda():
        from triton_dist.utils import has_fullmesh_nvlink
        hw_info.update({"nvlink.fullmesh": has_fullmesh_nvlink()})
        if not has_fullmesh_nvlink():
            # TODO(houqi.1993) add the PCI-e topo into this
            pass
    else:
        hw_info.update({})

    if with_nic:
        # TODO(houqi.1993) add the NIC info into this
        pass
    return hw_info


@functools.lru_cache()
def hw_hash(with_nic=False):
    return hashlib.sha256(json.dumps(get_hardware_info(with_nic), sort_keys=True).encode("utf-8")).hexdigest()


class AutoTuner:

    def __init__(self, func, config_space, key_fn: Callable, prune_fn: Callable | None = None):
        """

        func is Callable, not a lambda or partial
        """
        assert config_space, "config_space is empty"
        self.fn = func
        self.config_space = config_space
        self.key_fn = key_fn
        self.prune_fn = prune_fn
        self.best_configs = {}
        # Caching components
        self.cache_dir = Path.home() / ".triton_dist" / "autotune"
        try:
            # TODO(houqi.1993) source code is not enough for import different functions with same name
            self.func_hash = hashlib.sha256(inspect.getsource(func).encode("utf-8")).hexdigest()
            # more human friendly path tag: even function content updated with the same name
            self.func_dir_name = f"{func.__name__}"
        except (TypeError, OSError) as e:  # For built-ins, lambdas in repl, etc.
            logger.error(f"Failed to get source code for function {func}: {e}")
            self.func_hash = func.__name__
            self.func_dir_name = func.__name__
        warn_if_cuda_launch_blocking()

    def func_dir(self):
        return self.cache_dir / self.func_dir_name

    def get_pruned_config(self, *args, **kwargs):
        config_space = self.config_space
        if self.prune_fn:  # you can prune by (*args, **kwargs) too
            N = len(config_space)
            config_space = [x for x in config_space if self.prune_fn(x, *args, **kwargs)]
            logger.info(f"pruned config space from {N} to {len(config_space)}")
        return config_space

    @staticmethod
    def _has_duplicate_kwargs(kwargs1, kwargs2):
        return set(kwargs1).intersection(kwargs2)

    def __call__(
        self,
        *args,
        autotune=True,
        autotune_verbose=False,
        autotune_allow_arg_overwrite=False,
        autotune_pg: torch.distributed.ProcessGroup | None = None,
        **kwargs,
    ):
        """
        autotune:
            If True, will use autotune config_space to tune the function. else use the first config from config_space
        autotune_verbose:
            If True, will print more verbose log.
        autotune_allow_arg_overwrite:
            If True, allow run the tuned function with keyword arguments in AutoTune config_space and overwrite the value in tuned function.
        """
        with pretty_triton_config_repr(), set_stream_handler_log_level(
                logging.INFO if autotune_verbose else logging.WARNING):
            config_space = self.config_space

            if dups := AutoTuner._has_duplicate_kwargs(kwargs, config_space[0]):
                if autotune_allow_arg_overwrite:
                    warnings.warn(
                        "Duplicate keyword arguments, but autotune_allow_arg_overwrite is True, will use the argument provided by the user."
                    )
                else:
                    raise ValueError(f"Duplicate keyword arguments {dups}, and autotune_allow_arg_overwrite is False.")

            if not autotune:
                return self.fn(
                    *args,
                    **{**config_space[0], **kwargs},
                )

            key = self.key_fn(*args, **kwargs) if self.key_fn else None
            best_config = self.best_configs.get(key, None)
            if best_config is None:
                timings = self.load_tune_config(key)
                if timings is None or _autotune_always_tune():
                    logger.info(f"Tuning for key {key}")
                    autotune_pg = autotune_pg or get_triton_dist_world()
                    config_space = self.get_pruned_config(*args, **kwargs)
                    assert config_space
                    with log_to_file(self._get_tune_log_path(key, autotune_pg)):
                        logger.info(f"tuning config space: {config_space} for func {self.fn.__name__}")
                        logger.info(f"key: {key}")
                        logger.info(f"deps: {get_deps()}")
                        wait_until_max_gpu_clock_or_warning()
                        timings = self.tune(config_space, autotune_pg, *args, **kwargs)
                    # also save to cache
                    if autotune_pg is None or autotune_pg.rank() == 0:
                        self.save_tune_config(key, timings)

                timings.sort(key=lambda x: x[0])
                assert len(timings)
                best_config = timings[0][1]
                logger.info(f"use tune config for key {key}: {best_config}")
                self.best_configs[key] = best_config

            merged_kwargs = {**best_config, **kwargs}

        return self.fn(*args, **merged_kwargs)

    def load_tune_config(self, key):
        logger.info(f"Loading tuning config for key {key}")
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                timings = load_autotune_data(cache_path)
                logger.debug(f"Loaded timings from cache for key {key}: {timings}")
                if not _compare_deps(timings["deps"]):
                    if _check_autotune_version():
                        return None
                    warnings.warn(
                        "triton_dist is tuned from a different software environment, please check your environment.")
                return timings["config"]
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache file {cache_path}: {e}")
        return None

    def save_tune_config(self, key, timings):
        cache_path = self._get_cache_path(key)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.exists():
                try:
                    ts = load_autotune_data(cache_path)["timestamp"]
                except (json.JSONDecodeError, IOError, KeyError):
                    ts = datetime.datetime.fromtimestamp(cache_path.stat().st_mtime).strftime("%Y-%m-%d_%H-%M-%S")
                cache_path.rename(cache_path.with_suffix(f".{ts}"))
            store_autotune_data(cache_path, self.fn.__name__, key, timings)
            logger.info(f"Saved timings to cache for key {key}: {timings}")
        except IOError as e:
            logger.warning(f"Failed to save cache file {cache_path}: {e}")

    def _hash(self, key):
        return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()

    def _full_key(self, key):
        return (self.func_hash, hw_hash(), key)

    def _get_cache_path(self, key):  # Create a stable hash from the key tuple
        key = self._full_key(key)
        key_hash = self._hash(key)
        os.makedirs(self.func_dir(), exist_ok=True)
        return self.func_dir() / f"{key_hash}.json"

    def _get_tune_log_path(self, key, pg: torch.distributed.ProcessGroup | None):
        key = self._full_key(key)
        key_hash = self._hash(key)
        if pg is None:
            suffix = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            import time
            t = torch.tensor([time.time_ns()], device="cuda", dtype=torch.int64)
            pg.broadcast(t, 0)
            torch.cuda.synchronize()
            suffix = datetime.datetime.fromtimestamp(t.item() / 1e9,
                                                     tz=datetime.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            suffix = f"{suffix}-rank{pg.rank()}"
        return self.func_dir() / f"{key_hash}.log-{suffix}"

    def tune(self, config_space, autotune_pg, *args, **kwargs):

        timings = []
        for config in config_space:
            # this is not so efficient in python. in case CPU bound, do this once.
            merged_args = {**config, **kwargs}

            def bench_func():
                self.fn(*args, **merged_args)

            # Benchmarking using torch.cuda.Event for GPU, with a fallback to timeit for CPU.
            try:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                # Warmup runs
                if autotune_pg:
                    barrier_async(autotune_pg)

                for _ in range(5):
                    bench_func()

                if autotune_pg:
                    barrier_async(autotune_pg)

                # Timed runs
                start_event.record()
                repeats = 10
                for _ in range(repeats):
                    bench_func()
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event) / repeats
                logger.info(f"Timing for config {config}: {elapsed_time} ms")
                timings.append((elapsed_time, config))
            except triton.runtime.OutOfResources:
                timings.append((float("inf"), config))
            except Exception as e:
                logger.warning(f"Failed to benchmark config {config}: {e}")
                timings.append((float("inf"), config))

        if autotune_pg:
            # collect data and check the max
            barrier_async(autotune_pg)
            x = torch.tensor([x[0] for x in timings], dtype=torch.float32).cuda()
            torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.MAX)
            x = x.cpu().numpy().tolist()
            timings = [(x[i], timings[i][1]) for i in range(len(timings))]

        timings.sort(key=lambda x: x[0])  # TODO(houqi.1993) is this stable?
        logger.info(f"timings: {timings}")
        return timings


def autotune(config_space, key_fn, prune_fn=None):

    def wrapper(func):
        return AutoTuner(func, config_space, key_fn, prune_fn)

    return wrapper
