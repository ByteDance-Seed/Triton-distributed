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

from contextlib import nullcontext
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional
import torch
import gc

import logging
import shutil
import tempfile
from pathlib import Path
import json
import string
import re
import gzip


def load_json(json_file):
    with open(json_file, "r", encoding="utf-8", errors="replace") as file:
        content = file.read()

        # torch 2.4+ profile with with_stack makes some invalid argument, which makes chrome/edge unhappy
        # use work around here: https://github.com/pytorch/pytorch/issues/121219
        # Decode Unicode escape sequences
        content = content.encode().decode("unicode_escape")

        # Regex to find "name": "<value>"
        def replace_non_ascii_and_quotes(match):
            name = match.group(1)
            visible_printable = "".join(c for c in string.printable if c not in "\t\n\r\x0b\x0c}{")
            cleaned_name = "".join(c if c in visible_printable else "x" for c in name)
            cleaned_name = cleaned_name.replace('"', "y")  # Replace internal quotes
            return f'"name": "{cleaned_name}"'

        # Apply regex to clean names
        cleaned_content = re.sub(
            r'"name": "([\s\S]*?)"(?=, |\}|\s*\})',
            replace_non_ascii_and_quotes,
            content,
            flags=re.DOTALL,
        )

    return json.loads(cleaned_content, strict=False)


def process_trace_json(json_file):
    RANK_MAX_PID = 100000000

    def _mapping(x, delta):
        if isinstance(x, str):
            return f"{x}_{delta}"
        return x + delta

    def _process_item(item, rank, delta):
        # remapping tid and pid
        item["pid"] = _mapping(item["pid"], delta)
        item["tid"] = _mapping(item["tid"], delta)
        # rename metadata name
        if item["ph"] == "M":
            if item["name"] in ["process_name", "thread_name"]:
                name = item["args"]["name"]
                item["args"]["name"] = f"{name}_rank{rank}"
            elif item["name"] == "process_labels":
                labels = item["args"]["labels"]
                item["args"]["labels"] = f"{labels}_{rank}"

    logging.info(f"process {json_file}")
    trace = load_json(json_file)
    events = trace["traceEvents"]
    rank = trace["distributedInfo"]["rank"]
    delta = rank * RANK_MAX_PID
    [_process_item(x, rank, delta) for x in events]
    return trace


def _merge_json_v1(to_merge_files: List[Path], output_json: Path, compress: bool = True):
    events = []
    for json_file in to_merge_files:
        logging.info(f"process {json_file}")
        trace = process_trace_json(json_file)
        events.extend(trace["traceEvents"])

    logging.info("compress...")
    trace["traceEvents"] = events
    if compress:
        with gzip.open(str(output_json) + ".tar.gz", mode="wt", compresslevel=3) as g:
            json.dump(trace, g)
    else:
        with open(output_json, "w") as f:
            json.dump(trace, f)

    logging.info("done.")


class ParallelJsonDumper:

    def __init__(self, parallel_field: str, chunk_size: int = 5000):
        self.chunk_size = chunk_size
        self.cpu_count = cpu_count()
        self.parallel_field = parallel_field

    def dump(self, data: Dict[str, Any], output_path: Path) -> None:
        """Dump JSON with parallel processing of large parallel_field field"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pvalue = data.pop(self.parallel_field)

        # Split the large list into manageable chunks
        chunks = self._chunkify_list(pvalue)

        # Create processing pool
        with Pool(processes=min(len(chunks), self.cpu_count)) as pool:
            # Process chunks in parallel but maintain order
            chunk_strings = pool.map(self._process_chunk, chunks)

            # Stream results to disk
            self._write_output(data, chunk_strings, output_path)

    def _chunkify_list(self, pvalue: List[Any]) -> List[List[Any]]:
        """Split list into chunks for parallel processing"""
        return [pvalue[i:i + self.chunk_size] for i in range(0, len(pvalue), self.chunk_size)]

    def _process_chunk(self, chunk: List[Any]) -> str:
        """Convert chunk to JSON and strip enclosing brackets"""
        chunk_json = json.dumps(chunk, separators=(",", ":"))
        return chunk_json[1:-1]  # Remove [ and ]

    def _write_output(self, base_data: Dict[str, Any], chunk_strings: List[str], output_path: Path) -> None:
        """Write JSON to disk with proper structure"""
        with open(output_path, "w") as f:
            # Write base data
            f.write(json.dumps(base_data, separators=(",", ":"))[:-1])

            # Append pvalue header
            f.write(f',"{self.parallel_field}":[')

            # Write chunks with proper commas
            for i, chunk_str in enumerate(chunk_strings):
                if i > 0:
                    f.write(",")
                f.write(chunk_str)

            # Close JSON structure
            f.write("]}")


def _merge_json_v2(
    to_merge_files: List[Path],
    output_json: Path,
    compress: bool = True,
):
    events = []
    with Pool(processes=min(len(to_merge_files), cpu_count())) as pool:
        for trace in pool.map(process_trace_json, to_merge_files):
            events.extend(trace["traceEvents"])

    trace["traceEvents"] = events
    logging.info("dump json")
    ParallelJsonDumper("traceEvents", 100000).dump(trace, Path(output_json))

    if compress:
        with gzip.open(output_json.with_suffix(".tar.gz"), mode="wb", compresslevel=3) as g, open(output_json,
                                                                                                  "rb") as f:
            logging.info("compress...")
            g.write(f.read())
        output_json.unlink()
    logging.info("done.")


def _merge_json(
    to_merge_files: List[Path],
    output_json: Path,
    compress: bool = True,
    version: int = 2,
):
    if version == 1:
        _merge_json_v1(to_merge_files, output_json, compress)
    elif version == 2:
        _merge_json_v2(to_merge_files, output_json, compress)


class group_profile:

    def __init__(
        self,
        name: str,
        do_prof: bool = True,
        merge_group: bool = True,
        keep_merged_only: bool = True,
        compress: bool = True,
        group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        self.name = name
        self.do_prof = do_prof
        self.profile = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        )
        self.group: torch.distributed.ProcessGroup = group or torch.distributed.group.WORLD
        self.merge_group = merge_group
        self.keep_merged_only = keep_merged_only
        self.compress = compress
        self.trace_file = (Path("prof") / f"{self.name}" / f"rank{self.group.rank()}.json")

    def __enter__(self):
        if self.do_prof:
            self.profile.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.do_prof:
            self.profile.__exit__(exc_type, exc_val, exc_tb)
            # export chrome trace
            self.trace_file.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"export chrome trace to {self.trace_file}")
            self.profile.export_chrome_trace(str(self.trace_file))
            if self.merge_group:
                self.merge_all()

    def _collect_all_to_rank0(self):
        # merge all
        if self.merge_group:
            torch.cuda.synchronize()  # wait for all ranks export
            with open(self.trace_file, "rb") as f:
                trace_content = f.read()
            trace_content_list = [None for _ in range(self.group.size())]
            torch.distributed.gather_object(
                trace_content,
                trace_content_list if self.group.rank() == 0 else None,
                dst=0,
                group=self.group,
            )
            torch.cuda.synchronize()  # wait for all ranks export
            return trace_content_list if self.group.rank() == 0 else None

    def _merge_all_trace(self, trace_content_list):
        logging.info("merge profiles...")

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir).mkdir(exist_ok=True)

            for n in range(self.group.size()):
                with open(Path(tmpdir) / f"trace_{n}.json", "wb") as f:
                    f.write(trace_content_list[n])

            # merge all json
            to_merge_files = [Path(tmpdir) / f"trace_{n}.json" for n in range(self.group.size())]
            merged_json = Path("prof") / f"{self.name}_merged.json"
            _merge_json(to_merge_files, merged_json, self.compress)

    def merge_all(self):
        trace_content_list = self._collect_all_to_rank0()
        if self.group.rank() == 0:
            self._merge_all_trace(trace_content_list)
        self.group.barrier()
        torch.cuda.synchronize()
        outdir = Path("prof") / f"{self.name}"
        if self.keep_merged_only:
            logging.info(f"remove profile directory: {outdir}")
            self.trace_file.unlink(missing_ok=True)
            if torch.cuda.current_device() == 0:  # run once for a device
                shutil.rmtree(self.trace_file.parent, ignore_errors=True)


def get_torch_prof_ctx(do_prof: bool):
    ctx = (torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
    ) if do_prof else nullcontext())
    return ctx


def perf_func_with_l2_reset(func, iters, warmup_iters):
    # total 256MB is enough to clear L2 cache
    cache = torch.zeros((64 * 1024 * 1024, ), dtype=torch.int, device="cuda")
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stop_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(warmup_iters):
        output = func()
        cache.zero_()
    for start_event, stop_event in zip(start_events, stop_events):
        start_event.record()
        output = func()
        stop_event.record()
        cache.zero_()

    torch.cuda.current_stream().synchronize()
    duration_ms = sum(
        [start_event.elapsed_time(stop_event) for start_event, stop_event in zip(start_events, stop_events)])
    return output, duration_ms / iters


def sleep_async(duration_ms: int):
    clock_rate_hz = torch.cuda.clock_rate() * 1e6
    torch.cuda._sleep(int(clock_rate_hz * duration_ms / 1000))


def perf_func(func, iters, warmup_iters):
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    # Warmup
    for _ in range(warmup_iters):
        _ = func()
    torch.cuda.synchronize()
    # Benchmark
    start_event.record()
    for _ in range(iters):
        output = func()
    stop_event.record()
    torch.cuda.synchronize()
    duration_ms = start_event.elapsed_time(stop_event)
    return output, duration_ms / iters


def benchmark_latency_memory(func, iters=100, warmup_iters=10, pre_func=None):
    for _ in range(warmup_iters):
        if pre_func:
            pre_func()
        func()
    _, time_ms = perf_func(func, iters, warmup_iters)

    if pre_func:
        pre_func()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_mem = torch.cuda.memory_allocated()
    y = func()

    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated()
    peak_mem_mb = (peak_mem - start_mem) / (1024 * 1024)

    del y
    gc.collect()
    torch.cuda.empty_cache()

    return time_ms, peak_mem_mb


def print_benchmark_comparison(all_implementations, test_name="", param_names=None, title_params=None):
    """
    Print complete benchmark comparison for all configurations.
    
    Args:
        all_implementations: Dict mapping config keys to implementations data.
                           Format: {
                               config_key: {
                                   'impl_name_pass': {'latency': float, 'memory': float, 'precision': bool/str},
                                   ...
                               }
                           }
                           Where impl_name_pass can be any name like 'torch_fwd', 'torch_fwd_bwd', etc.
        test_name: Name of the test/operation
        param_names: List of parameter names for config tuple elements. 
                    If None, uses default names like 'param_0', 'param_1', etc.
                    Example: ['M', 'Dim', 'Vocab', 'SM'] for config (M, Dim, Vocab, SM)
        title_params: Dict mapping parameter names to their values for title.
                     Example: {'SM_margin': 0} will add "(SM_margin=0)" to title
                     These parameters will be excluded from table columns
    """
    if not all_implementations:
        return

    # ANSI color codes - fixed colors for latency and memory
    LATENCY_COLOR = '\033[92m'  # Green for latency
    MEMORY_COLOR = '\033[94m'  # Blue for memory
    RESET = '\033[0m'

    def colorize_latency(time_val):
        """Color latency - always green"""
        return f"{LATENCY_COLOR}{time_val:.3f}{RESET}"

    def colorize_memory(memory_val):
        """Color memory - always blue"""
        return f"{MEMORY_COLOR}{memory_val:.2f}{RESET}"

    all_results = {}
    for config_key, implementations in all_implementations.items():
        results = []
        for impl_name, data in implementations.items():
            latency_val = data.get('latency', data.get('time', 0.0))
            memory_val = data.get('memory', 0.0)
            precision_val = data.get('precision', 'unknown')
            if isinstance(precision_val, bool):
                precision_val = "✅" if precision_val else "❌"
            elif precision_val is None:
                precision_val = "N/A"
            colored_latency = colorize_latency(latency_val)
            colored_memory = colorize_memory(memory_val)

            results.append([impl_name, colored_latency, colored_memory, precision_val])

        all_results[config_key] = {}
        for result in results:
            impl_name, latency_str, memory_str, precision = result
            latency_val = float(
                latency_str.replace('\033[92m', '').replace('\033[93m', '').replace('\033[1m',
                                                                                    '').replace('\033[0m', ''))
            memory_val = float(memory_str.replace('\033[94m', '').replace('\033[1m', '').replace('\033[0m', ''))

            all_results[config_key][impl_name] = {'latency': latency_val, 'memory': memory_val, 'precision': precision}

    # Build title with optional parameters
    title = f"{test_name} Benchmark Summary"
    if title_params:
        param_str = ", ".join([f"{k}={v}" for k, v in title_params.items()])
        title += f" ({param_str})"
    title += " (format: latency(ms)/peak_memory(MB)/precision)"

    print("\n" + "=" * min(130, len(title) + 10))
    print(title)
    print("=" * min(130, len(title) + 10))

    summary_data = []
    for config, config_results in all_results.items():
        if isinstance(config, tuple):
            # Filter out title_params from config tuple
            if title_params and param_names:
                filtered_config = []
                for i, value in enumerate(config):
                    if i < len(param_names) and param_names[i] not in title_params:
                        filtered_config.append(value)
                row = filtered_config
            else:
                row = list(config)
        else:
            row = [config]

        # Smart ordering: group by pass type (fwd first, then fwd_bwd), then by implementation
        fwd_impls = []
        bwd_impls = []
        other_impls = []

        for impl_name in config_results.keys():
            if impl_name.endswith('_fwd'):
                fwd_impls.append(impl_name)
            elif impl_name.endswith('_fwd_bwd'):
                bwd_impls.append(impl_name)
            else:
                other_impls.append(impl_name)

        # Sort within each group to maintain consistent order
        fwd_impls.sort()
        bwd_impls.sort()
        other_impls.sort()

        # Combine in desired order: fwd implementations first, then bwd, then others
        impl_names = fwd_impls + bwd_impls + other_impls

        for impl_name in impl_names:
            if impl_name in config_results:
                latency = config_results[impl_name]['latency']
                memory = config_results[impl_name]['memory']
                precision = config_results[impl_name]['precision']

                colored_latency = f"{LATENCY_COLOR}{latency:.3f}{RESET}"
                colored_memory = f"{MEMORY_COLOR}{memory:.2f}{RESET}"

                cell_value = f"{colored_latency}/{colored_memory}/{precision}"

                row.append(cell_value)
            else:
                row.append("N/A")

        summary_data.append(row)

    # Generate column names
    if summary_data:
        first_config = next(iter(all_results.keys()))
        if isinstance(first_config, tuple):
            config_cols = []
            for i in range(len(first_config)):
                if param_names and i < len(param_names):
                    # Skip parameters that are in title_params
                    if title_params and param_names[i] in title_params:
                        continue
                    config_cols.append(param_names[i])
                else:
                    # Skip parameters that are in title_params (by index)
                    if title_params and param_names:
                        # Find the parameter name for this index
                        if i < len(param_names) and param_names[i] in title_params:
                            continue
                    config_cols.append(f"param_{i}")
        else:
            config_cols = ["config"]

        # Get implementation names in smart order from first config
        first_config_results = next(iter(all_results.values()))

        # Smart ordering: group by pass type (fwd first, then fwd_bwd), then by implementation
        fwd_impls = []
        bwd_impls = []
        other_impls = []

        for impl_name in first_config_results.keys():
            if impl_name.endswith('_fwd'):
                fwd_impls.append(impl_name)
            elif impl_name.endswith('_fwd_bwd'):
                bwd_impls.append(impl_name)
            else:
                other_impls.append(impl_name)

        # Sort within each group to maintain consistent order
        fwd_impls.sort()
        bwd_impls.sort()
        other_impls.sort()

        # Combine in desired order: fwd implementations first, then bwd, then others
        impl_cols = fwd_impls + bwd_impls + other_impls

        if config_cols:
            summary_data.sort(key=lambda x: x[0] if isinstance(x[0], (int, float)) else 0)

        all_cols = config_cols + impl_cols
        col_widths = []

        for col in all_cols:
            col_widths.append(len(col))

        for row in summary_data:
            for i, cell in enumerate(row):
                # Strip ANSI color codes for width calculation
                clean_cell = str(cell)
                if '\033[' in clean_cell:
                    import re
                    clean_cell = re.sub(r'\033\[[0-9;]*m', '', clean_cell)
                col_widths[i] = max(col_widths[i], len(clean_cell))

        # Add minimum padding
        col_widths = [max(w, 8) for w in col_widths]  # minimum 8 characters

        # Print with flexible widths
        if summary_data:
            header_parts = []
            for i, col in enumerate(all_cols):
                header_parts.append(f"{col:>{col_widths[i]}}")
            print(" ".join(header_parts))

            # Print separator
            separator_length = sum(col_widths) + len(all_cols) - 1  # sum of widths + spaces
            print("=" * separator_length)

            # Print data rows
            for row in summary_data:
                row_parts = []
                for i, cell in enumerate(row):
                    if i < len(config_cols):
                        # Config columns - right align
                        if isinstance(cell, (int, float)):
                            row_parts.append(f"{cell:>{col_widths[i]}}")
                        else:
                            row_parts.append(f"{cell:>{col_widths[i]}}")
                    else:
                        # Data columns
                        cell_str = str(cell)
                        if '\033[' in cell_str:
                            # For colored cells, we need to pad without breaking colors
                            import re
                            clean_cell = re.sub(r'\033\[[0-9;]*m', '', cell_str)
                            padding = col_widths[i] - len(clean_cell)
                            if padding > 0:
                                # Add padding before the colored content
                                row_parts.append(" " * padding + cell_str)
                            else:
                                row_parts.append(cell_str)
                        else:
                            row_parts.append(f"{cell_str:>{col_widths[i]}}")
                print(" ".join(row_parts))
