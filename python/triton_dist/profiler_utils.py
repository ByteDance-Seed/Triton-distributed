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


def perf_func(func, iters, warmup_iters):
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    for n in range(iters + warmup_iters):
        if n == warmup_iters:
            start_event.record()
        output = func()
    stop_event.record()
    start_event.wait()
    stop_event.wait()
    torch.cuda.current_stream().synchronize()
    duration_ms = start_event.elapsed_time(stop_event)
    return output, duration_ms / iters
