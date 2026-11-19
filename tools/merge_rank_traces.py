################################################################################
#
# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
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
"""Merge per-rank PyTorch Chrome traces into one compressed trace."""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Iterable
from pathlib import Path

_RANK_ID_STRIDE = 100_000_000
_RANK_LOCAL_ARG_IDS = ("External id", "correlation")


def _load_trace(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        trace = json.load(f)
    if not isinstance(trace, dict) or not isinstance(trace.get("traceEvents"), list):
        raise ValueError(f"{path} is not a PyTorch Chrome trace")
    return trace


def _write_trace(path: Path, trace: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        json.dump(trace, f, separators=(",", ":"))


def _offset_id(value, delta: int):
    if isinstance(value, str):
        return f"{value}_{delta}"
    return value + delta


def _merge_events(trace: dict, rank: int) -> list[dict]:
    delta = rank * _RANK_ID_STRIDE
    events: list[dict] = []

    for event in trace["traceEvents"]:
        event = dict(event)
        if "pid" in event:
            event["pid"] = _offset_id(event["pid"], delta)
        if "tid" in event:
            event["tid"] = _offset_id(event["tid"], delta)
        if "id" in event:
            event["id"] = _offset_id(event["id"], delta)
        if "bind_id" in event:
            event["bind_id"] = _offset_id(event["bind_id"], delta)

        args = dict(event.get("args") or {})
        for key in _RANK_LOCAL_ARG_IDS:
            if key in args:
                args[key] = _offset_id(args[key], delta)

        if event.get("ph") == "M" and event.get("name") in ("process_name", "thread_name"):
            args["name"] = f"{args.get('name', '')}_rank{rank}"
        elif event.get("ph") == "M" and event.get("name") == "process_labels":
            args["labels"] = f"{args.get('labels', '')}_{rank}"
        if args or "args" in event:
            event["args"] = args
        events.append(event)
    return events


def _nest_gpu_user_annotations(events: list[dict]) -> None:
    gpu_annotations = [
        event for event in events if event.get("ph") == "X" and event.get("cat") == "gpu_user_annotation"
    ]
    for cpu_parent in events:
        if cpu_parent.get("ph") != "X" or cpu_parent.get("cat") != "user_annotation":
            continue
        external_id = (cpu_parent.get("args") or {}).get("External id")
        if external_id is None:
            continue

        cpu_start = cpu_parent["ts"]
        cpu_end = cpu_start + cpu_parent["dur"]
        by_track: dict[tuple[object, object], list[dict]] = {}
        for child in gpu_annotations:
            child_start = child["ts"]
            child_end = child_start + child["dur"]
            if cpu_start <= child_start and child_end <= cpu_end:
                by_track.setdefault((child.get("pid"), child.get("tid")), []).append(child)

        for children in by_track.values():
            parents = [
                child for child in children if child.get("name") == cpu_parent.get("name") and (
                    child.get("args") or {}).get("External id") == external_id
            ]
            if not parents:
                continue
            parent = min(parents, key=lambda event: event["ts"])
            nested_start = min(child["ts"] for child in children)
            nested_end = max(child["ts"] + child["dur"] for child in children)
            parent["ts"] = nested_start
            parent["dur"] = nested_end - nested_start


def merge_rank_traces(trace_paths: Iterable[str | Path], output_path: str | Path) -> Path:
    paths = [Path(path) for path in trace_paths]
    if not paths:
        raise ValueError("at least one rank trace is required")

    traces = [_load_trace(path) for path in paths]
    events = []
    for fallback_rank, trace in enumerate(traces):
        rank = trace.get("distributedInfo", {}).get("rank", fallback_rank)
        rank_events = _merge_events(trace, rank)
        _nest_gpu_user_annotations(rank_events)
        events.extend(rank_events)

    merged = {
        key: value
        for key, value in traces[0].items()
        if key not in ("traceEvents", "distributedInfo", "deviceProperties")
    }
    merged["traceEvents"] = events
    merged["deviceProperties"] = traces[0].get("deviceProperties", [])
    merged["distributedInfo"] = {
        "world_size": len(paths),
        "rank_trace_files": [path.name for path in paths],
    }
    merged["rankDeviceProperties"] = [trace.get("deviceProperties", []) for trace in traces]

    output = Path(output_path)
    _write_trace(output, merged)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", help="Per-rank .json or .json.gz traces")
    parser.add_argument("-o", "--output", required=True, help="Merged .json.gz output")
    args = parser.parse_args()
    print(merge_rank_traces(args.traces, args.output))


if __name__ == "__main__":
    main()
