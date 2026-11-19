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
"""Shared benchmark reporting helpers for SM100 communication-GEMM kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import torch

_SM100_PEAKS = {
    "GB200": (2500.0, 900.0),
    "B200": (2250.0, 900.0),
}

Metric = Literal["tflops", "busbw"]
SolKind = Literal["compute", "bus", "sum", "max"]


@dataclass(frozen=True)
class ReportRow:
    label: str
    key: str
    metric: Metric
    sol: SolKind | None = None
    note: str = ""


@dataclass(frozen=True)
class SpeedupRow:
    label: str
    numerator: str
    denominator: str
    note: str


def print_benchmark_report(
    *,
    title: str,
    times: Mapping[str, float],
    rows: list[ReportRow],
    flops: float,
    comm_bytes: float,
    device: torch.device | int,
    comm_label: str = "busbw",
) -> None:
    device_name = torch.cuda.get_device_name(device)
    peaks = next((value for name, value in _SM100_PEAKS.items() if name in device_name), None)
    if peaks is None:
        raise ValueError(f"missing peak specification for {device_name}")
    peak_tflops, peak_bus_bw_gbps = peaks
    ideal_gemm_ms = flops / (peak_tflops * 1e9)
    ideal_comm_ms = comm_bytes / (peak_bus_bw_gbps * 1e6)
    sol_numerator = {
        "sum": ideal_gemm_ms + ideal_comm_ms,
        "max": max(ideal_gemm_ms, ideal_comm_ms),
    }
    active_rows = [row for row in rows if row.key in times]
    label_width = max(len(row.label) for row in active_rows) + 1

    print(f"\n=== {title} ===")
    print(f"upper_bound: {device_name}, {peak_tflops:.1f} TFLOP/s BF16, {peak_bus_bw_gbps:.1f} GB/s NVLink busbw")
    for row in active_rows:
        ms = times[row.key]
        if row.metric == "busbw":
            value = comm_bytes / (ms * 1e6)
            body = f"{ms:.3f} ms, {value:.1f} GB/s {comm_label}"
        else:
            value = flops / (ms * 1e9)
            body = f"{ms:.3f} ms, {value:.1f} TFLOP/s"

        if row.sol == "bus":
            sol = 100.0 * value / peak_bus_bw_gbps
        elif row.sol == "compute":
            sol = 100.0 * value / peak_tflops
        elif row.sol in sol_numerator:
            sol = 100.0 * sol_numerator[row.sol] / ms
        else:
            sol = None
        if sol is not None:
            body += f", {sol:.1f}% SOL"
        if row.note:
            body += f" {row.note}"
        print(f"{row.label:<{label_width}}{body}")


def print_speedups(times: Mapping[str, float], rows: list[SpeedupRow]) -> None:
    active_rows = [row for row in rows if row.numerator in times and row.denominator in times]
    if not active_rows:
        return
    label_width = max(len(row.label) for row in active_rows) + 1
    for row in active_rows:
        ratio = times[row.numerator] / times[row.denominator]
        print(f"{row.label:<{label_width}}{ratio:.2f}x {row.note}")
