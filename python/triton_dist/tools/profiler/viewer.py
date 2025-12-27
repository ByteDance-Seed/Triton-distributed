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

from typing import Any, Dict, List, Tuple
import numpy as np
from .language import (
    NUM_BITS_ID,
    NUM_BITS_TASK_TYPE,
    NUM_BITS_EVENT,
)
from .context import is_empty_slot

import torch
from dataclasses import dataclass


# adapt from flashinfer/flashinfer/profiler/__init__.py
def decode_tag(tag, num_groups):
    """
    Decode a profiler tag into (block_idx, group_idx, task_type, is_start).
    Tag layout:  GLOBAL_ID | TASK TYPE | IS START
    """
    global_id = (tag >> (NUM_BITS_TASK_TYPE + NUM_BITS_EVENT)) & ((1 << NUM_BITS_ID) - 1)
    task_type = (tag >> NUM_BITS_EVENT) & ((1 << NUM_BITS_TASK_TYPE) - 1)
    is_start = tag & NUM_BITS_EVENT
    block_idx = global_id // num_groups
    group_idx = global_id % num_groups
    assert NUM_BITS_EVENT == 1
    return block_idx, group_idx, task_type, is_start


def _track_iter(profiler_buffer: np.ndarray, num_blocks, num_groups):
    empty_count = 0
    for i in range(len(profiler_buffer)):
        if is_empty_slot(profiler_buffer[i]):
            empty_count += 1
            if empty_count > num_blocks * num_groups:
                return
            continue
        empty_count = 0
        tag, timestamp = profiler_buffer[i:i + 1].view(np.uint32)
        tag = int(tag)
        timestamp = int(timestamp)
        block_idx, group_idx, task_type, is_start = decode_tag(tag, num_groups)
        yield block_idx, group_idx, task_type, is_start, timestamp


def _verify_and_reorg_tracks(profiler_buffer: np.ndarray, num_blocks, num_groups):
    """
    return List[(block_idx, group_idx, task_type, start_time, end_time)] sorted by start_time
    """
    tracks = {}
    records = []
    for block_idx, group_idx, task_type, is_start, timestamp in _track_iter(profiler_buffer, num_blocks, num_groups):
        track_key = block_idx, group_idx, task_type
        if is_start:
            assert track_key not in tracks, f"track ({track_key}) is opened again when it's not closed"
            tracks[track_key] = timestamp
        else:
            ts_start = tracks[track_key]
            assert ts_start <= timestamp
            records.append((block_idx, group_idx, task_type, ts_start, timestamp))
            tracks.pop(track_key)

    assert not tracks, "some records is not closed"
    records.sort(key=lambda x: x[3])
    return records


class Tracker:
    """ this tracker contains multiple tracks to support overlaped tracks"""

    def __init__(self, parent, track_name):
        self.parent = parent
        self.tracks = []  # (track, track_ts_end)
        self.grp = self.parent.create_group(track_name)

    def track(self, ts_start, ts_end, annotation: str):
        track = self._choose_track(ts_start, ts_end)
        track.open(ts_start, annotation)
        track.close(ts_end)

    def _choose_track(self, ts_start, ts_end):
        for index, (track, track_ts_end) in enumerate(self.tracks):
            if track_ts_end <= ts_start:  # track is idle
                self.tracks[index][1] = ts_end
                return track
        self.tracks.append([self.grp.create_track(), ts_end])
        return self.tracks[-1][0]


# adapt from flashinfer/flashinfer/profiler/__init__.py
def export_to_perfetto_trace(profiler_buffer: torch.Tensor, task_names: List[str], file_name: str,
                             verbose: bool = False) -> None:
    from tg4perfetto import TraceGenerator

    if not file_name.endswith(".perfetto-trace"):
        file_name = file_name + ".perfetto-trace"
    assert profiler_buffer.dtype == torch.uint64
    profiler_buffer_host = profiler_buffer.cpu()
    num_blocks, num_groups = profiler_buffer_host[:1].view(dtype=torch.int32)
    num_blocks = int(num_blocks)
    num_groups = int(num_groups)

    tgen = TraceGenerator(file_name)

    pid_map = {}
    track_map: Dict[Tuple[int, int, int], Any] = {}

    block_idx_to_smid = {}
    profiler_buffer_host = profiler_buffer_host[1:]
    # for better view
    if num_groups == 1:
        pid_master = tgen.create_group("tracks of all SMs")

    for i in range(num_blocks):
        block_idx, sm_id = profiler_buffer_host[i:i + 1].view(dtype=torch.uint32)
        block_idx, sm_id = int(block_idx), int(sm_id)
        block_idx_to_smid[block_idx] = sm_id
        if num_groups > 1:
            if block_idx not in pid_map:
                pid_map[block_idx] = tgen.create_group(f"block_{block_idx}_sm_{sm_id}")
        else:
            pid_map[block_idx] = pid_master
    if verbose:
        print(f"block_idx_to_smid = {block_idx_to_smid}, {len(block_idx_to_smid)}")

    profiler_buffer_host = profiler_buffer_host[num_blocks:].numpy()
    for block_idx, group_idx, task_type, ts_start, ts_end in _verify_and_reorg_tracks(
            profiler_buffer_host, num_blocks, num_groups):
        sm_id = block_idx_to_smid[block_idx]
        if verbose:
            print(
                f'block_idx = {block_idx}, group_idx: {group_idx}, task_type = {task_type}, range =[{ts_start}, {ts_end}]'
            )
        # create trackers
        pid = pid_map[block_idx]
        cur_task_name = task_names[task_type]
        track_key = (sm_id, )

        if (track := track_map.get(track_key, None)) is None:
            if num_groups > 1:
                track = Tracker(pid, str(sm_id))
            else:
                track = Tracker(pid, str(sm_id))
            track_map[track_key] = track

        track.track(ts_start, ts_end, f"{cur_task_name}:{block_idx}")

    tgen.flush()


@dataclass
class Task:
    tag: int
    task_type: int
    start_time: int  # ns
    duration: int  # ns


def parse_to_tracks(profiler_buffer: torch.Tensor):
    assert profiler_buffer.dtype == torch.uint64
    profiler_buffer_host = profiler_buffer.cpu()
    num_blocks, num_groups = profiler_buffer_host[:1].view(dtype=torch.int32)
    num_blocks = int(num_blocks)
    num_groups = int(num_groups)

    begin_timestamp_map = {}

    block_idx_to_smid = {}
    profiler_buffer_host = profiler_buffer_host[1:]
    block_idx_to_tracks = {}
    for i in range(num_blocks):
        block_idx, sm_id = profiler_buffer_host[i:i + 1].view(dtype=torch.uint32)
        block_idx, sm_id = int(block_idx), int(sm_id)
        block_idx_to_smid[block_idx] = sm_id
        block_idx_to_tracks[block_idx] = []

    profiler_buffer_host = profiler_buffer_host[num_blocks:].numpy()

    empty_count = 0
    for i in range(len(profiler_buffer_host)):
        if is_empty_slot(profiler_buffer_host[i]):
            empty_count += 1
            if empty_count > num_blocks * num_groups:
                break
            continue
        empty_count = 0
        tag, timestamp = profiler_buffer_host[i:i + 1].view(np.uint32)
        tag = int(tag)
        timestamp = int(timestamp)
        block_idx, group_idx, task_type, is_start = decode_tag(tag, num_groups)
        sm_id = block_idx_to_smid[block_idx]

        if is_start:
            begin_timestamp_map[(block_idx, group_idx, task_type)] = timestamp
        else:
            begin_timestamp = begin_timestamp_map[(block_idx, group_idx, task_type)]
            assert begin_timestamp < timestamp, f"timestamp overflow, start = {begin_timestamp}, end = {timestamp}"
            track = Task(tag=tag, task_type=task_type, start_time=begin_timestamp, duration=timestamp - begin_timestamp)
            block_idx_to_tracks[block_idx].append(track)
    return block_idx_to_tracks
