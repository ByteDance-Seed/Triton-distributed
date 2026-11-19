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
"""Validate logical data channels and lower homogeneous MPMC runtime pipelines."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from little_kernel.core.memory_effects import Access, MemoryEffectKind, R_SMEM_TCGEN05_COMMIT, R_SMEM_TMA_BULK, R_TMEM_TCGEN05_COMMIT, W_SMEM_TMA_TX, W_TMEM_TCGEN05_COMMIT
from little_kernel.backends.nv.sm100.language.dataflow.declarations import Flow
from little_kernel.backends.nv.sm100.core.pipeline import ArriveKind, BarrierSide, CountBasis, PIPELINE_BY_EFFECT_KINDS, PIPELINE_TMA_GENERIC, PIPELINE_TMA_STORE_CROSS_ROLE, PIPELINE_TMA_STORE_SAME_ROLE, PIPELINE_TMA_UMMA, PipelineSpec


def reads(access: Access) -> bool:
    return access in {Access.R, Access.RW}


def writes(access: Access) -> bool:
    return access in {Access.W, Access.RW}


def dtype_size(dtype) -> int:
    width = getattr(dtype, "width", None)
    if width is not None:
        return int(width)
    name = str(dtype).split(".")[-1]
    if name in {"bfloat16", "float16", "int16", "uint16"}:
        return 2
    if name in {"float32", "int32", "uint32"}:
        return 4
    if name in {"float64", "int64", "uint64"}:
        return 8
    if name in {"int8", "uint8"}:
        return 1
    raise ValueError(f"unsupported dtype size: {dtype}")


def _tma_payload_bytes(memory, elements: int) -> int:
    bits = memory.tma_transaction_bits
    if bits is None:
        return elements * dtype_size(memory.dtype)
    total_bits = elements * bits
    if total_bits % 8:
        raise ValueError(f"{memory.name}: TMA payload is not byte-aligned")
    return total_bits // 8


def numel(shape) -> int:
    out = 1
    for dim in shape if isinstance(shape, (tuple, list)) else [shape]:
        out *= int(dim)
    return out


def channel_roles(channel: dict, side: str) -> tuple[str, ...]:
    plural, singular = (("producers", "src") if side == "producer" else ("consumers", "dst"))
    roles = channel.get(plural)
    if roles is None and singular in channel:
        roles = (channel[singular], )
    return tuple(roles or ())


def producer_roles(item: dict) -> tuple[str, ...]:
    return tuple(item.get("producers") or (item["name"].split("->", 1)[0], ))


def consumer_roles(item: dict) -> tuple[str, ...]:
    return tuple(item.get("consumers") or (item["name"].split("->", 1)[1], ))


def channel_name(producers: tuple[str, ...], consumers: tuple[str, ...]) -> str:
    return f"{'+'.join(producers)}->{'+'.join(consumers)}"


# TX producer_count. Missing keys (cta_group=1) default to 1.
# (cta_group, multicast, plain_cg2_tma) -> count
_TX_PRODUCER_COUNT = {
    (2, False, False): 2,
    (2, False, True): 2,
    (2, True, False): 1,
    (2, True, True): 2,
}


@dataclass(frozen=True)
class PipelineParticipation:
    cta_group_size: int
    producer_count: int | None = None
    consumer_count: int | None = None
    expected_tx_bytes: int | None = None
    consumer_masks: tuple[int, ...] | None = None


def _roles_warps(graph: dict, roles: tuple[str, ...]) -> int:
    return sum(len(graph["roles"][role].warps) for role in roles)


def _roles_threads(graph: dict, roles: tuple[str, ...]) -> int:
    return 32 * _roles_warps(graph, roles)


def _pipeline_cta_group_size(graph: dict, pipeline: dict) -> int:
    roles = set(producer_roles(pipeline)) | set(consumer_roles(pipeline))
    via = set(pipeline["via"])
    groups = {event.cta_group for event in graph.get("mem_events", ()) if event.role in roles and event.mem in via}
    if not groups:
        return 1
    if not groups <= {1, 2}:
        raise ValueError(f"{pipeline['name']}: channel {sorted(via)} uses unsupported CTA "
                         f"groups {sorted(groups)}")
    return 2 if 2 in groups else 1


def _pipeline_uses_cluster_multicast(graph: dict, pipeline: dict) -> bool:
    producers = set(producer_roles(pipeline))
    via = set(pipeline["via"])
    return any(event.role in producers and event.mem in via and event.cluster_multicast
               for event in graph.get("mem_events", ()))


def _pipeline_has_plain_cg2_tma(graph: dict, pipeline: dict) -> bool:
    producers = set(producer_roles(pipeline))
    via = set(pipeline["via"])
    return any(event.role in producers and event.mem in via and event.effect_kind == W_SMEM_TMA_TX
               and event.cta_group == 2 and not event.cluster_multicast for event in graph.get("mem_events", ()))


def _planned_consumer_components(
    graph: dict,
    pipeline: dict,
    cta_group_size: int,
) -> tuple[int, tuple[int, ...]] | None:
    producers = set(producer_roles(pipeline))
    via = set(pipeline["via"])
    plans = {
        event.transfer_plan
        for event in graph.get("mem_events", ())
        if event.role in producers and event.mem in via and event.transfer_plan is not None and event.cluster_multicast
    }
    if not plans:
        return None

    cluster_shape = graph["cluster_shape"]
    group_shape = graph["cta_group_shape"]
    group_grid = tuple(c // g for c, g in zip(cluster_shape, group_shape))
    num_groups = numel(group_grid)
    adjacency = [set([group]) for group in range(num_groups)]
    for plan in plans:
        topology = graph["transfer_topologies"][plan]
        for group, peers in zip(topology["group_indices"], topology["group_peers_by_rank"], strict=True):
            peer_set = set(peers)
            adjacency[group].update(peer_set)
            for peer in peer_set:
                adjacency[peer].update(peer_set)

    def group_coord(group: int) -> tuple[int, int, int]:
        gx = group % group_grid[0]
        gy = (group // group_grid[0]) % group_grid[1]
        gz = group // (group_grid[0] * group_grid[1])
        return gx, gy, gz

    components: list[set[int]] = []
    unseen = set(range(num_groups))
    while unseen:
        root = unseen.pop()
        component = {root}
        stack = [root]
        while stack:
            component.update(adjacency[stack.pop()])
            pending = component & unseen
            unseen -= pending
            stack.extend(pending)
        components.append(component)

    sizes = {len(component) for component in components}
    if len(sizes) != 1:
        raise ValueError(f"{pipeline['name']}: asymmetric multicast completion components "
                         f"{sorted(sizes)} are unsupported")

    masks = [0] * num_groups
    for component in components:
        cta_mask = 0
        for group in component:
            gx, gy, gz = group_coord(group)
            for lane in range(cta_group_size):
                x = gx * group_shape[0] + lane
                rank = x + cluster_shape[0] * (gy * group_shape[1] + cluster_shape[1] * gz * group_shape[2])
                cta_mask |= 1 << rank
        for group in component:
            masks[group] = cta_mask
    return next(iter(sizes)), tuple(masks)


def _planned_incoming_tx_bytes(
    graph: dict,
    pipeline: dict,
    cta_group_size: int,
) -> int:
    """Derive the equal per-group TMA payload from the transfer graph."""
    cluster_shape = graph.get("cluster_shape") or (1, 1, 1)
    group_shape = graph.get("cta_group_shape")
    if group_shape is None:
        raise ValueError(f"{pipeline['name']}: missing compiler-validated cta_group_shape")
    group_grid = tuple(c // g for c, g in zip(cluster_shape, group_shape))
    num_groups = numel(group_grid)
    incoming = [0] * num_groups
    producers = set(producer_roles(pipeline))

    for mem_name in pipeline.get("producer_via", pipeline["via"]):
        events = [
            event for event in graph.get("mem_events", ()) if event.role in producers and event.mem == mem_name
            and event.effect_kind == W_SMEM_TMA_TX and writes(event.access)
        ]
        plans = {event.transfer_plan for event in events}
        if len(plans) > 1:
            raise ValueError(f"{pipeline['name']}: {mem_name} uses multiple TMA transfer "
                             "plans in one transaction")
        plan = next(iter(plans), None)
        memory = graph["memories"][mem_name]
        group_bytes = _tma_payload_bytes(memory, numel(memory.shape)) * cta_group_size
        if plan is None or not graph["transfer_topologies"][plan]["multicast"]:
            incoming = [value + group_bytes for value in incoming]
            continue

        topology = graph["transfer_topologies"][plan]
        seen_issuers: set[int] = set()
        for rank, (group, issue, peers) in enumerate(
                zip(topology["group_indices"], topology["issue_pred_by_rank"], topology["group_peers_by_rank"],
                    strict=True)):
            del rank
            if not issue or group in seen_issuers:
                continue
            seen_issuers.add(group)
            for peer in peers:
                incoming[peer] += group_bytes

    payloads = set(incoming)
    if len(payloads) != 1 or not payloads or next(iter(payloads)) <= 0:
        raise ValueError(f"{pipeline['name']}: asymmetric incoming TMA bytes per CTA group are unsupported: {incoming}")
    return next(iter(payloads))


def _barrier_count(graph: dict, pipeline: dict, side: BarrierSide, roles: tuple[str, ...], cta_group_size: int) -> int:
    if side.count == CountBasis.THREADS:
        count = _roles_threads(graph, roles)
    elif side.count == CountBasis.WARPS:
        count = _roles_warps(graph, roles)
    elif side.count == CountBasis.COMPLETION_DOMAINS:
        count = pipeline.get("completion_domains")
        if count is None:
            raise ValueError(f"{pipeline['name']}: {side.arrive.value} requires statically "
                             "inferred TMA-store completion domains")
    else:
        raise ValueError(f"{pipeline['name']}: unsupported barrier count basis "
                         f"{side.count.value}")
    return int(count) * (cta_group_size if side.cluster_scale else 1)


def _tx_producer_count(cta_group_size: int, multicast: bool, plain_cg2_tma: bool) -> int:
    return _TX_PRODUCER_COUNT.get((cta_group_size, multicast, plain_cg2_tma), 1)


def barrier_participation(graph: dict, pipeline: dict) -> PipelineParticipation:
    """Resolve barrier counts from the declared role topology."""
    producers = producer_roles(pipeline)
    consumers = consumer_roles(pipeline)
    template: PipelineSpec = pipeline["template"]
    producer_warps = _roles_warps(graph, producers)
    consumer_warps = _roles_warps(graph, consumers)
    pipe_name = pipeline["name"]
    if not producers or producer_warps == 0:
        raise ValueError(f"{pipe_name}: producer role domain is empty")
    if not consumers or consumer_warps == 0:
        raise ValueError(f"{pipe_name}: consumer role domain is empty")

    cta_group_size = _pipeline_cta_group_size(graph, pipeline)
    if template.named_barrier:
        return PipelineParticipation(cta_group_size)
    if not template.has_barriers or template.producer is None:
        raise ValueError(f"{pipe_name}: incomplete barrier protocol for {template.name}")

    cluster_multicast = _pipeline_uses_cluster_multicast(graph, pipeline)
    if template.producer.arrive == ArriveKind.MBAR_TX:
        producer_count = _tx_producer_count(cta_group_size, cluster_multicast,
                                            _pipeline_has_plain_cg2_tma(graph, pipeline))
    else:
        producer_count = _barrier_count(graph, pipeline, template.producer, producers, cta_group_size)
    consumer_count = (_barrier_count(graph, pipeline, template.consumer, consumers, cta_group_size)
                      if template.consumer is not None else None)
    expected_tx_bytes = (_planned_incoming_tx_bytes(graph, pipeline, cta_group_size)
                         if template.producer.arrive == ArriveKind.MBAR_TX else None)
    consumer_masks = None
    if cluster_multicast:
        if template.consumer is None:
            raise ValueError(f"{pipe_name}: {template.name} does not support cluster multicast")
        cluster_shape = graph["cluster_shape"]
        cluster_ctas = cluster_shape[0] * cluster_shape[1] * cluster_shape[2]
        if (template not in {PIPELINE_TMA_UMMA, PIPELINE_TMA_GENERIC} or cluster_ctas % cta_group_size
                or cluster_ctas > 16):
            raise ValueError(f"{pipe_name}: planned multicast requires a "
                             "TMA-to-UMMA or TMA-to-Generic cluster of at most 16 CTAs, got "
                             f"cluster_shape={cluster_shape}, cta_group={cta_group_size}")
        planned_components = _planned_consumer_components(graph, pipeline, cta_group_size)
        if planned_components is None:
            consumer_count = cluster_ctas // cta_group_size
            consumer_masks = ((1 << cluster_ctas) - 1, )
        else:
            consumer_count, consumer_masks = planned_components
        if template == PIPELINE_TMA_GENERIC:
            consumer_count = bin(consumer_masks[0]).count("1") * _roles_threads(graph, consumers)
    if producer_count <= 0 or (consumer_count is not None and consumer_count <= 0):
        raise ValueError(
            f"{pipe_name}: barrier counts must be positive, got producer={producer_count}, consumer={consumer_count}")
    if template.producer.arrive == ArriveKind.MBAR_TX and pipeline["producer_kind"] != W_SMEM_TMA_TX:
        raise ValueError(f"{pipe_name}: TMA transaction arrival requires a TMA-load producer")
    if template.producer.arrive == ArriveKind.MBAR_UMMA and pipeline["producer_kind"] != W_TMEM_TCGEN05_COMMIT:
        raise ValueError(f"{pipe_name}: UMMA completion arrival requires an UMMA producer")
    if (template.consumer is not None and template.consumer.arrive == ArriveKind.MBAR_UMMA
            and pipeline["consumer_kind"] not in {R_SMEM_TCGEN05_COMMIT, R_TMEM_TCGEN05_COMMIT}):
        raise ValueError(f"{pipe_name}: UMMA completion arrival requires an UMMA consumer")
    if (template.consumer is not None and template.consumer.arrive == ArriveKind.STORE
            and pipeline["consumer_kind"] != R_SMEM_TMA_BULK):
        raise ValueError(f"{pipe_name}: TMA-store completion arrival requires a TMA-store consumer")
    return PipelineParticipation(
        cta_group_size,
        producer_count=producer_count,
        consumer_count=consumer_count,
        expected_tx_bytes=expected_tx_bytes,
        consumer_masks=consumer_masks,
    )


def validate_pipeline_protocols(graph: dict) -> None:
    """Resolve and validate every pipeline protocol after stream analysis."""
    next_named_barrier = 1
    for pipeline in graph["pipelines"]:
        template: PipelineSpec = pipeline["template"]
        if template.named_barrier:
            if next_named_barrier > 15:
                raise ValueError(f"{pipeline['name']}: exhausted the 15 user named-barrier IDs")
            pipeline["named_barrier_id"] = next_named_barrier
            next_named_barrier += 1
        pipeline["participation"] = barrier_participation(graph, pipeline)


def event_text(events: list) -> list[str]:
    return [
        f"{event.role}:{event.effect_kind.value}:{event.intrin}:"
        f"{event.access.value}:{event.mem}" for event in events
    ]


def side_kind(channel: str, side: str, events: list, access: Access) -> MemoryEffectKind:
    kinds = {event.effect_kind.for_access(access) for event in events}
    if len(kinds) > 1:
        detail = ", ".join(sorted(kind.value for kind in kinds))
        raise ValueError(f"{channel}: mixed {side} memory effects are unsupported: {detail}")
    return next(iter(kinds))


def validate_fixed_payload_regions(regions: list) -> None:
    """Require fixed async payloads to execute as one all-or-none group."""
    for region in regions:
        for kind in {W_SMEM_TMA_TX, R_SMEM_TMA_BULK}:
            op_paths: dict[object, tuple[tuple[str, int, str], ...]] = {}
            for item in region.events:
                event = item.event
                if event.effect_kind != kind:
                    continue
                previous = op_paths.setdefault(event.op_id, event.control_path)
                if previous != event.control_path:
                    raise ValueError(f"{region.role}: stage key {region.key} has inconsistent "
                                     f"{kind.value} structure for one operation")
            if not op_paths:
                continue
            paths = set(op_paths.values())
            # TMA loads fix mbarrier expect_tx bytes and must be all-or-none.
            # FIXME: adhoc — allow TMA_STORE under a runtime peer loop so AG
            # multicast can be a for+tma_st_2d instead of a dedicated intrinsic.
            # Formalize store-side loop rules separately from load expect_tx.
            if kind == W_SMEM_TMA_TX and any(frame[0] in {"for", "while"} for path in paths for frame in path):
                raise ValueError(f"{region.role}: stage key {region.key} {kind.value} payload "
                                 "cannot be controlled by a runtime loop")
            if kind == W_SMEM_TMA_TX and len(paths) != 1:
                raise ValueError(f"{region.role}: stage key {region.key} {kind.value} operations "
                                 "must share one all-or-none control path")


def infer_tma_payload_bytes(dataflow: Flow, regions: list) -> dict[tuple[int, str], int]:
    """Derive per-endpoint bytes from the logical-memory offset partition."""
    tma_bytes: dict[tuple[int, str], int] = {}
    for channel in dataflow.channels:
        for kind, side, pred in ((W_SMEM_TMA_TX, "producer", writes), ):
            by_role: dict[str, dict[tuple[str, ...], list]] = {}
            for role in channel_roles(channel, side):
                role_groups: dict[tuple[str, ...], list] = {}
                for region in regions:
                    if region.key != channel["key"] or region.role != role:
                        continue
                    events = [
                        item.event for item in region.events if item.event.effect_kind == kind
                        and item.event.mem in channel["via"] and pred(item.event.access)
                    ]
                    if not events:
                        continue
                    role_groups.setdefault(region.loops, []).append(region)
                if role_groups:
                    by_role[role] = role_groups
            if not by_role:
                continue

            stacks = {stack for groups in by_role.values() for stack in groups}
            for loops in stacks:
                count = max(len(groups.get(loops, ())) for groups in by_role.values())
                for index in range(count):
                    transaction = [
                        groups[loops][index] for groups in by_role.values() if index < len(groups.get(loops, ()))
                    ]
                    for role, byte_count in _tma_transaction_bytes(dataflow, channel, kind, transaction).items():
                        key = (channel["key"], role)
                        previous = tma_bytes.setdefault(key, byte_count)
                        if previous != byte_count:
                            raise ValueError(f"{role}: stage key {channel['key']} has varying TMA "
                                             f"payload bytes ({previous} vs {byte_count})")
    return tma_bytes


def _tma_transaction_bytes(dataflow: Flow, channel: dict, kind: MemoryEffectKind, regions: list) -> dict[str, int]:
    by_mem: dict[str, dict[object, object]] = {}
    for region in regions:
        for item in region.events:
            event = item.event
            if event.effect_kind != kind or event.mem not in channel["via"]:
                continue
            by_mem.setdefault(event.mem, {})[event.op_id] = event

    bytes_by_role: dict[str, int] = {}
    for mem_name in channel["via"]:
        events = list(by_mem.get(mem_name, {}).values())
        mem_shape = tuple(int(dim) for dim in dataflow.memories[mem_name].shape)
        expected = numel(mem_shape)
        if events and all(event.logical_coord is not None and event.logical_shape is not None for event in events):
            covered = 0
            for index, event in enumerate(events):
                coord = event.logical_coord
                shape = event.logical_shape
                if len(coord) != len(mem_shape) or len(shape) != len(mem_shape):
                    raise ValueError(f"stage key {channel['key']} TMA region for {mem_name!r} "
                                     f"must have rank {len(mem_shape)}")
                if any(start + size > limit for start, size, limit in zip(coord, shape, mem_shape)):
                    raise ValueError(f"stage key {channel['key']} TMA region {coord}+{shape} for "
                                     f"{mem_name!r} exceeds logical shape {mem_shape}")
                for previous in events[:index]:
                    if all(a < pb + ps and pb < a + size
                           for a, size, pb, ps in zip(coord, shape, previous.logical_coord, previous.logical_shape)):
                        raise ValueError(f"stage key {channel['key']} TMA regions for {mem_name!r} overlap")
                volume = math.prod(shape)
                covered += volume
                byte_count = _tma_payload_bytes(dataflow.memories[mem_name], volume)
                bytes_by_role[event.role] = bytes_by_role.get(event.role, 0) + byte_count
            if covered != expected:
                raise ValueError(f"stage key {channel['key']} TMA regions for {mem_name!r} cover "
                                 f"{covered} of {expected} elements")
            continue

        if any(event.logical_offset is None for event in events):
            raise ValueError(f"stage key {channel['key']} TMA partition for {mem_name!r} has no "
                             "static logical-memory region")
        events.sort(key=lambda event: event.logical_offset)
        offsets = [int(event.logical_offset) for event in events]
        if not offsets or offsets[0] != 0:
            raise ValueError(f"stage key {channel['key']} TMA partition for {mem_name!r} must "
                             "begin at element 0")
        if len(offsets) != len(set(offsets)):
            raise ValueError(f"stage key {channel['key']} TMA partition for {mem_name!r} has "
                             "duplicate logical-memory offsets")
        if offsets[-1] >= expected:
            raise ValueError(f"stage key {channel['key']} TMA partition for {mem_name!r} starts "
                             f"outside its {expected}-element logical extent")
        for event, end in zip(events, offsets[1:] + [expected], strict=True):
            extent = end - int(event.logical_offset)
            byte_count = _tma_payload_bytes(dataflow.memories[mem_name], extent)
            bytes_by_role[event.role] = bytes_by_role.get(event.role, 0) + byte_count
    return bytes_by_role


def _endpoint_events(mem_events: list, roles: tuple[str, ...], via: set[str], pred: Callable[[Access], bool]) -> list:
    return [event for event in mem_events if event.role in roles and event.mem in via and pred(event.access)]


def _validate_endpoint_kinds(
    channel: str,
    side: str,
    roles: tuple[str, ...],
    events: list,
    pred: Callable[[Access], bool],
    via: set[str],
    access: Access,
) -> MemoryEffectKind:
    kinds: set[MemoryEffectKind] = set()
    for role in roles:
        endpoint_events = [event for event in events if event.role == role and event.mem in via and pred(event.access)]
        if not endpoint_events:
            action = "writes" if side == "producer" else "reads"
            raise ValueError(f"{channel}: {role} has no {action} for {sorted(via)}")
        kinds.add(side_kind(channel, f"{side} {role}", endpoint_events, access))
    if len(kinds) != 1:
        detail = ", ".join(sorted(kind.value for kind in kinds))
        raise ValueError(f"{channel}: heterogeneous {side} memory effects are unsupported: {detail}")
    return next(iter(kinds))


def _select_pipeline_template(
    channel: str,
    producers: tuple[str, ...],
    consumers: tuple[str, ...],
    producer_kind: MemoryEffectKind,
    consumer_kind: MemoryEffectKind,
) -> PipelineSpec:
    template = PIPELINE_BY_EFFECT_KINDS.get((producer_kind, consumer_kind))
    if template is None:
        raise ValueError(f"{channel}: unsupported pipeline effect pair "
                         f"{producer_kind.value}->{consumer_kind.value}")
    if (template == PIPELINE_TMA_STORE_CROSS_ROLE and len(producers) == len(consumers) == 1
            and producers[0] == consumers[0]):
        return PIPELINE_TMA_STORE_SAME_ROLE
    return template


def analyze_channels(dataflow: Flow, mem_events: list) -> list[dict]:
    channels: list[dict] = []
    for channel in dataflow.channels:
        producers = channel_roles(channel, "producer")
        consumers = channel_roles(channel, "consumer")
        via = set(channel["via"])
        name = channel_name(producers, consumers)
        producer_writes = _endpoint_events(mem_events, producers, via, writes)
        consumer_reads = _endpoint_events(mem_events, consumers, via, reads)
        for mem in via:
            if not any(event.mem == mem for event in producer_writes):
                raw_mapping = getattr(dataflow, "memory_mapping_plan", ())
                raw_mapping = getattr(raw_mapping, "entries", raw_mapping)
                if isinstance(raw_mapping, dict):
                    entry = raw_mapping.get(mem)
                    physical = entry if isinstance(entry, str) else getattr(entry, "physical", mem)
                else:
                    entry = next((item for item in raw_mapping if getattr(item, "logical", None) == mem), None)
                    physical = getattr(entry, "physical", mem)
                if physical != mem and any(event.role in producers and event.mem == physical and writes(event.access)
                                           for event in mem_events):
                    raise ValueError(f"{name}: read-only channel alias {mem!r} -> {physical!r} is unsafe; "
                                     "the producer must write the logical payload")
                raise ValueError(f"{name}: no producer writes {mem!r}")
            if not any(event.mem == mem for event in consumer_reads):
                raise ValueError(f"{name}: no consumer reads {mem!r}")
        producer_kind = _validate_endpoint_kinds(name, "producer", producers, producer_writes, writes, via, Access.W)
        consumer_kind = _validate_endpoint_kinds(name, "consumer", consumers, consumer_reads, reads, via, Access.R)
        channels.append({
            "key": channel["key"],
            "name": name,
            "producers": producers,
            "consumers": consumers,
            "via": channel["via"],
            "template": _select_pipeline_template(name, producers, consumers, producer_kind, consumer_kind),
            "producer_kind": producer_kind,
            "consumer_kind": consumer_kind,
            "producer_writes": event_text(producer_writes),
            "consumer_reads": event_text(consumer_reads),
        })
    return channels


def lower_mpmc_pipelines(channels: list[dict], mem_events: list, tma_bytes: dict[tuple[int, str], int]) -> list[dict]:
    """Lower one logical MPMC channel to one runtime pipeline per producer."""
    lowered: list[dict] = []
    for channel in channels:
        producers = producer_roles(channel)
        if len(producers) == 1:
            lowered.append({
                **channel,
                "tx_bytes": tma_bytes.get((channel["key"], producers[0])),
            })
            continue

        for producer in producers:
            producer_events = [
                event for event in mem_events
                if event.role == producer and event.mem in channel["via"] and writes(event.access)
            ]
            producer_via = sorted({event.mem for event in producer_events})
            if not producer_via:
                raise ValueError(f"{channel['name']}: producer {producer} has no runtime payload")
            runtime_producers = (producer, )
            runtime_name = channel_name(runtime_producers, consumer_roles(channel))
            lowered.append({
                **channel,
                "name":
                runtime_name,
                "source_channel":
                channel["name"],
                "producers":
                runtime_producers,
                "producer_via":
                producer_via,
                "consumer_via":
                channel["via"],
                "tx_bytes":
                tma_bytes.get((channel["key"], producer)),
                "template":
                _select_pipeline_template(runtime_name, runtime_producers, consumer_roles(channel),
                                          channel["producer_kind"], channel["consumer_kind"]),
                "producer_writes":
                event_text(producer_events),
            })
    return lowered


def _channel_stream(channel: dict, role: str, side: str, regions: list) -> Counter[tuple[str, ...]]:
    pred = writes if side == "producer" else reads
    via = set(channel["via"])
    raw_stream = Counter(region.loops
                         for region in regions
                         if region.key == channel["key"] and region.role == role and any(
                             item.event.mem in via and pred(item.event.access) for item in region.events))
    if not raw_stream:
        raise ValueError(f"{channel['name']}: {role} has no bound {side} transaction stream")
    unproven = [loop for stack in raw_stream for loop in stack if loop.startswith("unproven:")]
    if unproven:
        raise ValueError(f"{channel['name']}: cannot prove {role} {side} transaction count "
                         f"for {unproven[0]}")
    normalized: Counter[tuple[str, ...]] = Counter()
    for stack, count in raw_stream.items():
        factor = 1
        numerator = count
        normalized_stack = []
        for loop in stack:
            repeat_prefix, repeat_separator, repeat_suffix = loop.partition("|")
            if repeat_prefix == "repeat" and repeat_separator and repeat_suffix.isdigit():
                numerator *= int(repeat_suffix)
                continue
            prefix, separator, suffix = loop.rpartition("|step")
            if not separator:
                normalized_stack.append(loop)
                continue
            step = int(suffix) if separator and suffix.isdigit() else 1
            base, scale_separator, scale_suffix = prefix.rpartition("|scale")
            scale = int(scale_suffix) if scale_separator and scale_suffix.isdigit() else 1
            factor *= step
            numerator *= scale
            normalized_stack.append(f"{base if scale_separator else prefix}|step1")
        if numerator % factor == 0:
            normalized[tuple(normalized_stack)] += numerator // factor
        else:
            normalized[stack] += count
    return normalized


def _stream_text(stream: Counter[tuple[str, ...]]) -> str:
    return ", ".join(f"{count}×({' / '.join(stack) if stack else 'once'})" for stack, count in sorted(stream.items()))


def _is_rmw_pair(dataflow: Flow, channels: list[dict], mem: str) -> bool:
    if len(channels) != 2 or dataflow.memories[mem].__class__.__name__ != "TMem":
        return False
    a_prod = channel_roles(channels[0], "producer")
    a_cons = channel_roles(channels[0], "consumer")
    b_prod = channel_roles(channels[1], "producer")
    b_cons = channel_roles(channels[1], "consumer")
    return (len(a_prod) == len(a_cons) == len(b_prod) == len(b_cons) == 1 and a_prod == b_cons and a_cons == b_prod)


def validate_channels(dataflow: Flow, channels: list[dict], regions: list) -> None:
    """Validate unique ownership and equivalent endpoint transaction counts."""
    owners: dict[str, list[dict]] = {}
    for channel in dataflow.channels:
        for mem in channel["via"]:
            owners.setdefault(mem, []).append(channel)
    for mem, owned_channels in owners.items():
        if len(owned_channels) > 1 and not _is_rmw_pair(dataflow, owned_channels, mem):
            keys = [channel["key"] for channel in owned_channels]
            raise ValueError(f"logical memory {mem!r} belongs to multiple channels {keys}; "
                             "declare one MPMC depends channel instead")

    for channel in channels:
        endpoint_streams = [
            (role, "producer", _channel_stream(channel, role, "producer", regions)) for role in producer_roles(channel)
        ] + [(role, "consumer", _channel_stream(channel, role, "consumer", regions))
             for role in consumer_roles(channel)]
        expected = endpoint_streams[0][2]
        for role, side, stream in endpoint_streams[1:]:
            if stream != expected:
                raise ValueError(f"{channel['name']}: transaction stream mismatch; expected "
                                 f"{_stream_text(expected)}, but {role} {side} executes "
                                 f"{_stream_text(stream)}")


__all__ = [
    "analyze_channels",
    "barrier_participation",
    "channel_roles",
    "channel_name",
    "consumer_roles",
    "dtype_size",
    "event_text",
    "infer_tma_payload_bytes",
    "lower_mpmc_pipelines",
    "numel",
    "PipelineParticipation",
    "producer_roles",
    "reads",
    "side_kind",
    "validate_channels",
    "validate_fixed_payload_regions",
    "validate_pipeline_protocols",
    "writes",
]
