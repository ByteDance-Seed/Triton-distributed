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
import functools
import warnings

import torch
import os
import subprocess

from triton_dist.utils import has_fullmesh_nvlink


@functools.lru_cache()
def get_network_interfaces(no_local_and_loopbacks=True):
    """Get list of all network interfaces using sysfs"""

    def _is_local_interface(interface):
        return (interface.startswith("lo") or interface.startswith("docker") or interface.startswith("carma_br")
                or interface.startswith("veth") or interface.startswith("br-") or interface.startswith("tun")
                or interface.startswith("lxc") or interface.startswith("qemu"))

    nics = os.listdir("/sys/class/net/")
    if no_local_and_loopbacks:
        nics = [nic for nic in nics if not _is_local_interface(nic)]
    return nics


@functools.lru_cache(maxsize=16)
def get_max_nic_bandwidth_gpbs(interface="eth0"):
    """
    Get maximum theoretical bandwidth of a NIC (in Gbps).
    Returns -1 if unknown.
    """
    # Linux sysfs method
    if os.path.exists(f"/sys/class/net/{interface}/speed"):
        try:
            with open(f"/sys/class/net/{interface}/speed", "r") as f:
                speed_mbps = int(f.read().strip())
                return speed_mbps / 1000  # Convert Mbps to Gbps
        except Exception:
            pass

    # Linux ethtool fallback
    try:
        result = subprocess.run(["ethtool", interface], capture_output=True, text=True, check=True)
        for line in result.stdout.split("\n"):
            if "Speed:" in line:
                speed_str = line.split("Speed:")[1].strip()
                if "Gb/s" in speed_str:
                    return float(speed_str.replace("Gb/s", "").strip())
                elif "Mb/s" in speed_str:
                    return float(speed_str.replace("Mb/s", "").strip()) / 1000
    except Exception:
        pass

    warnings.warn(f"can not get max bandwidth of interface {interface}, return 0...")
    return 0


@functools.lru_cache()
def get_nic_gbps_per_gpu():
    """ in GB/s not Gbps """
    interfaces = get_network_interfaces()
    bws = [get_max_nic_bandwidth_gpbs(interface) for interface in interfaces]
    # suppose use the max speed ones only
    max_bw = max(bws)
    outs = [(interface, bw) for interface, bw in zip(interfaces, bws) if bw == max_bw]
    total_bw = sum([x[1] for x in outs]) / 8  # from Gbps => GB/s
    return total_bw / torch.cuda.device_count()


def estimate_reduce_scatter_time_ms(nbytes, world_size, local_world_size, intranode_bw, internode_bw):
    """
    intranode_bw/internode_bw in GB/s
    """
    if world_size != local_world_size:
        assert world_size % local_world_size == 0
        nnodes = world_size // local_world_size
        intra_node_ms = nbytes / world_size * (local_world_size - 1) / 1e6 / intranode_bw
        inter_node_ms = nbytes / world_size / 1e6 / internode_bw
        if has_fullmesh_nvlink():
            # with nvlink full mesh, intra/inter node overlaps
            return min(intra_node_ms, inter_node_ms) * (nnodes - 1) + intra_node_ms
        else:
            return (intra_node_ms + inter_node_ms) * (nnodes - 1) + intra_node_ms

    return nbytes / 1e6 / local_world_size * (local_world_size - 1) / intranode_bw


def estimate_all_gather_time_ms(nbytes, world_size, local_world_size, intranode_bw, internode_bw):
    """ just as reduce_scatter.
    intranode_bw/internode_bw in GB/s
    """
    return estimate_reduce_scatter_time_ms(nbytes, world_size, local_world_size, intranode_bw, internode_bw)
