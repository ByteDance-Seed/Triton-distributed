from threading import Lock
import functools
import subprocess
import torch

class MxSmiUtil:

    @staticmethod
    def get_mxlink_adjacency_matrix():
        output = subprocess.check_output(["mx-smi", "topo", "-m"], text=True)
        lines = [line.strip() for line in output.split("\n") if line.startswith("GPU")]

        device_count = len(lines)
        matrix = [[-1 for _ in range(device_count)] for _ in range(device_count)]

        # 解析每行数据
        for i, line in enumerate(lines):
            parts = line.split()
            for j in range(1, len(parts)):
                if "MX" in parts[j]:
                    matrix[i][j - 1] = 1

        return matrix

    @staticmethod
    def get_gpu_numa_node(gpu_index=0):
        # get GPU PCI bus ID
        cmd = f"mx-smi topo -t"
        pci_id = subprocess.check_output(cmd, shell=True).decode().strip()
        lines = pci_id.splitlines()
        gpu_to_addr = {}
        current_addr = None
        for line in lines:
            branch_line = line.strip()
            parts = branch_line.split('-+-')
            last_part = parts[-1].strip()
            if line.startswith('+-pci') or line.startswith('\\-pci'):
                current_addr = parts[1]
                if 'GPU#' in last_part:
                    comp, gpu_str = last_part.split(' ')
                    gpu_num = int(gpu_str[4:])
                    gpu_to_addr[gpu_num] = current_addr
            elif "GPU#" in parts[-1]:
                comp, gpu_str = last_part.split(' ')
                gpu_num = int(gpu_str[4:])
                gpu_to_addr[gpu_num] = current_addr

        pci_address = gpu_to_addr[gpu_index]        
        # use sysfs to get NUMA id
        numa_node_path = f"/sys/bus/pci/devices/0000:{pci_address}/numa_node"
        with open(numa_node_path, "r") as f:
            numa_node = int(f.read().strip())

        assert numa_node >= 0
        return numa_node if numa_node >= 0 else 0


_lock = Lock()


@functools.lru_cache()
def get_numa_node(gpu_index):
    return MxSmiUtil.get_gpu_numa_node(gpu_index)


@functools.lru_cache()
def has_fullmesh_mxlink():
    mxlink_matrix = MxSmiUtil.get_mxlink_adjacency_matrix()
    has_mxlink = any([any(x == 1 for x in row) for row in mxlink_matrix])
    _has_fullmesh_mxlink = all([i == j or v == 1 for i, row in enumerate(mxlink_matrix) for j, v in enumerate(row)])
    if has_mxlink and not _has_fullmesh_mxlink:
        print(
            "found mxlink but not fullmesh MXLink, this may cause undefined behavior, please check your GPU topology"
        )
    return _has_fullmesh_mxlink

@functools.lru_cache()
def has_fullmesh_mxlink_ngpus(ngpus):
    assert ngpus > 0  and torch.cuda.device_count() % ngpus == 0, "device_count mod(ngpus) must == 0"
    mxlink_matrix = MxSmiUtil.get_mxlink_adjacency_matrix()

    for first_dev in range(0, torch.cuda.device_count(), ngpus):
        dev_ids = [first_dev + i for i in range(ngpus)]
        _has_fullmesh_mxlink = all([i == j or mxlink_matrix[i][j] == 1 for i in dev_ids for j in dev_ids])
        if not _has_fullmesh_mxlink:
            return False
    
    return True

@functools.lru_cache()
def get_numa_world_size():
    numa_node = [get_numa_node(n) for n in range(torch.cuda.device_count())]
    numa_node_set = set(numa_node)
    assert len(numa_node_set) <= 2  # TODO(houqi.1993) only 2 NUMA node supported now.
    if len(numa_node_set) == 1:
        return torch.cuda.device_count()

    gpu_count_per_numa = [numa_node.count(x) for x in numa_node_set]
    assert gpu_count_per_numa[0] == gpu_count_per_numa[1]
    return torch.cuda.device_count() // 2

