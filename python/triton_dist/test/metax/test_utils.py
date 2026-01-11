import torch
import triton

from triton_dist.utils import (
    has_fullmesh_nvlink,
    get_numa_node,
)

full_mesh = has_fullmesh_nvlink()    
numa_node = get_numa_node(0)    
print(full_mesh)    
print(numa_node)    

    
