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

import os
import time
import shutil
import torch
import torch.distributed as dist


class TorchDistContext:
    
    def __init__(self, rank, world_size, master_addr="localhost", master_port="12335",
                 device_id=None, backend="cpu:gloo,cuda:nccl"):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.device_id = device_id if device_id is not None else self.rank
        self.backend = backend
    
    def __enter__(self):
        if self.master_addr is not None:
            os.environ["MASTER_ADDR"] = self.master_addr
        if self.master_port is not None:
            os.environ["MASTER_PORT"] = str(self.master_port)
        
        torch.cuda.set_device(self.device_id)
        device = torch.device("cuda", self.device_id)
        
        dist.init_process_group(
            backend=self.backend,
            rank=self.rank,
            world_size=self.world_size,
            device_id=device,
        )
        
        # Register the "default" process group for mori
        world_group = torch.distributed.group.WORLD
        assert world_group is not None
        torch._C._distributed_c10d._register_process_group("default", world_group)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def _test_mori_shmem_init(rank, world_size, master_port):
    """Test mori shmem initialization with PyTorch process group"""
    
    with TorchDistContext(rank=rank, world_size=world_size, master_port=str(master_port)):
        import mori.shmem as mori_shmem
        
        mori_shmem.shmem_torch_process_group_init("default")
        
        my_pe = mori_shmem.shmem_mype()
        n_pes = mori_shmem.shmem_npes()
        
        assert my_pe == rank, f"PE mismatch: expected {rank}, got {my_pe}"
        assert n_pes == world_size, f"nPEs mismatch: expected {world_size}, got {n_pes}"
        
        ptr1 = mori_shmem.shmem_malloc(4096)
        assert ptr1 != 0, f"shmem_malloc returned NULL"
        
        ptr2 = mori_shmem.shmem_malloc_align(256, 8192)
        assert ptr2 != 0, f"shmem_malloc_align returned NULL"
        assert ptr2 % 256 == 0, f"Memory not aligned: 0x{ptr2:x}"
        
        mori_shmem.shmem_barrier_all()
        
        mori_shmem.shmem_free(ptr1)
        mori_shmem.shmem_free(ptr2)
        
        mori_shmem.shmem_finalize()
        
        if rank == 0:
            print(f"PyTorch init test passed with {world_size} processes")


def _test_mori_uniqueid_init(rank, world_size, master_port):
    """Test mori shmem initialization with UniqueID (no PyTorch distributed)"""
    
    import mori.shmem as mori_shmem
    
    # Set GPU device for this rank
    torch.cuda.set_device(rank)
    
    os.environ['MORI_SOCKET_IFNAME'] = 'lo'
    
    uid_dir = f"/tmp/mori_shmem_test_{master_port}"
    uid_file = os.path.join(uid_dir, "uniqueid")
    ready_file = os.path.join(uid_dir, f"ready_{rank}")
    
    try:
        if rank == 0:
            os.makedirs(uid_dir, exist_ok=True)
            unique_id = mori_shmem.shmem_get_unique_id()
            assert len(unique_id) == 128, f"Invalid unique ID length: {len(unique_id)}"
            
            with open(uid_file, 'wb') as f:
                f.write(unique_id)
            with open(ready_file, 'w') as f:
                f.write('ready')
        else:
            max_wait = 30
            for i in range(max_wait * 10):
                if os.path.exists(uid_dir):
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(f"Timeout waiting for directory")
            
            rank0_ready = os.path.join(uid_dir, "ready_0")
            for i in range(max_wait * 10):
                if os.path.exists(rank0_ready) and os.path.exists(uid_file):
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(f"Timeout waiting for unique ID file")
            
            with open(uid_file, 'rb') as f:
                unique_id = f.read()
        
        assert len(unique_id) == 128, f"Invalid unique ID length: {len(unique_id)}"
        time.sleep(0.01 * rank)
        
        ret = mori_shmem.shmem_init_attr(
            mori_shmem.MORI_SHMEM_INIT_WITH_UNIQUEID,
            rank,
            world_size,
            unique_id
        )
        assert ret == 0, f"shmem_init_attr failed with code {ret}"
        
        my_rank = mori_shmem.shmem_mype()
        npes = mori_shmem.shmem_npes()
        assert my_rank == rank, f"Rank mismatch: expected {rank}, got {my_rank}"
        assert npes == world_size, f"World size mismatch: expected {world_size}, got {npes}"
        
        ptr1 = mori_shmem.shmem_malloc(4096)
        assert ptr1 != 0, f"shmem_malloc returned NULL"
        
        ptr2 = mori_shmem.shmem_malloc_align(256, 8192)
        assert ptr2 != 0, f"shmem_malloc_align returned NULL"
        assert ptr2 % 256 == 0, f"Memory not aligned: 0x{ptr2:x}"
        
        mori_shmem.shmem_barrier_all()
        
        mori_shmem.shmem_free(ptr1)
        mori_shmem.shmem_free(ptr2)
        
        mori_shmem.shmem_finalize()
        
        if rank == 0:
            print(f"UniqueID init test passed with {world_size} processes")
        
    finally:
        if rank == 0:
            time.sleep(2.0)
            if os.path.exists(uid_dir):
                shutil.rmtree(uid_dir, ignore_errors=True)


def test_mori_shmem_torch_init():
    """Test mori shmem with PyTorch process group initialization"""
    
    world_size = int(os.environ.get('WORLD_SIZE', 8))
    master_port = int(os.environ.get('MASTER_PORT', 29500))
    
    torch.multiprocessing.spawn(
        _test_mori_shmem_init,
        args=(world_size, master_port),
        nprocs=world_size,
        join=True,
    )


def test_mori_shmem_uniqueid_init():
    """Test mori shmem with UniqueID initialization (no PyTorch distributed)"""
    
    world_size = int(os.environ.get('WORLD_SIZE', 8))
    master_port = int(os.environ.get('MASTER_PORT', 29501))
    
    torch.multiprocessing.spawn(
        _test_mori_uniqueid_init,
        args=(world_size, master_port),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    test_mori_shmem_torch_init()
    test_mori_shmem_uniqueid_init()
    print("All tests passed!")
