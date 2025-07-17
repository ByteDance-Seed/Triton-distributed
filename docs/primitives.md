# Primitives Provided by Triton-distributed

All the primitives are exposed by `triton_dist.language`
#### Low-level primitives
##### Context Querying Primitives
```py
rank(axis=-1, _builder=None)
num_ranks(axis=-1, _builder=None)
symm_at(ptr, rank, _builder=None)

```
##### Signal Control Primitives
```py
wait(barrierPtrs, numBarriers, scope: str, semantic: str, _builder=None)
consume_token(value, token, _builder=None)
notify(ptr, rank, signal=1, sig_op="set", comm_scope="inter_node", _builder=None)
```
##### NVSHMEM-related Primitives

Besides the primitives, Triton-distributed also expose all the NVSHMEM primitives to Python, allowing users to program communication kernels purely in Python.

All the NVSHMEM-related device-side primitives are exposed by `triton.language.extra.libshmem_device`
```py
my_pe()
n_pes()
int_p(dest, value, pe)
remote_ptr(local_ptr, pe)
barrier_all()
barrier_all_block()
barrier_all_warp()
sync_all()
sync_all_block()
sync_all_warp()
quiet()
fence()
getmem_nbi_block(dest, source, bytes, pe)
getmem_block(dest, source, bytes, pe)
getmem_nbi_warp(dest, source, bytes, pe)
getmem_warp(dest, source, bytes, pe)
getmem_nbi(dest, source, bytes, pe)
getmem(dest, source, bytes, pe)
putmem_block(dest, source, bytes, pe)
putmem_nbi_block(dest, source, bytes, pe)
putmem_warp(dest, source, bytes, pe)
putmem_nbi_warp(dest, source, bytes, pe)
putmem(dest, source, bytes, pe)
putmem_nbi(dest, source, bytes, pe)
putmem_signal_nbi(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal_nbi_block(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal_block(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal_nbi_warp(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal_warp(dest, source, bytes, sig_addr, signal, sig_op, pe)
signal_op(sig_addr, signal, sig_op, pe)
signal_wait_until(sig_addr, cmp_, cmp_val)
```

#### High-level primitives
To provide better programming experience, we also provide a set of high-level primitives for communication and signal control. These primitives, as decribed in our [MLSys 2025 paper](https://mlsys.org/virtual/2025/poster/3248), use a tile-centric design philosophy. These high-level primitives will be released soon after MLSys 2025.