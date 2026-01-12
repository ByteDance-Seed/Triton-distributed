#include <cuda.h>

__global__ void flush_l2c_dev() {
  // TODO
}

extern "C" void flush_l2c(cudaStream_t stream) {
  flush_l2c_dev<<<1, 64, 0, stream>>>();
}