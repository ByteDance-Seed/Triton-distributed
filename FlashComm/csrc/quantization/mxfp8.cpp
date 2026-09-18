/*
 * Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "flash_comm/quantization/mxfp8.h"
#include "flash_comm/torch_utils.h"

namespace flash_comm {
namespace quantization {

void mxfp8_quantize(torch::Tensor x, torch::Tensor packed, int32_t num_sm) {
  check_tensor_common(x, "x", true, torch::kBFloat16, 2);
  check_tensor_common(packed, "packed", true, torch::kBFloat16, 2);
  int32_t num_token = x.size(0);
  int32_t hidden_size = x.size(1);
  int32_t packed_row_bytes = mxfp8_packed_row_bytes(hidden_size);
  check_tensor_shape(packed, "packed", {num_token, packed_row_bytes / 2});
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  mxfp8_quantize_cuda(x.data_ptr(), packed.data_ptr(), num_token, hidden_size,
                      packed_row_bytes, num_sm, stream);
}

void bind_quantization_ops(py::module &m) {
  m.def("mxfp8_packed_row_bytes", &mxfp8_packed_row_bytes,
        py::arg("hidden_size"),
        "Packed MXFP8 row size in bytes (block-32 E4M3 + E8M0, transport "
        "aligned)");
  m.def("mxfp8_quantize", &mxfp8_quantize, py::arg("x"), py::arg("packed"),
        py::arg("num_sm") = 0, "Quantize BF16 rows to packed block-32 MXFP8");
}

} // namespace quantization
} // namespace flash_comm
