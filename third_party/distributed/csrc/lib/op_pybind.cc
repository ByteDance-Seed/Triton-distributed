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
#include "apis.cuh"
#include "registry.h"
#include <c10/cuda/CUDAStream.h>
#ifdef USE_TRITON_DISTRIBUTED_AOT
#include "triton_aot_generated/triton_distributed_kernel.h"
#endif

namespace distributed {

namespace ops {

PYBIND11_MODULE(libtriton_distributed, m) {
  m.doc() = "pybind11 distributed.";
  auto d = m.def_submodule("distributed");

  OpInitRegistry::instance().register_one(
      "moe_ag_scatter_align_block_size", [](py::module &m) {
        m.def("moe_ag_scatter_align_block_size",
              &moe_ag_scatter_align_block_size_op);
      });

  OpInitRegistry::instance().initialize_all(d);
}

} // namespace ops

} // namespace distributed