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

#include "flash_comm/buffer/symmetric_memory.h"
#include <c10/core/StorageImpl.h>
#include <torch/extension.h>

void bind_symmetric_memory(py::module &m) {
  py::enum_<flash_comm::buffer::BlockBackend>(m, "BlockBackend")
      .value("CUDA_IPC", flash_comm::buffer::BlockBackend::CUDA_IPC)
      .value("VMM", flash_comm::buffer::BlockBackend::VMM)
      .export_values();

  // Helper functions for support check
  m.def("is_fabric_supported",
        &flash_comm::buffer::ShareableBlock::is_fabric_supported,
        "Check if Fabric Handle is supported");
  m.def("is_vmm_supported",
        &flash_comm::buffer::ShareableBlock::is_vmm_supported,
        "Check if VMM is supported");

  // Helper to create tensor from raw pointer (bypassing strict checks)
  m.def(
      "create_tensor_from_ptr",
      [](uint64_t ptr, std::vector<int64_t> shape, torch::ScalarType dtype,
         int device_id) {
        auto options =
            torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device_id);

        int64_t numel = 1;
        for (auto s : shape)
          numel *= s;
        size_t element_size = c10::elementSize(dtype);
        size_t total_bytes = numel * element_size;

        // Create DataPtr with no-op deleter
        c10::DataPtr data_ptr((void *)ptr, nullptr, [](void *) {},
                              torch::Device(torch::kCUDA, device_id));

        auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
            c10::StorageImpl::use_byte_size_t(), total_bytes,
            std::move(data_ptr),
            /*allocator=*/nullptr,
            /*resizable=*/false);

        at::Tensor tensor = torch::empty({0}, options);
        tensor.set_(at::Storage(storage_impl), 0, shape);
        return tensor;
      },
      "Create a tensor from a raw pointer");

  py::class_<flash_comm::buffer::SymmetricMemory>(m, "SymmetricMemory")
      .def(py::init<size_t, flash_comm::buffer::BlockBackend>(),
           py::arg("size"), py::arg("backend"))
      .def("get_local_ptr",
           [](const flash_comm::buffer::SymmetricMemory &sm) {
             return (uint64_t)sm.get_local_ptr();
           })
      .def("get_handle",
           [](const flash_comm::buffer::SymmetricMemory &sm) {
             auto pair = sm.get_handle();
             // Return tuple (backend, bytes)
             return py::make_tuple(
                 pair.first, py::bytes(pair.second.data(), pair.second.size()));
           })
      .def("register_peer",
           [](flash_comm::buffer::SymmetricMemory &sm, int peer_rank,
              const py::bytes &handle) {
             std::string h_str = handle;
             std::vector<char> h_vec(h_str.begin(), h_str.end());
             sm.register_peer(peer_rank, h_vec);
           })
      .def("get_peer_ptr",
           [](const flash_comm::buffer::SymmetricMemory &sm, int peer_rank) {
             return (uint64_t)sm.get_peer_ptr(peer_rank);
           });
}
