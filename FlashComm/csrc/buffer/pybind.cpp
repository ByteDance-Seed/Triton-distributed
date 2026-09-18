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

#include "flash_comm/buffer/nccl_gin.h"
#include "flash_comm/buffer/nccl_symmetric_memory.h"
#include "flash_comm/buffer/symmetric_memory.h"
#include "flash_comm/common.h"
#include <c10/core/StorageImpl.h>
#include <torch/extension.h>
#include <vector>

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

  m.attr("NCCL_GIN_CONNECTION_NONE") =
      static_cast<int>(NCCL_GIN_CONNECTION_NONE);
  m.attr("NCCL_GIN_CONNECTION_FULL") =
      static_cast<int>(NCCL_GIN_CONNECTION_FULL);
  m.attr("NCCL_GIN_CONNECTION_RAIL") =
      static_cast<int>(NCCL_GIN_CONNECTION_RAIL);

  m.def("nccl_gin_unique_id_bytes",
        &flash_comm::buffer::nccl_gin_unique_id_bytes);
  m.def(
      "nccl_gin_get_unique_id",
      []() {
        std::vector<uint8_t> buf(
            flash_comm::buffer::nccl_gin_unique_id_bytes());
        flash_comm::buffer::nccl_gin_get_unique_id(buf.data());
        return torch::tensor(buf, torch::dtype(torch::kUInt8));
      },
      "NCCL unique id for GIN communicator bootstrap");
  m.def(
      "nccl_gin_init",
      [](const torch::Tensor &uid, int rank, int nranks, int local_world_size,
         int gin_contexts, int gin_signals, int rail_barriers,
         int gin_queue_depth, int gin_connection_type, int ep_num_qps) {
        FLASH_CHECK(uid.device().is_cpu()) << "uid must be a CPU tensor";
        FLASH_CHECK(uid.is_contiguous()) << "uid must be contiguous";
        FLASH_CHECK(uid.scalar_type() == torch::kUInt8);
        FLASH_CHECK(uid.numel() ==
                    flash_comm::buffer::nccl_gin_unique_id_bytes())
            << "uid has invalid length";
        return flash_comm::buffer::nccl_gin_init_rank(
            uid.data_ptr(), static_cast<int>(uid.numel()), rank, nranks,
            local_world_size, gin_contexts, gin_signals, rail_barriers,
            gin_queue_depth, gin_connection_type, ep_num_qps);
      },
      py::arg("uid"), py::arg("rank"), py::arg("nranks"),
      py::arg("local_world_size"), py::arg("gin_contexts"),
      py::arg("gin_signals"), py::arg("rail_barriers"),
      py::arg("gin_queue_depth"),
      py::arg("gin_connection_type") =
          static_cast<int>(NCCL_GIN_CONNECTION_FULL),
      py::arg("ep_num_qps") = 1);
  m.def("nccl_gin_destroy", &flash_comm::buffer::nccl_gin_destroy_rank);
  m.def("nccl_gin_retain", &flash_comm::buffer::nccl_gin_retain_rank);
  m.def("nccl_gin_release", &flash_comm::buffer::nccl_gin_release_rank);
  m.def("nccl_gin_is_initialized",
        &flash_comm::buffer::nccl_gin_is_initialized);
  m.def("nccl_gin_dev_comm_bytes", []() {
    return torch::from_blob(const_cast<ncclDevComm *>(
                                flash_comm::buffer::nccl_gin_dev_comm()),
                            {static_cast<int64_t>(sizeof(ncclDevComm))},
                            torch::kUInt8)
        .clone();
  });
  m.def("nccl_gin_rank", &flash_comm::buffer::nccl_gin_rank);
  m.def("nccl_gin_nranks", &flash_comm::buffer::nccl_gin_nranks);
  m.def("nccl_gin_local_world_size",
        &flash_comm::buffer::nccl_gin_local_world_size);
  m.def("nccl_gin_lsa_rank", &flash_comm::buffer::nccl_gin_lsa_rank);
  m.def("nccl_gin_lsa_size", &flash_comm::buffer::nccl_gin_lsa_size);
  m.def("nccl_gin_type", &flash_comm::buffer::nccl_gin_type);

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

  py::class_<flash_comm::buffer::NcclSymmetricMemory>(m, "NcclSymmetricMemory")
      .def(py::init<size_t>(), py::arg("size_bytes"))
      .def(py::init([](uint64_t ptr, size_t size_bytes) {
             return std::make_unique<flash_comm::buffer::NcclSymmetricMemory>(
                 flash_comm::buffer::nccl_gin_comm(),
                 reinterpret_cast<void *>(ptr), size_bytes,
                 NCCL_WIN_COLL_SYMMETRIC);
           }),
           py::arg("ptr"), py::arg("size_bytes"))
      .def("get_local_ptr",
           [](const flash_comm::buffer::NcclSymmetricMemory &m) {
             return reinterpret_cast<uint64_t>(m.get_local_ptr());
           })
      .def("get_peer_ptr",
           [](const flash_comm::buffer::NcclSymmetricMemory &m, int peer_rank) {
             return reinterpret_cast<uint64_t>(m.get_peer_ptr(peer_rank));
           })
      .def("get_window_handle",
           [](const flash_comm::buffer::NcclSymmetricMemory &m) {
             return reinterpret_cast<uint64_t>(m.get_window());
           })
      .def("get_size", &flash_comm::buffer::NcclSymmetricMemory::get_size);
}
