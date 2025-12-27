/*
 * Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
 */
#include <optional>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "python/src/ir.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"

#include "TritonDistributed/Dialect/Distributed/IR/Dialect.h"
#include "TritonDistributed/Dialect/SIMT/IR/Dialect.h"
#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"

namespace {

namespace py = pybind11;
using namespace mlir;
using namespace triton;

llvm::raw_fd_ostream &mlir_dumps() {
  std::error_code EC;
  static llvm::raw_fd_ostream S(::triton::tools::getStrEnv("MLIR_DUMP_PATH"),
                                EC, llvm::sys::fs::CD_CreateAlways);
  assert(!EC);
  return S;
}

llvm::raw_ostream &mlir_dumps_or_dbgs() {
  if (!::triton::tools::getStrEnv("MLIR_DUMP_PATH").empty()) {
    return mlir_dumps();
  } else {
    return llvm::dbgs();
  }
}

struct DistributedOpBuilder : public TritonOpBuilder {};

std::string locationToString(Location loc) {
  std::string str;
  llvm::raw_string_ostream os(str);
  loc.print(os);
  os.flush(); // Make sure all the content is dumped into the 'str' string
  return str;
}

// Function to parse a comma-separated string into a vector of C-style strings
llvm::SmallVector<const char *, 3>
parseCommaSeparatedValues(const std::string &input,
                          llvm::SmallVector<std::string, 3> &storage) {
  llvm::SmallVector<StringRef, 3> split;
  llvm::SmallVector<const char *, 3> result;
  StringRef(input.c_str()).split(split, ',');
  llvm::transform(split, std::back_inserter(result), [&storage](StringRef str) {
    // StringRefs are not always null-terminated.
    // The purpose for this storage pattern is to
    // produce a collection of C-strings that are.
    storage.push_back(str.str());
    return storage.back().c_str();
  });
  return result;
}

void outputWarning(Location loc, const std::string &msg) {
  std::string locStr = locationToString(loc);

  PyErr_WarnEx(PyExc_UserWarning, (locStr + ": " + msg).c_str(),
               /*stack_level=*/2);
}

// Allow dump a reproducer in the console on crash.
struct ConsoleReproducerStream : public mlir::ReproducerStream {
  ~ConsoleReproducerStream() override {}

  StringRef description() override {
    return "std::errs, please share the reproducer above with Triton project.";
  }
  raw_ostream &os() override { return llvm::errs(); }
};

static ReproducerStreamFactory makeConsoleReproducer() {
  return [](std::string &error) -> std::unique_ptr<ReproducerStream> {
    return std::make_unique<ConsoleReproducerStream>();
  };
}

} // anonymous namespace

/*****************************************************************************/
/* Python bindings for ir                                                    */
/*****************************************************************************/

void init_triton_distributed_ir(py::module &&m) {
  using ret = py::return_value_policy;
  using namespace pybind11::literals;

  py::enum_<distributed::SignalOp>(m, "SIGNAL_OP", py::module_local())
      .value("SET", distributed::SignalOp::SET)
      .value("ADD", distributed::SignalOp::ADD)
      .export_values();

  py::enum_<distributed::CommScope>(m, "COMM_SCOPE", py::module_local())
      .value("GPU", distributed::CommScope::GPU)
      .value("INTRA_NODE", distributed::CommScope::INTRA_NODE)
      .value("INTER_NODE", distributed::CommScope::INTER_NODE)
      .export_values();

  m.def("load_dialects", [](MLIRContext &context) {
    DialectRegistry registry;
    registry
        .insert<TritonDialect, ::mlir::triton::gpu::TritonGPUDialect,
                math::MathDialect, arith::ArithDialect, scf::SCFDialect,
                tensor::TensorDialect, ::mlir::gpu::GPUDialect,
                cf::ControlFlowDialect, ::mlir::triton::proton::ProtonDialect,
                ::mlir::triton::distributed::DistributedDialect,
                ::mlir::triton::simt::SIMTDialect, LLVM::LLVMDialect,
                mlir::ub::UBDialect>();
    mlir::LLVM::registerInlinerInterface(registry);
    registerBuiltinDialectTranslation(registry);
    registerLLVMDialectTranslation(registry);
    mlir::LLVM::registerInlinerInterface(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  // simt ops
  py::class_<triton::simt::BlockYieldOp, OpState>(m, "BlockYieldOp",
                                                  py::module_local());
  py::class_<triton::simt::SIMTExecRegionOp, OpState>(m, "SIMTExecRegionOp",
                                                      py::module_local())
      .def(
          "get_simt_entry_block",
          [](triton::simt::SIMTExecRegionOp &self) -> Block * {
            return &self.getDefaultRegion().front();
          },
          ret::reference);

  // DistributedOpBuilder inherit the original triton builder and add the
  // support of TritonDitributed. In this way, we can directly use this new
  // builder without maintaining two builders on the Python side. Because the
  // `InsertionPoint` needs to be consistent between builders.
  py::class_<DistributedOpBuilder, TritonOpBuilder>(
      m, "DistributedOpBuilder", py::module_local(), py::dynamic_attr())
      .def(py::init<MLIRContext *>())
      // SIMT ops
      .def("create_get_thread_id",
           [](TritonOpBuilder &self) -> Value {
             // triton only use 1D thread block
             Value tid = self.create<::mlir::gpu::ThreadIdOp>(
                 ::mlir::gpu::Dimension::x);
             Type ty_i32 = self.getBuilder().getIntegerType(32);
             tid = self.create<arith::IndexCastOp>(ty_i32, tid);
             return tid;
           })
      .def("create_get_block_size",
           [](TritonOpBuilder &self) -> Value {
             Value bs = self.create<::mlir::gpu::BlockDimOp>(
                 ::mlir::gpu::Dimension::x);
             Type ty_i32 = self.getBuilder().getIntegerType(32);
             bs = self.create<arith::IndexCastOp>(ty_i32, bs);
             return bs;
           })
      .def("create_simt_exec_region_op",
           [](TritonOpBuilder &self,
              std::vector<Value> &init_args) -> triton::simt::SIMTExecRegionOp {
             return self.create<mlir::triton::simt::SIMTExecRegionOp>(
                 init_args);
           })
      .def("create_block_yield_op",
           [](TritonOpBuilder &self,
              std::vector<Value> &yields) -> triton::simt::BlockYieldOp {
             return self.create<triton::simt::BlockYieldOp>(yields);
           })
      .def("create_extract",
           [](TritonOpBuilder &self, Value src,
              std::vector<Value> &indices) -> Value {
             std::vector<Value> to_index;
             for (size_t i = 0; i < indices.size(); ++i) {
               Value val = indices[i];
               if (!isa<IndexType>(val.getType())) {
                 to_index.push_back(self.create<arith::IndexCastOp>(
                     self.getBuilder().getIndexType(), val));
               } else {
                 to_index.push_back(val);
               }
             }
             Value ret = self.create<tensor::ExtractOp>(src, to_index);
             return ret;
           })
      .def("create_insert",
           [](TritonOpBuilder &self, Value scalar, Value dest,
              std::vector<Value> &indices) -> Value {
             std::vector<Value> to_index;
             for (size_t i = 0; i < indices.size(); ++i) {
               Value val = indices[i];
               if (!isa<IndexType>(val.getType())) {
                 to_index.push_back(self.create<arith::IndexCastOp>(
                     self.getBuilder().getIndexType(), val));
               } else {
                 to_index.push_back(val);
               }
             }
             return self.create<tensor::InsertOp>(scalar, dest, to_index);
           })
      // GPU Ops
      .def("create_laneid",
           [](TritonOpBuilder &self) -> Value {
             Value laneId =
                 self.create<mlir::gpu::LaneIdOp>(/*upperBound=*/nullptr);
             return self.create<arith::IndexCastOp>(
                 self.getBuilder().getI32Type(), laneId);
           })
      .def("create_warp_shuffle",
           [](TritonOpBuilder &self, Value &value, Value &offset, Value &width,
              const int mode) -> Value {
             mlir::gpu::ShuffleMode shflMode =
                 static_cast<mlir::gpu::ShuffleMode>(mode);
             auto shflOp = self.create<mlir::gpu::ShuffleOp>(value, offset,
                                                             width, shflMode);
             return shflOp.getShuffleResult();
           })

      // Distributed Ops
      .def("create_distributed_wait",
           [](TritonOpBuilder &self, Value &barrierPtrs, Value &numBarriers,
              Value &waitValue, MemSyncScope scope, MemSemantic semantic,
              Type &type) -> Value {
             return self.create<mlir::triton::distributed::WaitOp>(
                 type, barrierPtrs, numBarriers, waitValue, scope, semantic);
           })
      .def("create_distributed_consume_token",
           [](TritonOpBuilder &self, Value &input, Value &token) -> Value {
             return self.create<mlir::triton::distributed::ConsumeTokenOp>(
                 input, token);
           })
      .def("create_get_rank",
           [](TritonOpBuilder &self, Value axis) -> Value {
             return self.create<mlir::triton::distributed::GetRankOp>(axis);
           })
      .def("create_get_num_ranks",
           [](TritonOpBuilder &self, Value axis) -> Value {
             return self.create<mlir::triton::distributed::GetNumRanksOp>(axis);
           })
      .def("create_symm_at",
           [](TritonOpBuilder &self, Value ptr, Value rank) -> Value {
             return self.create<mlir::triton::distributed::SymmAtOp>(
                 ptr.getType(), ptr, rank);
           })
      .def("create_notify",
           [](TritonOpBuilder &self, Value ptr, Value signal, Value rank,
              distributed::SignalOp sigOp,
              distributed::CommScope commScope) -> void {
             self.create<mlir::triton::distributed::NotifyOp>(ptr, signal, rank,
                                                              sigOp, commScope);
           })
      .def("create_extern_call",
           [](TritonOpBuilder &self, const std::string &libName,
              const std::string &libPath, const std::string &symbol,
              std::vector<Value> &argList, const std::vector<Type> &retTypes,
              bool isPure) -> OpState {
             return self.create<mlir::triton::distributed::ExternCallOp>(
                 retTypes, argList, libName, libPath, symbol, isPure);
           });
}

void init_triton_distributed_env_vars(py::module &m) {
  m.def("get_cache_invalidating_env_vars",
        []() -> std::map<std::string, std::string> {
          std::map<std::string, std::string> ret;
          for (const auto &envVar : CACHE_INVALIDATING_ENV_VARS) {
            auto strVal = triton::tools::getStrEnv(envVar);
            if (strVal.empty())
              continue;
            auto boolV = triton::tools::isEnvValueBool(strVal);
            if (boolV.has_value())
              ret[envVar] = boolV.value() ? "true" : "false";
            else
              ret[envVar] = strVal;
          }
          return ret;
        });
}
