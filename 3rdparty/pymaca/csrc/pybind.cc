#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "maca_runtime_wrapper.cuh"

namespace py = pybind11;

PYBIND11_MODULE(maca, m) {
    py::module_ maca_m = m.def_submodule("maca");
    py::module_ macart_m = m.def_submodule("macart");
    py::enum_<mcMemcpyKind>(macart_m, "mcMemcpyKind")
        .value("mcMemcpyHostToHost", mcMemcpyKind::mcMemcpyHostToHost)
        .value("mcMemcpyHostToDevice", mcMemcpyKind::mcMemcpyHostToDevice)
        .value("mcMemcpyDeviceToHost", mcMemcpyKind::mcMemcpyDeviceToHost)
        .value("mcMemcpyDeviceToDevice", mcMemcpyKind::mcMemcpyDeviceToDevice)
        .value("mcMemcpyDefault", mcMemcpyKind::mcMemcpyDefault)
        .export_values();

    macart_m.def("mcMemcpyAsync", &mcMemcpyAsyncWrapper, "A wrapper for mcMemcpyAsync");
    py::enum_<mcError_t>(macart_m, "mcError_t")
        .value("mcSuccess", mcError_t::mcSuccess);
    macart_m.def("mcGetErrorString", &mcGetErrorStringWrapper, "A wrapper for mcGetErrorString");

    py::enum_<mcStreamWriteValue_flags>(maca_m, "mcStreamWriteValue_flags")
        .value("MC_STREAM_WRITE_VALUE_DEFAULT", mcStreamWriteValue_flags::MC_STREAM_WRITE_VALUE_DEFAULT )
        .value("MC_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER", mcStreamWriteValue_flags::MC_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER )
        .export_values();
    maca_m.def("mcStreamWriteValue32", &mcStreamWriteValue32Wrapper, "A wrapper for mcStreamWriteValue32");
    maca_m.def("mcStreamWriteValue64", &mcStreamWriteValue64Wrapper, "A wrapper for mcStreamWriteValue64");

    py::enum_<mcStreamWaitValue_flags>(maca_m, "mcStreamWaitValue_flags")
        .value("MC_STREAM_WAIT_VALUE_GEQ", mcStreamWaitValue_flags::MC_STREAM_WAIT_VALUE_GEQ)
        .value("MC_STREAM_WAIT_VALUE_EQ", mcStreamWaitValue_flags::MC_STREAM_WAIT_VALUE_EQ)
        .value("MC_STREAM_WAIT_VALUE_AND", mcStreamWaitValue_flags::MC_STREAM_WAIT_VALUE_AND)
        .value("MC_STREAM_WAIT_VALUE_NOR", mcStreamWaitValue_flags::MC_STREAM_WAIT_VALUE_NOR)
        .value("MC_STREAM_WAIT_VALUE_FLUSH", mcStreamWaitValue_flags::MC_STREAM_WAIT_VALUE_FLUSH)
        .export_values();
    maca_m.def("mcStreamWaitValue32", &mcStreamWaitValue32Wrapper, "A wrapper for mcStreamWaitValue32");
    maca_m.def("mcStreamWaitValue64", &mcStreamWaitValue64Wrapper, "A wrapper for mcStreamWaitValue64");
    maca_m.def("mcGetErrorName", &mcGetErrorNameWrapper, "A wrapper for mcGetErrorName");
    py::enum_<mcParallelCopyEngine>(macart_m, "mcParallelCopyEngine")
        .value("ParallelCopyEngine0", mcParallelCopyEngine::ParallelCopyEngine0)
        .value("ParallelCopyEngine1", mcParallelCopyEngine::ParallelCopyEngine1)
        .value("ParallelCopyEngine2", mcParallelCopyEngine::ParallelCopyEngine2)
        .value("ParallelCopyEngine3", mcParallelCopyEngine::ParallelCopyEngine3)
        .value("ParallelCopyEngine4", mcParallelCopyEngine::ParallelCopyEngine4)
        .value("ParallelCopyEngineDefault", mcParallelCopyEngine::ParallelCopyEngineDefault)
        .export_values();
    macart_m.def("mcExtBatchCopyFlagAndWait", &mcExtBatchCopyFlagAndWaitWrapper, "A wrapper for mcExtBatchCopyFlagAndWait");
}
