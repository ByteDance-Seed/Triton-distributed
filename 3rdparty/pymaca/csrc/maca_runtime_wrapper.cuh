#include "maca.h"
#include "mc_runtime.h"
#include <cstdint>
#include <tuple>
#include <vector>

std::tuple<mcError_t> mcMemcpyAsyncWrapper(int64_t dst, const int64_t src,
                                           size_t count, mcMemcpyKind kind,
                                           int64_t stream) {
  auto result = mcMemcpyAsync(reinterpret_cast<void *>(dst),
                              reinterpret_cast<void *>(src), count, kind,
                              reinterpret_cast<mcStream_t>(stream));
  return std::make_tuple(result);
}

std::tuple<mcError_t>
mcStreamWriteValue32Wrapper(int64_t stream, int64_t addr, uint32_t value,
                            mcStreamWriteValue_flags flags) {
  auto result =
      mcStreamWriteValue32(reinterpret_cast<mcStream_t>(stream),
                           reinterpret_cast<void *>(addr), value, flags);
  return std::make_tuple(result);
}

std::tuple<mcError_t>
mcStreamWriteValue64Wrapper(int64_t stream, int64_t addr, uint64_t value,
                            mcStreamWriteValue_flags flags) {
  auto result =
      mcStreamWriteValue64(reinterpret_cast<mcStream_t>(stream),
                           reinterpret_cast<void *>(addr), value, flags);
  return std::make_tuple(result);
}

std::tuple<mcError_t>
mcStreamWaitValue32Wrapper(int64_t stream, int64_t addr, uint32_t value,
                           mcStreamWaitValue_flags flags) {
  auto result =
      mcStreamWaitValue32(reinterpret_cast<mcStream_t>(stream),
                          reinterpret_cast<void *>(addr), value, flags);
  return std::make_tuple(result);
}

std::tuple<mcError_t>
mcStreamWaitValue64Wrapper(int64_t stream, int64_t addr, uint64_t value,
                           mcStreamWaitValue_flags flags) {
  auto result =
      mcStreamWaitValue64(reinterpret_cast<mcStream_t>(stream),
                          reinterpret_cast<void *>(addr), value, flags);
  return std::make_tuple(result);
}

const char *mcGetErrorNameWrapper(mcError_t error) {
  return mcGetErrorName(error);
}

const char *mcGetErrorStringWrapper(mcError_t error) {
  return mcGetErrorString(error);
}

std::tuple<mcError_t> mcExtBatchCopyFlagAndWaitWrapper(
    std::vector<int64_t> &dst, std::vector<int64_t> &src,
    std::vector<mcParallelCopyEngine> &engine, std::vector<size_t> &count,
    std::vector<int64_t> &write_flag, std::vector<uint64_t> &write_value,
    std::vector<int64_t> &wait_flag, std::vector<uint64_t> &wait_value,
    int64_t stream) {
  size_t reqNum = dst.size();
  std::vector<mcCopyFlag_t> copyBatchArray(reqNum);
  for (int i = 0; i < reqNum; i++) {
    mcCopyFlag_t copyFlag;
    mcExtFlag_t writeFlag;
    mcExtFlag_t waitFlag;
    copyFlag.dst = reinterpret_cast<void *>(dst[i]);
    copyFlag.src = reinterpret_cast<void *>(src[i]);
    copyFlag.engine = engine[i];
    copyFlag.count = count[i];
    if (wait_flag.size() > 0) {
      copyFlag.waitNum = 1;
      waitFlag.flag = reinterpret_cast<uint64_t *>(wait_flag[i]);
      waitFlag.value = wait_value[i];
      copyFlag.waits[0] = waitFlag;
    } else {
      copyFlag.waitNum = 0;
    }
    copyFlag.writeNum = 1;
    writeFlag.flag = reinterpret_cast<uint64_t *>(write_flag[i]);
    writeFlag.value = write_value[i];
    copyFlag.writes[0] = writeFlag;
    copyBatchArray[i] = copyFlag;
  }

  auto result =
      mcExtBatchCopyFlagAndWait(copyBatchArray.data(), reqNum, nullptr, 0,
                                reinterpret_cast<mcStream_t>(stream));
  return std::make_tuple(result);
}
