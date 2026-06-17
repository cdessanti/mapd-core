/*
 * Copyright 2022 HEAVY.AI, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "CudaMgr/CudaMgr.h"

#include <algorithm>
#include <iostream>

#include <random>
#include <stdexcept>

#include <tbb/parallel_for.h>
#include <boost/filesystem.hpp>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include "Logger/Logger.h"
#include "OSDependent/heavyai_path.h"
#include "QueryEngine/NvidiaKernel.h"
#include "Shared/checked_alloc.h"
#include "Shared/measure.h"
#include "Shared/thread_count.h"
#include "nvcomp.h"
#include "nvcomp/ans.h"
#include "nvcomp/bitcomp.h"
#include "nvcomp/gdeflate.h"
#include "nvcomp/lz4.h"
#include "nvcomp/native/gdeflate_cpu.h"

bool g_enable_gpu_dynamic_smem{true};
bool g_enable_d2h_compression{true};
bool g_enable_d2h_comp_double_buffering{false};
size_t g_threshold_d2h_compression{1 << 20};
size_t g_comp_max_estimate_buffer_size{1 << 27};
size_t g_comp_min_estimate_buffer_size{1 << 24};
size_t g_comp_preferred_num_chunks{16};
extern bool g_enable_compilation_optimization;

namespace CudaMgr_Namespace {

boost::filesystem::path get_gpu_rt_path() {
  boost::filesystem::path gpu_rt_path{heavyai::get_root_abs_path()};
  gpu_rt_path /= "QueryEngine";
  if (g_enable_compilation_optimization) {
    gpu_rt_path /= "cuda_mapd_rt_minimal.fatbin";
  } else {
    gpu_rt_path /= "cuda_mapd_rt.fatbin";
  }
  if (!boost::filesystem::exists(gpu_rt_path)) {
    throw std::runtime_error("HeavyDB GPU runtime library not found at " +
                             gpu_rt_path.string());
  }
  return gpu_rt_path;
}

void CudaMgr::loadGpuRtFatbinIntoCache() {
  if (gpu_rt_fatbin_.size() == 0) {
    FILE* fp = fopen(get_gpu_rt_path().c_str(), "rb");

    if (fp == NULL)
      return;

    if (fseek(fp, 0, SEEK_END) < 0) {
      fclose(fp);
      return;
    }

    size_t file_size = ftell(fp);
    gpu_rt_fatbin_.resize(file_size);
    fseek(fp, 0, 0);
    auto read_size = fread(gpu_rt_fatbin_.data(), sizeof(char), file_size, fp);
    if (file_size != read_size) {
      gpu_rt_fatbin_.resize(0);
    }
    fclose(fp);
  }
}

CudaMgr::CudaMgr(const int num_gpus, const int start_gpu)
    : start_gpu_(start_gpu)
    , min_shared_memory_per_block_for_all_devices(0)
    , min_num_mps_for_all_devices(0)
    , log_memory_activity_(false)
    , device_memory_allocation_map_{std::make_unique<DeviceMemoryAllocationMap>()} {
  checkError(cuInit(0));
  checkError(cuDeviceGetCount(&device_count_));

  if (num_gpus > 0) {  // numGpus <= 0 will just use number of gpus found
    device_count_ = std::min(device_count_, num_gpus);
  } else {
    // if we are using all gpus we cannot start on a gpu other than 0
    CHECK_EQ(start_gpu_, 0);
  }

  // Fill device properties and initialize group/contexts
  fillDeviceProperties();
  initDeviceGroup();
  createDeviceContexts();

  // warm up the GPU JIT
  LOG(INFO) << "Warming up the GPU JIT Compiler... (this may take several seconds)";
  setContext(0);
  //loadGpuRtFatbinIntoCache();

  nvidia_jit_warmup();
  LOG(INFO) << "GPU JIT Compiler initialized.";

  jump_buffer_transfer_mgr_ =
      std::make_unique<JumpBufferTransferMgr>(device_count_, device_contexts_);

  LOG(INFO) << "Calculating the performance statistics of the system... (this may take "
               "several seconds)";
  fillDevicePerfProperties();
  logDeviceProperties();
}

void CudaMgr::initDeviceGroup() {
  CHECK(device_properties_initialized_);
  for (int device_id = 0; device_id < device_count_; device_id++) {
    device_group_.push_back(
        {device_id, device_id + start_gpu_, device_properties_[device_id].uuid});
  }
}

CudaMgr::~CudaMgr() {
  try {
    // We don't want to remove the cudaMgr before all other processes have cleaned up.
    // This should be enforced by the lifetime policies, but take this lock to be safe.
    std::lock_guard<std::mutex> device_lock(device_mutex_);
    jump_buffer_transfer_mgr_.reset();

    synchronizeDevices();

    CHECK(getDeviceMemoryAllocationMap().mapEmpty());
    device_memory_allocation_map_ = nullptr;

    for (int d = 0; d < device_count_; ++d) {
      checkError(cuCtxDestroy(device_contexts_[d]));
    }
  } catch (const CudaErrorException& e) {
    if (e.getStatus() == CUDA_ERROR_DEINITIALIZED) {
      // TODO(adb / asuhan): Verify cuModuleUnload removes the context
      return;
    }
    LOG(ERROR) << "CUDA Error: " << e.what();
  } catch (const std::runtime_error& e) {
    LOG(ERROR) << "CUDA Error: " << e.what();
  }
}

size_t CudaMgr::computePaddedBufferSize(size_t buf_size, size_t granularity) const {
  return (((buf_size + (granularity - 1)) / granularity) * granularity);
}

size_t CudaMgr::getGranularity(const int device_num) const {
  CUmemAllocationProp allocation_prop{};
  allocation_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  allocation_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  allocation_prop.location.id = device_num;
  size_t granularity{};
  checkError(cuMemGetAllocationGranularity(
      &granularity, &allocation_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  return granularity;
}

void CudaMgr::synchronizeDevices() const {
  for (int d = 0; d < device_count_; ++d) {
    setContext(d);
    checkError(cuCtxSynchronize());
  }
}

bitcompAlgorithm_t CudaMgr::getNVCompCompressionAlgo(CudaCompression algo) {
  // CHECK_NE(algo, CudaCompression::kNoCompression);
  switch (algo) {
    case CudaCompression::kBitcompNativeSparse:
      return BITCOMP_SPARSE_ALGO;
    case CudaCompression::kBitcompNativeDefault:
      return BITCOMP_DEFAULT_ALGO;
      break;
    default:
      UNREACHABLE();
      break;
  }
  return (bitcompAlgorithm_t)-1;
}

size_t inline returnTPForSize(size_t size) {
  constexpr size_t test_buffer_size = 31;
  auto exp = floor(log2(size)) > 31 ? 31 : floor(log2(size));

  return test_buffer_size - exp > 7 ? 7 : test_buffer_size - exp;
}

std::string inline returnAlgoUsed(bitcompAlgorithm_t algo) {
  switch (algo) {
    case BITCOMP_SPARSE_ALGO:
      return "Bitcomp Sparse";
      break;
    case BITCOMP_DEFAULT_ALGO:
      return "Bitcomp Default";
      break;
    default:
      return "No compression";
  }
}

size_t inline returnChunkSize(size_t num_bytes) {
  return num_bytes < g_comp_min_estimate_buffer_size
             ? g_comp_min_estimate_buffer_size
             : std::max(std::min((size_t)num_bytes / g_comp_preferred_num_chunks,
                                 g_comp_max_estimate_buffer_size),
                        g_comp_min_estimate_buffer_size);
}

bitcompAlgorithm_t CudaMgr::getBestCompressionAlgo(size_t num_bytes,
                                                   const int device_num,
                                                   size_t estimated_num_bytes_compressed,
                                                   bitcompAlgorithm_t algo) {
  auto index = returnTPForSize(num_bytes);

  return device_properties_[device_num].cpuDecompressionBandwidthBs[index] >
                 device_properties_[device_num].cpuDecompressionSparseBandwidthBs[index]
             ? BITCOMP_DEFAULT_ALGO
             : BITCOMP_SPARSE_ALGO;
}

size_t CudaMgr::evaluateCompression(size_t num_bytes,
                                    const int device_num,
                                    size_t estimated_num_bytes_compressed,
                                    bitcompAlgorithm_t algo) {
  if (num_bytes < g_threshold_d2h_compression) {
    return false;
  }
  auto index = returnTPForSize(num_bytes);
  size_t gpu_max_comp_tp_ms =
      algo == BITCOMP_SPARSE_ALGO
          ? device_properties_[device_num].gpuCompressionSparseBandwidthBs[index]
          : device_properties_[device_num].gpuCompressionBandwidthBs[index];
  size_t cpu_max_decomp_tp_ms =
      algo == BITCOMP_SPARSE_ALGO
          ? device_properties_[device_num].cpuDecompressionSparseBandwidthBs[index]
          : device_properties_[device_num].cpuDecompressionBandwidthBs[index];
  size_t gpu_max_transfer_tp_nocomp_ms =
      jump_buffer_transfer_mgr_->shouldUseForDeviceToHostTransfer(num_bytes)
          ? device_properties_[device_num].deviceToHostBandwidthJumpBufferBs[index]
          : device_properties_[device_num].deviceToHostBandwidthNotPinnedBs[index];
  size_t gpu_max_transfer_tp_ms =
      jump_buffer_transfer_mgr_->getDecompJumpBufferMemorySize(device_num) > num_bytes
          ? device_properties_[device_num].deviceToHostBandwidthPinnedBs[index]
          : device_properties_[device_num].deviceToHostBandwidthNotPinnedBs[index];
  auto estimated_transfer_time = num_bytes * 1000 / gpu_max_transfer_tp_nocomp_ms;
  auto estimated_decompression_time = num_bytes * 1000 / cpu_max_decomp_tp_ms;
  auto gpu_compression_time = num_bytes * 1000 / gpu_max_comp_tp_ms;

  if (!estimated_num_bytes_compressed) {
    VLOG(2) << "Estimated transfer time no compression " << estimated_transfer_time
            << ", compression " << estimated_decompression_time
            << ", gpu time for compression " << gpu_compression_time
            << " decompression jumpbuffer memory size "
            << jump_buffer_transfer_mgr_->getDecompJumpBufferMemorySize(device_num)
            << " estimate chunk size " << returnChunkSize(num_bytes)
            << " estimated transfer time: " << estimated_transfer_time
            << " us, Estimated decompression + compression time: "
            << (estimated_decompression_time + gpu_compression_time) << " us.";

    return estimated_transfer_time >
                   (estimated_decompression_time + gpu_compression_time * 2)
               ? bitcompMaxBuflen(returnChunkSize(num_bytes))
               : false;
  }
  auto estimated_transfer_time_compressed =
      estimated_num_bytes_compressed * 1000 / gpu_max_transfer_tp_ms;
  auto estimated_total_time_compressed = estimated_decompression_time +
                                         estimated_transfer_time_compressed +
                                         gpu_compression_time * 2;

  VLOG(2) << "Estimation, Buffer size: " << format_num_bytes(num_bytes)
          << " Compression / Decompression time: "
          << format_microseconds(
                 (estimated_decompression_time + (gpu_compression_time * 2)))
          << ", Ratio: " << num_bytes * 1.0 / estimated_num_bytes_compressed
          << " Algo: " << returnAlgoUsed(algo)
          << ", Trasfer time: " << format_microseconds(estimated_transfer_time_compressed)
          << ", Total time: " << format_microseconds(estimated_total_time_compressed)
          << ", Uncompressed Total time " << format_microseconds(estimated_transfer_time)
          << (estimated_transfer_time > estimated_total_time_compressed
                  ? ". Using compression because an estimated speed-up of " +
                        format_microseconds(estimated_transfer_time -
                                            estimated_total_time_compressed)
                  : ". Not using compression because an estimated slow-down of " +
                        format_microseconds(estimated_total_time_compressed -
                                            estimated_transfer_time));

  return estimated_transfer_time > estimated_total_time_compressed;
}

size_t CudaMgr::estimateCompressedBufferSize(size_t num_bytes,
                                             const int8_t* device_ptr,
                                             size_t max_chunk_size,
                                             int8_t* device_estimate_ptr,
                                             bitcompAlgorithm_t algo) {
  bitcompHandle_t handle;
  // estimate the memory needed on gpu to compress the entire chunk
  size_t estimate_chunk_size = returnChunkSize(num_bytes);
  size_t num_chunks = (size_t)num_bytes / estimate_chunk_size;
  size_t last_chunk_size = num_bytes - (estimate_chunk_size * num_chunks);
  bitcompCreatePlan(
      &handle, estimate_chunk_size, BITCOMP_UNSIGNED_8BIT, BITCOMP_LOSSLESS, algo);
  size_t num_bytes_compressed_e;
  size_t estimated_num_bytes_compressed{0};
  size_t estimated_num_bytes_compressed_first_buffer{0};
  for (size_t i = 0; i < num_chunks; i += 1) {
    bitcompCompressLossless(
        handle, device_ptr + i * estimate_chunk_size, device_estimate_ptr);
    cudaDeviceSynchronize();
    bitcompGetCompressedSize(device_estimate_ptr, &num_bytes_compressed_e);
    cudaDeviceSynchronize();
    estimated_num_bytes_compressed += num_bytes_compressed_e;
    if (i + 1 == num_chunks / 2) {
      estimated_num_bytes_compressed_first_buffer = estimated_num_bytes_compressed;
    }
  }
  if (last_chunk_size) {
    bitcompDestroyPlan(handle);
    bitcompCreatePlan(
        &handle, last_chunk_size, BITCOMP_UNSIGNED_8BIT, BITCOMP_LOSSLESS, algo);
    bitcompCompressLossless(
        handle, device_ptr + num_chunks * estimate_chunk_size, device_estimate_ptr);
    cudaDeviceSynchronize();
    bitcompGetCompressedSize(device_estimate_ptr, &num_bytes_compressed_e);
    cudaDeviceSynchronize();
    estimated_num_bytes_compressed += num_bytes_compressed_e;
  }
  bitcompDestroyPlan(handle);
  // Play safe and add 5% to the estimate
  if (g_enable_d2h_comp_double_buffering) {
    return computePaddedBufferSize(
        std::max(
            estimated_num_bytes_compressed - estimated_num_bytes_compressed_first_buffer,
            estimated_num_bytes_compressed_first_buffer) *
            210 / 100,
        16);
  }
  return computePaddedBufferSize(estimated_num_bytes_compressed * 105 / 100, 8);
}

void CudaMgr::copyDeviceToHostCompressed(int8_t* host_ptr,
                                         const int8_t* device_ptr,
                                         const size_t num_bytes,
                                         std::optional<std::string_view> tag,
                                         CUstream cuda_stream,
                                         bitcompAlgorithm_t algo,
                                         int8_t* device_temp_ptr,
                                         const size_t device_temp_num_bytes,
                                         bool decompress) {
  CHECK_NE(device_temp_ptr, nullptr);
  size_t estimated_num_bytes_compressed = device_temp_num_bytes;
  if (!g_enable_d2h_compression) {
    copyDeviceToHost(host_ptr, device_ptr, num_bytes, tag, cuda_stream);
    return;
  }
  auto const cu_device_ptr = reinterpret_cast<CUdeviceptr>(device_ptr);
  int const device_num = getDeviceNumFromDevicePtr(cu_device_ptr, num_bytes);

  if (!evaluateCompression(num_bytes, device_num, 0, algo)) {
    copyDeviceToHost(host_ptr, device_ptr, num_bytes, tag, cuda_stream);
    return;
  }
  bitcompHandle_t handle;
  // estimate the memory needed if not provided
  if (!estimated_num_bytes_compressed) {
    size_t estimate_chunk_size = std::min((size_t)num_bytes / 16, (size_t)num_bytes / 16);
    size_t max_out_bytes_e = bitcompMaxBuflen(estimate_chunk_size);
    estimated_num_bytes_compressed = estimateCompressedBufferSize(
        num_bytes, device_ptr, max_out_bytes_e, device_temp_ptr, algo);
  }
  // if (!evaluateCompression(num_bytes, device_num, estimated_num_bytes_compressed,
  // algo)) {
  //   copyDeviceToHost(host_ptr, device_ptr, num_bytes, tag, cuda_stream);
  //   return;
  // }

  auto num_buffers = (jump_buffer_transfer_mgr_->getDecompJumpBufferMemorySize(
                          device_num) >= estimated_num_bytes_compressed &&
                      g_enable_d2h_comp_double_buffering)
                         ? 2
                         : 1;
  auto num_bytes_buffer = num_bytes / num_buffers;

  int8_t* temp_cpu{nullptr};
  bool decomp_buffer_isused = jump_buffer_transfer_mgr_->getDecompJumpBufferMemorySize(
                                  device_num) >= estimated_num_bytes_compressed;

  if (decomp_buffer_isused) {
    temp_cpu = jump_buffer_transfer_mgr_->getDecompJumpBufferMemoryLocation(device_num);
  }
  if (temp_cpu == nullptr) {
    decomp_buffer_isused = false;
    temp_cpu = reinterpret_cast<int8_t*>(checked_malloc(estimated_num_bytes_compressed));
  }

  auto deleter = [this, decomp_buffer_isused, device_num](void* ptr) {
    if (decomp_buffer_isused) {
      this->jump_buffer_transfer_mgr_->freeDecompJumpBufferMemory(device_num);
    } else {
      free(ptr);
    }
  };

  std::unique_ptr<void, decltype(deleter)> guard(temp_cpu, deleter);

  auto temp_buffer_size = estimated_num_bytes_compressed / num_buffers;
  auto timer = timer_start();

  bitcompCheck(bitcompCreatePlan(
      &handle, num_bytes_buffer, BITCOMP_UNSIGNED_8BIT, BITCOMP_LOSSLESS, algo));
  bitcompCheck(bitcompCompressLossless(handle, device_ptr, device_temp_ptr));
  size_t num_bytes_compressed, total_compressed;

  bitcompGetCompressedSize(device_temp_ptr, &num_bytes_compressed);
  total_compressed = num_bytes_compressed;

  int64_t compression_time = timer_stop_microseconds(timer);
  int64_t decompression_time{0};
  int64_t transfer_time{0};

  if (num_buffers != 1) {
    checkError(cuMemcpyDtoHAsync(temp_cpu,
                                 reinterpret_cast<CUdeviceptr>(device_temp_ptr),
                                 num_bytes_compressed,
                                 cuda_stream));
    bitcompCheck(bitcompDestroyPlan(handle));
    bitcompCheck(bitcompCreatePlan(&handle,
                                   num_bytes - num_bytes_buffer,
                                   BITCOMP_UNSIGNED_8BIT,
                                   BITCOMP_LOSSLESS,
                                   algo));

    bitcompCheck(bitcompCompressLossless(
        handle, device_ptr + num_bytes_buffer, device_temp_ptr + temp_buffer_size));
    bitcompGetCompressedSize(device_temp_ptr + temp_buffer_size, &num_bytes_compressed);
    total_compressed += num_bytes_compressed;
    transfer_time = timer_stop_microseconds(timer) - compression_time;
    checkError(cuMemcpyDtoHAsync(
        temp_cpu + temp_buffer_size,
        reinterpret_cast<CUdeviceptr>(device_temp_ptr + temp_buffer_size),
        num_bytes_compressed,
        cuda_stream));
  } else {
    checkError(cuMemcpyDtoH(
        temp_cpu, reinterpret_cast<CUdeviceptr>(device_temp_ptr), num_bytes_compressed));
  }
  const size_t chunk_count = cpu_threads() / 2;
  size_t chunk_size = computePaddedBufferSize(num_bytes_buffer / chunk_count,16);

  transfer_time = timer_stop_microseconds(timer) - compression_time;
  tbb::task_arena limited_arena(chunk_count);
  limited_arena.execute([&] {
    tbb::parallel_for(size_t(0), chunk_count, [=](size_t i) {
      auto chunk_size_thread = i+1 == chunk_count ? num_bytes_buffer - (chunk_size * i) : chunk_size;
      bitcompCheck(bitcompHostPartialUncompress(
          handle, temp_cpu, host_ptr + (chunk_size * i), chunk_size * i, chunk_size_thread));
      ;
    });
  });
  decompression_time = timer_stop_microseconds(timer) - compression_time - transfer_time;
  if (num_buffers != 1) {
    cuStreamSynchronize(cuda_stream);
    transfer_time =
        timer_stop_microseconds(timer) - compression_time - decompression_time;
    limited_arena.execute([&] {
      tbb::parallel_for(size_t(0), chunk_count, [=](size_t i) {
        auto chunk_size_thread = i+1 == chunk_count ? num_bytes_buffer - (chunk_size * i) : chunk_size;
        bitcompCheck(
            bitcompHostPartialUncompress(handle,
                                         temp_cpu + temp_buffer_size,
                                         host_ptr + (chunk_size * i) + num_bytes_buffer,
                                         chunk_size * i,
                                         chunk_size_thread));
        ;
      });
    });
    decompression_time =
        timer_stop_microseconds(timer) - compression_time - transfer_time;
  }

  const size_t decomp_tp_sec = num_bytes / decompression_time;

  updateDevicePerfProperties(device_num,
                             num_bytes,
                             transfer_time,
                             true,
                             decomp_buffer_isused,
                             decompression_time,
                             compression_time,
                             num_bytes_compressed,
                             algo);
  auto total_time = timer_stop_microseconds(timer);
  bitcompCheck(bitcompDestroyPlan(handle));

  VLOG(1) << "CUDA D2H Compressed, tag: " << *tag
          << " Buffer size: " << format_num_bytes(num_bytes)
          << ", GPU Compression time: " << format_microseconds(compression_time)
          << ", Troughtput: " << num_bytes / compression_time / 1000.0
          << " GB/sec, Ratio: " << num_bytes * 1.0 / total_compressed
          << ", Algo: " << returnAlgoUsed(algo)
          << ", Trasfer time: " << format_microseconds(transfer_time)
          << ", CPU Decompression time: " << format_microseconds(decompression_time)
          << ", Troughtput: " << decomp_tp_sec / 1000.0
          << " GB/sec, Total time: " << format_microseconds(total_time)
          << ", Overall Troughtput: " << num_bytes / (total_time) / 1000.0 << " GB/sec.";
};

void CudaMgr::copyHostToDevice(int8_t* device_ptr,
                               const int8_t* host_ptr,
                               const size_t num_bytes,
                               const int device_num,
                               std::optional<std::string_view> tag,
                               CUstream cuda_stream) {
  bool jump_buffer_used{false};
  if (jump_buffer_transfer_mgr_->shouldUseForHostToDeviceTransfer(num_bytes)) {
    jump_buffer_used = jump_buffer_transfer_mgr_->copyHostToDevice(
        device_ptr, host_ptr, num_bytes, device_num, cuda_stream);
  }

  auto timer = timer_start();

  if (!jump_buffer_used) {
    setContext(device_num);
    if (!cuda_stream) {
      checkError(
          cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(device_ptr), host_ptr, num_bytes));
    } else {
      checkError(cuMemcpyHtoDAsync(
          reinterpret_cast<CUdeviceptr>(device_ptr), host_ptr, num_bytes, cuda_stream));
      checkError(cuStreamSynchronize(cuda_stream));
    }
  }

  auto transfer_time = timer_stop_microseconds(timer);
  updateDevicePerfProperties(
      getDeviceNumFromDevicePtr(reinterpret_cast<CUdeviceptr>(device_ptr), num_bytes),
      num_bytes,
      transfer_time,
      false,
      jump_buffer_used);

  if (log_memory_activity_ && tag) {
    VLOG(1) << "CUDA H2D, tag: " << *tag
            << ", host address: " << static_cast<const void*>(host_ptr)
            << ", size: " << num_bytes << ", device: " << device_num
            << ",  device address: " << static_cast<const void*>(device_ptr);
  }
}

int CudaMgr::getDeviceNumFromDevicePtr(CUdeviceptr cu_device_ptr,
                                       const size_t allocated_mem_bytes) {
  std::lock_guard<std::mutex> device_lock(device_mutex_);
  auto const [allocation_base, allocation] =
      getDeviceMemoryAllocationMap().getAllocation(cu_device_ptr);
  CHECK_LE(cu_device_ptr + allocated_mem_bytes, allocation_base + allocation.size);
  CHECK_GE(allocation.device_num, 0);
  return allocation.device_num;
}

void CudaMgr::copyDeviceToHost(int8_t* host_ptr,
                               const int8_t* device_ptr,
                               const size_t num_bytes,
                               std::optional<std::string_view> tag,
                               CUstream cuda_stream) {
  auto const cu_device_ptr = reinterpret_cast<CUdeviceptr>(device_ptr);
  int const device_num = getDeviceNumFromDevicePtr(cu_device_ptr, num_bytes);

  bool jump_buffer_used{false};
  if (jump_buffer_transfer_mgr_->shouldUseForDeviceToHostTransfer(num_bytes)) {
    jump_buffer_used = jump_buffer_transfer_mgr_->copyDeviceToHost(
        host_ptr, device_ptr, num_bytes, device_num, cuda_stream);
  }

  auto timer = timer_start();
  if (!jump_buffer_used) {
    setContext(device_num);
    if (!cuda_stream) {
      checkError(cuMemcpyDtoH(host_ptr, cu_device_ptr, num_bytes));
    } else {
      checkError(cuMemcpyDtoHAsync(host_ptr, cu_device_ptr, num_bytes, cuda_stream));
      checkError(cuStreamSynchronize(cuda_stream));
    }
  }
  auto transfer_time = timer_stop_microseconds(timer);

  updateDevicePerfProperties(
      device_num, num_bytes, transfer_time, true, jump_buffer_used);

  if (log_memory_activity_ && tag) {
    VLOG(1) << "CUDA D2H(" << (jump_buffer_used ? "jump buffer" : "normal")
            << "), tag: " << *tag << ", host address: " << static_cast<void*>(host_ptr)
            << ", size: " << num_bytes << ", device: " << device_num
            << ", device address: " << static_cast<const void*>(device_ptr);
  }
}

void CudaMgr::copyDeviceToDevice(int8_t* dest_ptr,
                                 int8_t* src_ptr,
                                 const size_t num_bytes,
                                 const int dest_device_num,
                                 const int src_device_num,
                                 std::optional<std::string_view> tag,
                                 CUstream cuda_stream) {
  // dest_device_num and src_device_num are the device numbers relative to start_gpu_
  // (real_device_num - start_gpu_)
  if (src_device_num == dest_device_num) {
    setContext(src_device_num);
    if (!cuda_stream) {
      checkError(cuMemcpy(reinterpret_cast<CUdeviceptr>(dest_ptr),
                          reinterpret_cast<CUdeviceptr>(src_ptr),
                          num_bytes));
    } else {
      checkError(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(dest_ptr),
                               reinterpret_cast<CUdeviceptr>(src_ptr),
                               num_bytes,
                               cuda_stream));
      checkError(cuStreamSynchronize(cuda_stream));
    }
  } else {
    if (!cuda_stream) {
      checkError(cuMemcpyPeer(reinterpret_cast<CUdeviceptr>(dest_ptr),
                              device_contexts_[dest_device_num],
                              reinterpret_cast<CUdeviceptr>(src_ptr),
                              device_contexts_[src_device_num],
                              num_bytes));  // will we always have peer?
    } else {
      checkError(cuMemcpyPeerAsync(reinterpret_cast<CUdeviceptr>(dest_ptr),
                                   device_contexts_[dest_device_num],
                                   reinterpret_cast<CUdeviceptr>(src_ptr),
                                   device_contexts_[src_device_num],
                                   num_bytes,
                                   cuda_stream));  // will we always have peer?
      checkError(cuStreamSynchronize(cuda_stream));
    }
  }
  if (log_memory_activity_ && tag) {
    VLOG(1) << "CUDA D2D, tag: " << *tag << ", source device id: " << src_device_num
            << ", destination device id: " << dest_device_num
            << ", source address: " << static_cast<void*>(src_ptr)
            << ", size: " << num_bytes
            << ", destination address: " << static_cast<void*>(dest_ptr);
  }
}

void CudaMgr::loadGpuModuleData(CUmodule* module,
                                const void* image,
                                unsigned int num_options,
                                CUjit_option* options,
                                void** option_vals,
                                const int device_id) const {
  setContext(device_id);
  checkError(cuModuleLoadDataEx(module, image, num_options, options, option_vals));
}

void CudaMgr::unloadGpuModuleData(CUmodule* module, const int device_id) const {
  std::lock_guard<std::mutex> device_lock(device_mutex_);
  CHECK(module);
  setContext(device_id);
  try {
    auto code = cuModuleUnload(*module);
    // If the Cuda driver has already shut down, ignore the resulting errors.
    if (code != CUDA_ERROR_DEINITIALIZED) {
      checkError(code);
    }
  } catch (const std::runtime_error& e) {
    LOG(ERROR) << "CUDA Error: " << e.what();
  }
}

std::vector<CudaMgr::CudaMemoryUsage> CudaMgr::getCudaMemoryUsage() {
  std::vector<CudaMgr::CudaMemoryUsage> m;
  std::lock_guard<std::mutex> map_lock(device_mutex_);
  CUcontext cnow;
  checkError(cuCtxGetCurrent(&cnow));
  for (int device_num = 0; device_num < device_count_; ++device_num) {
    setContext(device_num);
    CudaMemoryUsage usage;
    cuMemGetInfo(&usage.free, &usage.total);
    m.push_back(usage);
  }
  cuCtxSetCurrent(cnow);
  return m;
}

std::string CudaMgr::getCudaMemoryUsageInString() {
  auto const device_mem_status = getCudaMemoryUsage();
  std::ostringstream oss;
  int device_id = 0;
  oss << "{ \"name\": \"GPU Memory Info\", ";
  for (auto& info : device_mem_status) {
    oss << "{\"device_id\": " << device_id++ << ", \"freeMB:\": " << info.free / 1048576.0
        << ", \"totalMB\": " << info.total / 1048576.0 << "} ";
  }
  oss << "}";
  return oss.str();
}

namespace {
bool is_arch_pascal_or_greater(int compute_major) {
  return compute_major >= 6;
}
}  // namespace

void generateSparseBuffer(size_t size,
                          double sparsity,
                          int8_t* buffer,
                          int32_t seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  size_t non_zero_bytes = 0;

  memset(buffer, 0, size);
  for (size_t i = 0; i < size; i += 10) {
    auto uniform = dist(rng);

    if (uniform > sparsity) {
      buffer[i] = rng() % 127;
      non_zero_bytes++;
    } else {
      buffer[i] = 0;
    }
  }
}

size_t CudaMgr::calculateTransferBandwidth(void* host_buffer,
                                           CUdeviceptr device_buffer,
                                           size_t buffer_size,
                                           MemCpyKind mode,
                                           bool use_jump_buffer,
                                           int device_num) {
  cudaStream_t default_stream = 0;
  auto copy_begin = timer_start();
  std::string copy_mode;
  switch (mode) {
    case MemCpyKind::kHostToDevice:
      if (use_jump_buffer)
      // &&
      //   jump_buffer_transfer_mgr_->shouldUseForHostToDeviceTransfer(buffer_size))
      {
        copy_mode = " H2D JumpBuffer ";
        jump_buffer_transfer_mgr_->copyHostToDevice(
            reinterpret_cast<int8_t*>(device_buffer),
            reinterpret_cast<const int8_t*>(host_buffer),
            buffer_size,
            device_num,
            default_stream);
      } else {
        copy_mode = " H2D Memcopy  ";
        checkError(cuMemcpyHtoD(device_buffer, host_buffer, buffer_size));
      }

      break;
    case MemCpyKind::kDeviceToHost:
      if (use_jump_buffer)  //&&
                            // jump_buffer_transfer_mgr_->shouldUseForDeviceToHostTransfer(buffer_size))
      {
        copy_mode = " D2H JumpBuffer ";
        jump_buffer_transfer_mgr_->copyDeviceToHost(
            reinterpret_cast<int8_t*>(host_buffer),
            reinterpret_cast<const int8_t*>(device_buffer),
            buffer_size,
            device_num,
            default_stream);
      } else {
        copy_mode = " D2H MemCpy ";
        checkError(cuMemcpyDtoH(host_buffer, device_buffer, buffer_size));
      }
      break;
    default:
      break;
  }
  auto copy_time_ms = timer_stop(copy_begin);
  VLOG(2) << "Computed transfer bandwidth for buffer of " << format_num_bytes(buffer_size)
          << " using " << copy_mode
          << format_num_bytes(buffer_size * 1000.0 / copy_time_ms) << "/sec";
  return buffer_size / copy_time_ms;
}

size_t CudaMgr::calculateCompressionBandwidth(CUdeviceptr device_buffer,
                                              size_t actual_size,
                                              bitcompAlgorithm_t algo) {
  bitcompHandle_t handle;
  bitcompCheck(bitcompCreatePlan(
      &handle, actual_size, BITCOMP_UNSIGNED_8BIT, BITCOMP_LOSSLESS, algo));
  auto compress_begin = timer_start();
  bitcompCheck(
      bitcompCompressLossless(handle,
                              reinterpret_cast<int8_t*>(device_buffer),
                              reinterpret_cast<int8_t*>(device_buffer) + actual_size));
  cudaDeviceSynchronize();
  size_t num_bytes_compressed;
  bitcompGetCompressedSize(reinterpret_cast<int8_t*>(device_buffer) + actual_size,
                           &num_bytes_compressed);
  cudaDeviceSynchronize();
  bitcompCheck(bitcompDestroyPlan(handle));
  auto compress_time = timer_stop_microseconds(compress_begin);
  VLOG(2) << "Compression " << returnAlgoUsed(algo) << " bytes: " << actual_size
          << " ratio: " << (double)actual_size / num_bytes_compressed
          << " time: " << format_microseconds(compress_time)
          << " tp: " << format_num_bytes(actual_size * 1000000 / (compress_time));
  //  return tp in bytes per ms
  return (actual_size * 1000.0) / compress_time;
}

size_t CudaMgr::calculateCpuDecompressionBandwidth(int8_t* host_temp_buffer,
                                                   size_t buffer_size,
                                                   size_t chunk_count,
                                                   bitcompAlgorithm_t algo) {
  bitcompHandle_t handle;
  bitcompCheck(bitcompCreatePlan(
      &handle, buffer_size, BITCOMP_UNSIGNED_8BIT, BITCOMP_LOSSLESS, algo));

  tbb::task_arena limited_arena(chunk_count);
  size_t chunk_size = buffer_size / chunk_count;
  auto decompress_begin = timer_start();
  int8_t* dest = (int8_t*)checked_malloc(buffer_size);
  limited_arena.execute([&] {
    tbb::parallel_for(size_t(0), chunk_count, [=](size_t i) {
      bitcompCheck(bitcompHostPartialUncompress(
          handle,
          host_temp_buffer,
          dest,  //(host_temp_buffer + buffer_size) + (chunk_size * i),
          chunk_size * i,
          chunk_size));
    });
  });
  free(dest);
  auto decompress_time = timer_stop_microseconds(decompress_begin);
  bitcompCheck(bitcompDestroyPlan(handle));
  VLOG(2) << "Computed " << returnAlgoUsed(algo) << " decompression bandwidth using "
          << chunk_count << " thread" << (chunk_count > 1 ? "s" : "") << ": "
          << (buffer_size * 1000.0) / decompress_time << ". Buffer size: " << buffer_size
          << " time: " << format_microseconds(decompress_time)
          << " tp: " << format_num_bytes((buffer_size * 1000000) / decompress_time);
  return (buffer_size * 1000.0) / decompress_time;
}

void CudaMgr::updateDevicePerfProperties(int device_num,
                                         size_t buffer_size,
                                         size_t transfer_time,
                                         bool device_to_host,
                                         bool use_jump_buffer,
                                         size_t cpu_decompression_time,
                                         size_t gpu_compression_time,
                                         size_t num_bytes_compressed,
                                         bitcompAlgorithm_t algo) {
  const size_t index = returnTPForSize(buffer_size);
  const size_t bytes_transferred =
      num_bytes_compressed ? num_bytes_compressed : buffer_size;
  const size_t transfer_bandwidth = bytes_transferred * 1000.0 / transfer_time;
  const size_t decompression_bandwidth =
      cpu_decompression_time != 0 ? buffer_size * 1000.0 / cpu_decompression_time : cpu_decompression_time;
  const size_t compression_bandwidth =
      gpu_compression_time != 0 ? buffer_size * 1000.0 / gpu_compression_time : gpu_compression_time;
  if (device_to_host) {
    if (use_jump_buffer && !gpu_compression_time) {
      device_properties_[device_num].deviceToHostBandwidthJumpBufferBs[index] =
          (device_properties_[device_num].deviceToHostBandwidthJumpBufferBs[index] +
           transfer_bandwidth) /
          2;
    } else {
      if (use_jump_buffer) {
        device_properties_[device_num].deviceToHostBandwidthPinnedBs[index] =
            (device_properties_[device_num].deviceToHostBandwidthPinnedBs[index] +
             transfer_bandwidth) /
            2;
      } else {
        device_properties_[device_num].deviceToHostBandwidthNotPinnedBs[index] =
            (device_properties_[device_num].deviceToHostBandwidthNotPinnedBs[index] +
             transfer_bandwidth) /
            2;
      }
    }
    if (decompression_bandwidth) {
      CHECK(algo == BITCOMP_DEFAULT_ALGO || BITCOMP_SPARSE_ALGO);
      if (algo == BITCOMP_SPARSE_ALGO) {
        device_properties_[device_num].cpuDecompressionSparseBandwidthBs[index] =
            (device_properties_[device_num].cpuDecompressionSparseBandwidthBs[index] +
             decompression_bandwidth) /
            2;
      } else {
        device_properties_[device_num].cpuDecompressionBandwidthBs[index] =
            (device_properties_[device_num].cpuDecompressionBandwidthBs[index] +
             decompression_bandwidth) /
            2;
      }
    }
    if (compression_bandwidth) {
      CHECK(algo == BITCOMP_DEFAULT_ALGO || BITCOMP_SPARSE_ALGO);
      if (algo == BITCOMP_SPARSE_ALGO) {
        device_properties_[device_num].gpuCompressionSparseBandwidthBs.push_back(
            buffer_size / gpu_compression_time);
      } else {
        device_properties_[device_num].gpuCompressionBandwidthBs.push_back(
            buffer_size / gpu_compression_time);
      }
    }
  } else {
    if (use_jump_buffer) {
      device_properties_[device_num].hostToDeviceBandwidthJumpBufferBs[index] =
          (device_properties_[device_num].hostToDeviceBandwidthJumpBufferBs[index] +
           transfer_bandwidth) /
          2;
    } else {
      device_properties_[device_num].hostToDeviceBandwidthNotPinnedBs[index] =
          (device_properties_[device_num].hostToDeviceBandwidthNotPinnedBs[index] +
           transfer_bandwidth) /
          2;
    }
  }
  VLOG(2) << "Index " << index << " Decompression bw for sparse "
          << device_properties_[device_num].cpuDecompressionSparseBandwidthBs[index]
          << " Decompression bw default "
          << device_properties_[device_num].cpuDecompressionBandwidthBs[index];
}

void CudaMgr::fillDevicePerfProperties() {
  constexpr size_t test_buffer_size = ((size_t)1 << 31);
  constexpr size_t overhead_buffer_size = ((size_t)1 << 16);
  void* host_buffer;
  void* host_buffer_m;
  CudaMgr::setContext(0);
  checkError(cuMemHostAlloc(&host_buffer, test_buffer_size, CU_MEMHOSTALLOC_PORTABLE));
  host_buffer_m = checked_malloc(test_buffer_size);
  for (int device_num = 0; device_num < device_count_; ++device_num) {
    CudaMgr::setContext(device_num);
    CUdeviceptr device_buffer;
    checkError(cuMemAlloc(&device_buffer, test_buffer_size));

    // size_t calculated_bandwith = 0;

    for (size_t i = 1; i < 256; i <<= 1) {
      auto actual_bandwidth = calculateTransferBandwidth(host_buffer,
                                                         device_buffer,
                                                         test_buffer_size / i,
                                                         MemCpyKind::kDeviceToHost,
                                                         true /* use_jump_buffer */,
                                                         device_num);
      device_properties_[device_num].deviceToHostBandwidthJumpBufferBs.emplace_back(
          actual_bandwidth);
      actual_bandwidth = calculateTransferBandwidth(host_buffer,
                                                    device_buffer,
                                                    test_buffer_size / i,
                                                    MemCpyKind::kHostToDevice,
                                                    true /* use_jump_buffer */,
                                                    device_num);
      device_properties_[device_num].hostToDeviceBandwidthJumpBufferBs.emplace_back(
          actual_bandwidth);
      actual_bandwidth = calculateTransferBandwidth(host_buffer,
                                                    device_buffer,
                                                    test_buffer_size / i,
                                                    MemCpyKind::kDeviceToHost,
                                                    false /* use_jump_buffer */,
                                                    device_num);
      device_properties_[device_num].deviceToHostBandwidthPinnedBs.emplace_back(
          actual_bandwidth);
      actual_bandwidth = calculateTransferBandwidth(host_buffer,
                                                    device_buffer,
                                                    test_buffer_size / i,
                                                    MemCpyKind::kHostToDevice,
                                                    false /* use_jump_buffer */,
                                                    device_num);
      device_properties_[device_num].hostToDeviceBandwidthPinnedBs.emplace_back(
          actual_bandwidth);
    }

    checkError(cuMemFreeHost(host_buffer));
    // host_buffer = checked_malloc(test_buffer_size);
    for (size_t i = 1; i < 256; i <<= 1) {
      host_buffer = checked_malloc(test_buffer_size);
      CudaMgr::setContext(device_num);
      auto actual_bandwidth =
          calculateTransferBandwidth(reinterpret_cast<int8_t*>(host_buffer) +
                                         test_buffer_size - test_buffer_size / i,
                                     device_buffer,
                                     test_buffer_size / i,
                                     MemCpyKind::kHostToDevice,
                                     false /* use_jump_buffer */,
                                     device_num);
      device_properties_[device_num].hostToDeviceBandwidthNotPinnedBs.emplace_back(
          actual_bandwidth);

      actual_bandwidth =
          calculateTransferBandwidth(reinterpret_cast<int8_t*>(host_buffer) +
                                         test_buffer_size - test_buffer_size / i,
                                     device_buffer,
                                     test_buffer_size / i,
                                     MemCpyKind::kDeviceToHost,
                                     false /* use_jump_buffer */,
                                     device_num);
      device_properties_[device_num].deviceToHostBandwidthNotPinnedBs.emplace_back(
          actual_bandwidth);
    }
    if (g_enable_d2h_compression) {
      if (g_enable_d2h_compression) {
        generateSparseBuffer(
            test_buffer_size, 0.990, reinterpret_cast<int8_t*>(host_buffer));

        checkError(cuMemcpyHtoD(device_buffer, host_buffer, test_buffer_size));
      }

      // curandGenerator_t generator;
      // curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MT19937);
      // curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);
      // curandGeneratePoisson(generator,
      //                       reinterpret_cast<uint32_t*>(device_buffer),
      //                       test_buffer_size / sizeof(int),
      //                       1000);
      int64_t overhead_in_us = 0;
      constexpr size_t num_iterations = 100;
      for (size_t i = 1; i < num_iterations; ++i) {
        // Warm up the GPU and calculate API overhead
        size_t max_out_bytes = bitcompMaxBuflen(overhead_buffer_size / 2);
        auto actual_size =
            computePaddedBufferSize(overhead_buffer_size - max_out_bytes, 8);
        overhead_in_us +=
            (actual_size * 1000) / calculateCompressionBandwidth(
                                       device_buffer, actual_size, BITCOMP_SPARSE_ALGO);
      }
      device_properties_[device_num].gpuCompressionApiOverhead =
          overhead_in_us / num_iterations;

      VLOG(2) << " Average of overhead of compression API: " << overhead_in_us << " us.";

      for (size_t i = 1; i < 256; i <<= 1) {
        size_t max_out_bytes = bitcompMaxBuflen(test_buffer_size / i / 2);
        auto actual_size =
            computePaddedBufferSize(test_buffer_size / i - max_out_bytes, 8);
        auto actual_bandwidth = calculateCompressionBandwidth(
            device_buffer, actual_size, BITCOMP_SPARSE_ALGO);

        device_properties_[device_num].gpuCompressionSparseBandwidthBs.emplace_back(
            actual_bandwidth);

        calculateTransferBandwidth(host_buffer_m,
                                   device_buffer + actual_size,
                                   max_out_bytes,
                                   MemCpyKind::kDeviceToHost,
                                   false /* use_jump_buffer */,
                                   device_num);

        actual_bandwidth =
            calculateCpuDecompressionBandwidth(reinterpret_cast<int8_t*>(host_buffer_m),
                                               actual_size,
                                               cpu_threads() / 2,
                                               BITCOMP_SPARSE_ALGO);

        device_properties_[device_num].cpuDecompressionSparseBandwidthBs.emplace_back(
            actual_bandwidth);

        actual_bandwidth =
            calculateCpuDecompressionBandwidth(reinterpret_cast<int8_t*>(host_buffer_m),
                                               actual_size,
                                               1,
                                               BITCOMP_SPARSE_ALGO);

        device_properties_[device_num]
            .cpuSerialDecompressionSparseBandwidthBs.emplace_back(actual_bandwidth);

        actual_bandwidth = calculateCompressionBandwidth(
            device_buffer, actual_size, BITCOMP_DEFAULT_ALGO);

        device_properties_[device_num].gpuCompressionBandwidthBs.emplace_back(
            actual_bandwidth);

        calculateTransferBandwidth(host_buffer_m,
                                   device_buffer + actual_size,
                                   max_out_bytes,
                                   MemCpyKind::kDeviceToHost,
                                   false /* use_jump_buffer */,
                                   device_num);

        actual_bandwidth =
            calculateCpuDecompressionBandwidth(reinterpret_cast<int8_t*>(host_buffer_m),
                                               actual_size,
                                               cpu_threads() / 2,
                                               BITCOMP_DEFAULT_ALGO);

        device_properties_[device_num].cpuDecompressionBandwidthBs.emplace_back(
            actual_bandwidth);

        actual_bandwidth =
            calculateCpuDecompressionBandwidth(reinterpret_cast<int8_t*>(host_buffer_m),
                                               actual_size,
                                               1,
                                               BITCOMP_DEFAULT_ALGO);

        device_properties_[device_num].cpuSerialDecompressionBandwidthBs.emplace_back(
            actual_bandwidth);
      }
    }
    checkError(cuMemFree(device_buffer));
  }
  free(host_buffer);
  free(host_buffer_m);
}
void CudaMgr::fillDeviceProperties() {
  device_properties_.resize(device_count_);
  cuDriverGetVersion(&gpu_driver_version_);
  for (int device_num = 0; device_num < device_count_; ++device_num) {
    checkError(
        cuDeviceGet(&device_properties_[device_num].device, device_num + start_gpu_));
    CUuuid cuda_uuid;
    checkError(cuDeviceGetUuid(&cuda_uuid, device_properties_[device_num].device));
    device_properties_[device_num].uuid = heavyai::UUID(cuda_uuid.bytes);
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].computeMajor,
                                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].computeMinor,
                                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                    device_properties_[device_num].device));
    checkError(cuDeviceTotalMem(&device_properties_[device_num].globalMem,
                                device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].constantMem,
                                    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
                                    device_properties_[device_num].device));
    checkError(
        cuDeviceGetAttribute(&device_properties_[device_num].sharedMemPerMP,
                             CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                             device_properties_[device_num].device));
    if (g_enable_gpu_dynamic_smem &&
        is_arch_pascal_or_greater(device_properties_[device_num].computeMajor)) {
      checkError(
          cuDeviceGetAttribute(&device_properties_[device_num].sharedMemPerBlockOptIn,
                               CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                               device_properties_[device_num].device));
    }
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].sharedMemPerBlock,
                                    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].numMPs,
                                    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].warpSize,
                                    CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].maxThreadsPerBlock,
                                    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].maxRegistersPerBlock,
                                    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].maxRegistersPerMP,
                                    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].pciBusId,
                                    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].pciDeviceId,
                                    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].clockKhz,
                                    CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].memoryClockKhz,
                                    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].memoryBusWidth,
                                    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
                                    device_properties_[device_num].device));
    device_properties_[device_num].memoryBandwidthGBs =
        device_properties_[device_num].memoryClockKhz / 1000000.0 / 8.0 *
        device_properties_[device_num].memoryBusWidth * 2;

    // capture memory allocation granularity
    device_properties_[device_num].allocationGranularity = getGranularity(device_num);
    finishDevicePropertiesInitialization();

    min_shared_memory_per_block_for_all_devices =
        computeMinSharedMemoryPerBlockForAllDevices();
    min_num_mps_for_all_devices = computeMinNumMPsForAllDevices();
  }
}

int8_t* CudaMgr::allocateDeviceMem(const size_t num_bytes,
                                   const int device_num,
                                   const bool is_slab) {
  std::lock_guard<std::mutex> map_lock(device_mutex_);
  setContext(device_num);

  CUdeviceptr device_ptr{};
  CUmemGenericAllocationHandle handle{};
  auto granularity = getGranularity(device_num);
  // reserve the actual memory
  auto padded_num_bytes = computePaddedBufferSize(num_bytes, granularity);
  auto status = cuMemAddressReserve(&device_ptr, padded_num_bytes, granularity, 0, 0);

  if (status == CUDA_SUCCESS) {
    // create a handle for the allocation
    CUmemAllocationProp allocation_prop{};
    allocation_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocation_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    allocation_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    allocation_prop.location.id = device_num + start_gpu_;
    status = cuMemCreate(&handle, padded_num_bytes, &allocation_prop, 0);

    if (status == CUDA_SUCCESS) {
      // map the memory
      status = cuMemMap(device_ptr, padded_num_bytes, 0, handle, 0);

      if (status == CUDA_SUCCESS) {
        // set the memory access
        CUmemAccessDesc access_desc{};
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = device_num + start_gpu_;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        status = cuMemSetAccess(device_ptr, padded_num_bytes, &access_desc, 1);
      }
    }
  }

  if (status != CUDA_SUCCESS) {
    // clean up in reverse order
    if (device_ptr && handle) {
      cuMemUnmap(device_ptr, padded_num_bytes);
    }
    if (handle) {
      cuMemRelease(handle);
    }
    if (device_ptr) {
      cuMemAddressFree(device_ptr, padded_num_bytes);
    }
    throw CudaErrorException(status);
  }
  // emplace in the map
  auto const& device_uuid = getDeviceProperties(device_num)->uuid;
  getDeviceMemoryAllocationMap().addAllocation(
      device_ptr, padded_num_bytes, handle, device_uuid, device_num, is_slab);
  // notify
  getDeviceMemoryAllocationMap().notifyMapChanged(device_uuid, is_slab);
  if (log_memory_activity_) {
    VLOG(1) << "Allocate GPU memory: address: " << reinterpret_cast<void*>(device_ptr)
            << ", requested: " << num_bytes << " bytes, allocated: " << padded_num_bytes
            << ", device id: " << device_num << ", slab? " << is_slab;
  }
  return reinterpret_cast<int8_t*>(device_ptr);
}

void CudaMgr::freeDeviceMem(int8_t* device_ptr) {
  // take lock
  std::lock_guard<std::mutex> map_lock(device_mutex_);
  // fetch and remove from map
  auto const cu_device_ptr = reinterpret_cast<CUdeviceptr>(device_ptr);
  auto allocation = getDeviceMemoryAllocationMap().removeAllocation(cu_device_ptr);
  // attempt to unmap, release, free
  auto status_unmap = cuMemUnmap(cu_device_ptr, allocation.size);
  auto status_release = cuMemRelease(allocation.handle);
  auto status_free = cuMemAddressFree(cu_device_ptr, allocation.size);
  // check for errors
  checkError(status_unmap);
  checkError(status_release);
  checkError(status_free);
  // notify
  getDeviceMemoryAllocationMap().notifyMapChanged(allocation.device_uuid,
                                                  allocation.is_slab);
  if (log_memory_activity_) {
    VLOG(1) << "Deallocate GPU memory: address: " << static_cast<void*>(device_ptr)
            << ", size: " << allocation.size << ", device id: " << allocation.device_num
            << ", slab? " << allocation.is_slab;
  }
}

void CudaMgr::zeroDeviceMem(int8_t* device_ptr,
                            const size_t num_bytes,
                            const int device_num,
                            CUstream cuda_stream) {
  setDeviceMem(device_ptr, 0, num_bytes, device_num, cuda_stream);
}

void CudaMgr::setDeviceMem(int8_t* device_ptr,
                           const unsigned char uc,
                           const size_t num_bytes,
                           const int device_num,
                           CUstream cuda_stream) {
  setContext(device_num);
  if (!cuda_stream) {
    checkError(cuMemsetD8(reinterpret_cast<CUdeviceptr>(device_ptr), uc, num_bytes));
  } else {
    checkError(cuMemsetD8Async(
        reinterpret_cast<CUdeviceptr>(device_ptr), uc, num_bytes, cuda_stream));
    checkError(cuStreamSynchronize(cuda_stream));
  }
}

/**
 * Returns true if all devices have Volta micro-architecture
 * Returns false, if there is any non-Volta device available.
 */
bool CudaMgr::isArchVoltaOrGreaterForAll() const {
  CHECK(device_properties_initialized_);
  for (int i = 0; i < device_count_; i++) {
    if (device_properties_[i].computeMajor < 7) {
      return false;
    }
  }
  return true;
}

/**
 * This function returns the minimum available dynamic shared memory that is available
 * per block for all GPU devices.
 */
size_t CudaMgr::computeMinSharedMemoryPerBlockForAllDevices() const {
  CHECK(device_properties_initialized_);
  int shared_mem_size = 0;
  for (int d = 0; d < device_count_; d++) {
    int size = g_enable_gpu_dynamic_smem ? device_properties_[d].sharedMemPerBlockOptIn
                                         : device_properties_[d].sharedMemPerBlock;
    shared_mem_size = (d == 0) ? size : std::min(shared_mem_size, size);
  }
  return shared_mem_size;
}

/**
 * This function returns the minimum number of multiprocessors (MPs, also known as SMs)
 * per device across all GPU devices
 */
size_t CudaMgr::computeMinNumMPsForAllDevices() const {
  CHECK(device_properties_initialized_);
  int num_mps = device_count_ > 0 ? device_properties_.front().numMPs : 0;
  for (int d = 1; d < device_count_; d++) {
    num_mps = std::min(num_mps, device_properties_[d].numMPs);
  }
  return num_mps;
}
void CudaMgr::createDeviceContexts() {
  CHECK(device_properties_initialized_);
  CHECK_EQ(device_contexts_.size(), size_t(0));
  device_contexts_.resize(device_count_);
  for (int d = 0; d < device_count_; ++d) {
    CUresult status = cuCtxCreate(&device_contexts_[d], 0, device_properties_[d].device);
    if (status != CUDA_SUCCESS) {
      // this is called from destructor so we need
      // to clean up
      // destroy all contexts up to this point
      for (int destroy_id = 0; destroy_id <= d; ++destroy_id) {
        try {
          checkError(cuCtxDestroy(device_contexts_[destroy_id]));
        } catch (const CudaErrorException& e) {
          LOG(ERROR) << "Failed to destroy CUDA context for device ID " << destroy_id
                     << " with " << e.what()
                     << ". CUDA contexts were being destroyed due to an error creating "
                        "CUDA context for device ID "
                     << d << " out of " << device_count_ << " (" << error_message(status)
                     << ").";
        }
      }
      // checkError will translate the message and throw
      checkError(status);
    }
  }
}

void CudaMgr::setContext(const int device_num) const {
  // deviceNum is the device number relative to startGpu (realDeviceNum - startGpu_)
  CHECK_LT(device_num, device_count_);
  set_context(device_contexts_, device_num);
}

int CudaMgr::getContext() const {
  CUcontext cnow;
  checkError(cuCtxGetCurrent(&cnow));
  if (cnow == NULL) {
    throw std::runtime_error("no cuda device context");
  }
  int device_num{0};
  for (auto& c : device_contexts_) {
    if (c == cnow) {
      return device_num;
    }
    ++device_num;
  }
  // TODO(sy): Change device_contexts_ to have O(1) lookup? (Or maybe not worth it.)
  throw std::runtime_error("invalid cuda device context");
}

std::string returnAllDevicesProperties(const std::vector<size_t>& prop,
                                       const std::string& unit_size) {
  std::ostringstream oss;
  for (const auto& val : prop) {
    oss << " " << (val * 1000.0 / 1073741824.0) << unit_size;
  }
  oss << ".";

  return oss.str();
}

void CudaMgr::logDeviceProperties() const {
  CHECK(device_properties_initialized_);
  LOG(INFO) << "CUDA Driver version: " << gpu_driver_version_;
  LOG(INFO) << "Using " << device_count_ << " Gpus.";
  for (int d = 0; d < device_count_; ++d) {
    VLOG(1) << "Device: " << device_properties_[d].device;
    VLOG(1) << "UUID: " << device_properties_[d].uuid;
    VLOG(1) << "Clock (khz): " << device_properties_[d].clockKhz;
    VLOG(1) << "Compute Major: " << device_properties_[d].computeMajor;
    VLOG(1) << "Compute Minor: " << device_properties_[d].computeMinor;
    VLOG(1) << "PCI bus id: " << device_properties_[d].pciBusId;
    VLOG(1) << "PCI deviceId id: " << device_properties_[d].pciDeviceId;
    VLOG(1) << "Per device global memory: "
            << device_properties_[d].globalMem / 1073741824.0 << " GB";
    VLOG(1) << "Memory clock (khz): " << device_properties_[d].memoryClockKhz;
    VLOG(1) << "Memory bandwidth: " << device_properties_[d].memoryBandwidthGBs
            << " GB/sec";

    VLOG(1) << "Constant Memory: " << device_properties_[d].constantMem;
    VLOG(1) << "Shared memory per multiprocessor: "
            << device_properties_[d].sharedMemPerMP;
    VLOG(1) << "Shared memory per block: " << device_properties_[d].sharedMemPerBlock;
    if (g_enable_gpu_dynamic_smem) {
      VLOG(1) << "Shared memory per block (Dynamic): "
              << device_properties_[d].sharedMemPerBlockOptIn;
    }
    VLOG(1) << "Number of MPs: " << device_properties_[d].numMPs;
    VLOG(1) << "Warp Size: " << device_properties_[d].warpSize;
    VLOG(1) << "Max threads per block: " << device_properties_[d].maxThreadsPerBlock;
    VLOG(1) << "Max registers per block: " << device_properties_[d].maxRegistersPerBlock;
    VLOG(1) << "Max register per MP: " << device_properties_[d].maxRegistersPerMP;
    VLOG(1) << "Memory bus width in bits: " << device_properties_[d].memoryBusWidth;
    VLOG(1) << "Bus bandwidth Host to Device (Pinned): "
            << returnAllDevicesProperties(
                   device_properties_[d].hostToDeviceBandwidthPinnedBs, " GB/sec");
    VLOG(1) << "Bus bandwidth Device to Host (Pinned): "
            << returnAllDevicesProperties(
                   device_properties_[d].deviceToHostBandwidthPinnedBs, " GB/sec");
    VLOG(1) << "Bus bandwidth Host to Device (JumpBuffers): "
            << returnAllDevicesProperties(
                   device_properties_[d].hostToDeviceBandwidthJumpBufferBs, " GB/sec");
    VLOG(1) << "Bus bandwidth Device to Host (JumpBuffers): "
            << returnAllDevicesProperties(
                   device_properties_[d].deviceToHostBandwidthJumpBufferBs, " GB/sec");
    VLOG(1) << "Bus bandwidth Host to Device (Not Pinned): "
            << returnAllDevicesProperties(
                   device_properties_[d].hostToDeviceBandwidthNotPinnedBs, " GB/sec");
    VLOG(1) << "Bus bandwidth Device to Host (Not Pinned): "

            << returnAllDevicesProperties(
                   device_properties_[d].deviceToHostBandwidthNotPinnedBs, " GB/sec");
    if (g_enable_d2h_compression) {
      VLOG(1) << "Gpu Bitcomp Api Call Overhead "
              << device_properties_[d].gpuCompressionApiOverhead << "us.";
      VLOG(1) << "Gpu Bitcomp sparse compression bandwidth (GBs): "
              << returnAllDevicesProperties(
                     device_properties_[d].gpuCompressionSparseBandwidthBs, " GB/sec");
      VLOG(1) << "Gpu Bitcomp default compression bandwidth (GBs): "
              << returnAllDevicesProperties(
                     device_properties_[d].gpuCompressionBandwidthBs, " GB/sec");
      VLOG(1) << "Cpu Bitcomp sparse decompression max bandwidth (GBs): "
              << returnAllDevicesProperties(
                     device_properties_[d].cpuDecompressionSparseBandwidthBs, " GB/sec");
      VLOG(1) << "Cpu Bitcomp default max decompression bandwidth (GBs): "
              << returnAllDevicesProperties(
                     device_properties_[d].cpuDecompressionBandwidthBs, " GB/sec");
      VLOG(1) << "Cpu Bitcomp sparse decompression serial bandwidth (GBs): "
              << returnAllDevicesProperties(
                     device_properties_[d].cpuSerialDecompressionSparseBandwidthBs,
                     " GB/sec");
      VLOG(1) << "Cpu Bitcomp default max decompression serial bandwidth (GBs): "
              << returnAllDevicesProperties(
                     device_properties_[d].cpuSerialDecompressionBandwidthBs, " GB/sec");
    }
  }
}

void CudaMgr::checkError(CUresult status) const {
  check_error(status);
}

void CudaMgr::bitcompCheck(bitcompResult_t status) const {
  if (status != bitcompResult_t::BITCOMP_SUCCESS) {
    VLOG(1) << "Error compressing " << status;
    throw std::runtime_error("Bitcomp error: " + status);
  }
}

DeviceMemoryAllocationMap& CudaMgr::getDeviceMemoryAllocationMap() {
  CHECK(device_memory_allocation_map_);
  return *device_memory_allocation_map_;
}

int CudaMgr::exportHandle(const uint64_t handle) const {
  int fd{-1};
  checkError(cuMemExportToShareableHandle(
      &fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
  return fd;
}

void CudaMgr::enableMemoryActivityLog() {
  log_memory_activity_ = true;
}

bool CudaMgr::logMemoryActivity() const {
  return log_memory_activity_;
}

std::ostream& operator<<(std::ostream& os, NvidiaDeviceArch device_arch) {
  constexpr size_t array_size{8};
  constexpr char const* strings[array_size]{
      "Kepler", "Maxwell", "Pascal", "Volta", "Turing", "Ampere", "Ada", "Hopper"};

  auto index = static_cast<size_t>(device_arch);
  CHECK_LT(index, array_size);
  return os << strings[index];
}
}  // namespace CudaMgr_Namespace

std::string get_cuda_home(void) {
  static const char* CUDA_DEFAULT_PATH = "/usr/local/cuda";
  const char* env = nullptr;

  if (!(env = getenv("CUDA_HOME")) && !(env = getenv("CUDA_DIR"))) {
    // check if the default CUDA directory exists: /usr/local/cuda
    if (boost::filesystem::exists(boost::filesystem::path(CUDA_DEFAULT_PATH))) {
      env = CUDA_DEFAULT_PATH;
    }
  }

  if (env == nullptr) {
    LOG(WARNING) << "Could not find CUDA installation path: environment variables "
                    "CUDA_HOME or CUDA_DIR are not defined";
    return "";
  }

  // check if the CUDA directory is sensible:
  auto cuda_include_dir = env + std::string("/include");
  auto cuda_h_file = cuda_include_dir + "/cuda.h";
  if (!boost::filesystem::exists(boost::filesystem::path(cuda_h_file))) {
    LOG(WARNING) << "cuda.h does not exist in `" << cuda_include_dir << "`. Discarding `"
                 << env << "` as CUDA installation path.";
    return "";
  }

  return std::string(env);
}

std::string get_cuda_libdevice_dir(void) {
  static const char* CUDA_DEFAULT_PATH = "/usr/local/cuda";
  const char* env = nullptr;

  if (!(env = getenv("CUDA_HOME")) && !(env = getenv("CUDA_DIR"))) {
    // check if the default CUDA directory exists: /usr/local/cuda
    if (boost::filesystem::exists(boost::filesystem::path(CUDA_DEFAULT_PATH))) {
      env = CUDA_DEFAULT_PATH;
    }
  }

  if (env == nullptr) {
    LOG(WARNING) << "Could not find CUDA installation path: environment variables "
                    "CUDA_HOME or CUDA_DIR are not defined";
    return "";
  }

  // check if the CUDA directory is sensible:
  auto libdevice_dir = env + std::string("/nvvm/libdevice");
  auto libdevice_bc_file = libdevice_dir + "/libdevice.10.bc";
  if (!boost::filesystem::exists(boost::filesystem::path(libdevice_bc_file))) {
    LOG(WARNING) << "`" << libdevice_bc_file << "` does not exist. Discarding `" << env
                 << "` as CUDA installation path with libdevice.";
    return "";
  }

  return libdevice_dir;
}
