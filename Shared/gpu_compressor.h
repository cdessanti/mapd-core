#ifdef HAVE_CUDA
#include "nvcomp/native/bitcomp.h"
#else
#include "nocuda.h"
#endif  // HAVE_CUDA
#include <cstdlib>
#include <ostream>
#include <sstream>
#include "Logger/Logger.h"

#pragma once 

enum class GpuCompressionAlgo {
  kNocomopression = 0,
  kNvidiaBitcompNativeDefault,
  kNvidiaBitcompNativeSparse,
  kNvidiaBitcompNativeAuto,
  kNvidiaBitcompNativeSparseAuto,
  kNvidiaBitcompNativeDefaultAuto
};

bitcompAlgorithm_t inline getBitcompNativeAlgoType(GpuCompressionAlgo algo) {
  switch(algo) {
    case GpuCompressionAlgo::kNvidiaBitcompNativeDefault:
    return BITCOMP_DEFAULT_ALGO;
    break;
    case  GpuCompressionAlgo::kNvidiaBitcompNativeSparse:
     return BITCOMP_SPARSE_ALGO;
     break;
    default:
      UNREACHABLE(); // << "The requeste gpu compression algoritm " << algo << " doesn't exists.";
  }
  UNREACHABLE();
  return BITCOMP_DEFAULT_ALGO ;
}