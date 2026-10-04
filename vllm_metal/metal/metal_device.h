// SPDX-License-Identifier: Apache-2.0
// Hardware facts shared by the MLX and PyTorch launchers.
#pragma once

#include <atomic>
#include <stdexcept>
#include <string>

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

namespace vllm_metal::hardware {

// GPU core count via IORegistry.  Metal/MLX expose no core-count API, but the
// split-KV gate needs to scale per machine — a small laptop GPU and a large
// desktop one saturate at very different grid sizes. Cache the hardware query;
// both backends require a positive count before initializing their kernels.
// Tests may inject a count (zero simulates detection failure) through
// `_override_detected_gpu_core_count_for_test`. Production must not call it.
static std::atomic<int> g_test_gpu_core_count{-1};

static int hardware_gpu_core_count() {
  static const int v = []() {
    int cores = 0;
    io_iterator_t it;
    const auto status = IOServiceGetMatchingServices(
        kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &it);
    if (status != KERN_SUCCESS) {
      throw std::runtime_error(
          "Cannot determine Apple GPU core count: IOServiceGetMatchingServices "
          "failed with IOKit error " + std::to_string(status));
    }
    io_object_t obj;
    while ((obj = IOIteratorNext(it))) {
      CFTypeRef p = IORegistryEntrySearchCFProperty(
          obj, kIOServicePlane, CFSTR("gpu-core-count"),
          kCFAllocatorDefault, kIORegistryIterateRecursively);
      if (p) {
        if (CFGetTypeID(p) == CFNumberGetTypeID() &&
            !CFNumberGetValue((CFNumberRef)p, kCFNumberIntType, &cores))
          cores = 0;
        CFRelease(p);
      }
      IOObjectRelease(obj);
      if (cores > 0) break;
    }
    IOObjectRelease(it);
    return cores;
  }();
  return v;
}

static int detected_gpu_core_count() {
  const int override =
      g_test_gpu_core_count.load(std::memory_order_relaxed);
  const int cores = override >= 0 ? override : hardware_gpu_core_count();
  if (cores <= 0) {
    throw std::runtime_error(
        "Cannot determine Apple GPU core count: AGXAccelerator has no valid "
        "positive gpu-core-count property");
  }
  return cores;
}

static void override_detected_gpu_core_count_for_test(int cores) {
  if (cores < -1)
    throw std::invalid_argument(
        "test GPU core override must be >= -1 (got " +
        std::to_string(cores) + ")");
  g_test_gpu_core_count.store(cores, std::memory_order_relaxed);
}

// MTLGPUFamilyApple10 (M5/A19). Keep the numeric value usable with older SDKs.
constexpr int kApple10 = 1010;

static bool nax_supported(bool supports_apple10) {
  if (__builtin_available(macOS 26.2, *)) return supports_apple10;
  return false;
}

}  // namespace vllm_metal::hardware
