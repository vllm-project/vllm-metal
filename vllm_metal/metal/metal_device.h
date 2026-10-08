// SPDX-License-Identifier: Apache-2.0
// Hardware facts shared by the MLX and PyTorch launchers.
#pragma once

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

namespace vllm_metal::hardware {

// GPU core count via IORegistry.  Metal/MLX expose no core-count API, but the
// split-KV gate needs to scale per machine — a small laptop GPU and a large
// desktop one saturate at very different grid sizes. Cache the hardware query;
// zero means unknown, so GQA performance gates stay disabled.
// Tests may inject a count (zero simulates detection failure) through
// `_override_detected_gpu_core_count_for_test`. Production must not call it.
static std::atomic<int> g_test_gpu_core_count{-1};

static int hardware_gpu_core_count() {
  static const int v = []() {
    int cores = 0;
    io_iterator_t it;
    if (IOServiceGetMatchingServices(kIOMainPortDefault,
                                     IOServiceMatching("AGXAccelerator"),
                                     &it) == KERN_SUCCESS) {
      io_object_t obj;
      while ((obj = IOIteratorNext(it))) {
        CFTypeRef p = IORegistryEntrySearchCFProperty(
            obj, kIOServicePlane, CFSTR("gpu-core-count"),
            kCFAllocatorDefault, kIORegistryIterateRecursively);
        if (p) {
          if (CFGetTypeID(p) == CFNumberGetTypeID())
            CFNumberGetValue((CFNumberRef)p, kCFNumberIntType, &cores);
          CFRelease(p);
        }
        IOObjectRelease(obj);
        if (cores > 0) break;
      }
      IOObjectRelease(it);
    }
    return cores;
  }();
  return v;
}

static int detected_gpu_core_count() {
  const int override =
      g_test_gpu_core_count.load(std::memory_order_relaxed);
  return override >= 0 ? override : hardware_gpu_core_count();
}

// Keep the established split-KV budget on VMs that omit gpu-core-count.
static int gpu_core_count() {
  const int cores = detected_gpu_core_count();
  if (cores > 0) return cores;
  static std::atomic_flag warned = ATOMIC_FLAG_INIT;
  if (!warned.test_and_set(std::memory_order_relaxed))
    std::fputs("[vllm-metal] WARNING: GPU core count is unknown "
               "(hardware information unavailable; possibly a VM). "
               "Assuming 14 cores for split-KV scheduling.\n", stderr);
  return 14;
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
