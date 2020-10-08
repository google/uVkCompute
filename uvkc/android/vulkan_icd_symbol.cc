// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "uvkc/android/vulkan_icd_symbol.h"

namespace uvkc {
namespace android {

namespace {

// The following structs are from:
// https://android.googlesource.com/platform/hardware/libhardware/+/master/include/hardware/hardware.h

const char kHalModuleInfo[] = "HMI";

typedef struct hw_module_t {
  uint32_t tag;
  uint16_t module_api_version;
  uint16_t hal_api_version;
  const char *id;
  const char *name;
  const char *author;
  struct hw_module_methods_t *methods;
  void *dso;
#ifdef __LP64__
  uint64_t reserved[32 - 7];
#else
  uint32_t reserved[32 - 7];
#endif
} hw_module_t;

typedef struct hw_module_methods_t {
  int (*open)(const struct hw_module_t *module, const char *id,
              struct hw_device_t **device);
} hw_module_methods_t;

typedef struct hw_device_t {
  uint32_t tag;
  uint32_t version;
  struct hw_module_t *module;
#ifdef __LP64__
  uint64_t reserved[12];
#else
  uint32_t reserved[12];
#endif
  int (*close)(struct hw_device_t *device);
} hw_device_t;

// The following structs are from:
// https://android.googlesource.com/platform/frameworks/native/+/refs/heads/master/vulkan/include/hardware/hwvulkan.h

const char kHwVulkanDevice0[] = "vk0";

typedef struct hwvulkan_device_t {
  struct hw_device_t common;
  PFN_vkEnumerateInstanceExtensionProperties
      EnumerateInstanceExtensionProperties;
  PFN_vkCreateInstance CreateInstance;
  PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
} hwvulkan_device_t;

}  // namespace

absl::StatusOr<PFN_vkGetInstanceProcAddr> GetVulkanICDGetInstanceProceAddr(
    const DynamicLibrary &dylib) {
  // Mimicking the Android Vulkan loader to query vkGetInstanceProcAddr. See
  // https://source.android.com/devices/graphics/implement-vulkan#driver_emun
  // for more details.

  auto *module =
      reinterpret_cast<hw_module_t *>(dylib.GetSymbol(kHalModuleInfo));
  if (!module) {
    return absl::UnavailableError("cannot find the HMI symbol in Vulkan ICD");
  }

  hw_device_t *device = nullptr;
  if (module->methods->open(module, kHwVulkanDevice0, &device)) {
    return absl::UnavailableError("cannot open device from Vulkan ICD");
  }

  return reinterpret_cast<hwvulkan_device_t *>(device)->GetInstanceProcAddr;
}

}  // namespace android
}  // namespace uvkc
