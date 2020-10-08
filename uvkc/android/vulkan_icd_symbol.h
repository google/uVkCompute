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

#ifndef UVKC_ANDROID_VULKAN_ICD_SYMBOL_H_
#define UVKC_ANDROID_VULKAN_ICD_SYMBOL_H_

#include <vulkan/vulkan.h>

#include "absl/status/statusor.h"
#include "uvkc/base/dynamic_library.h"

namespace uvkc {
namespace android {

// Returns the vkGetInstanceProcAddr symbol from the Vulkan ICD dynamic library.
//
// On Android, the Vulkan ICD may not directly expose the vkGetInstanceProcAddr
// symbol because the Android Vulkan loader has a different contract to open and
// query the symbols from the vulkan ICDs than the desktop vulkan loaders. This
// function mimics the Vulkan loader to query and return vkGetInstanceProcAddr.
absl::StatusOr<PFN_vkGetInstanceProcAddr> GetVulkanICDGetInstanceProceAddr(
    const DynamicLibrary &dylib);

}  // namespace android
}  // namespace uvkc

#endif  // UVKC_ANDROID_VULKAN_ICD_SYMBOL_H_
