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

#ifndef UVKC_VULKAN_STATUS_UTIL_H_
#define UVKC_VULKAN_STATUS_UTIL_H_

#include <vulkan/vulkan.h>

#include "uvkc/base/status.h"

namespace uvkc {
namespace vulkan {

// Converts a VkResult to an absl::Status.
absl::Status VkResultToStatus(VkResult result);

// Executes an expression `rexpr` that returns a `VkResult`. On error, returns
// from the current function.
#define VK_RETURN_IF_ERROR(rexpr) UVKC_RETURN_IF_ERROR(VkResultToStatus(rexpr))

}  // namespace vulkan
}  // namespace uvkc

#endif  // UVKC_VULKAN_STATUS_UTIL_H_
