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

#include "uvkc/vulkan/status_util.h"

namespace uvkc {
namespace vulkan {

absl::Status VkResultToStatus(VkResult result) {
  switch (result) {
    case VK_SUCCESS:
    case VK_NOT_READY:
    case VK_TIMEOUT:
    case VK_EVENT_SET:
    case VK_EVENT_RESET:
    case VK_INCOMPLETE:
      return absl::OkStatus();

    case VK_ERROR_OUT_OF_HOST_MEMORY:
      return absl::ResourceExhaustedError("VK_ERROR_OUT_OF_HOST_MEMORY");
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      return absl::ResourceExhaustedError("VK_ERROR_OUT_OF_DEVICE_MEMORY");
    case VK_ERROR_OUT_OF_POOL_MEMORY:
      return absl::ResourceExhaustedError("VK_ERROR_OUT_OF_POOL_MEMORY");

    case VK_ERROR_INITIALIZATION_FAILED:
      return absl::InternalError("VK_ERROR_INITIALIZATION_FAILED");
    case VK_ERROR_DEVICE_LOST:
      return absl::InternalError("VK_ERROR_DEVICE_LOST");
    case VK_ERROR_MEMORY_MAP_FAILED:
      return absl::InternalError("VK_ERROR_MEMORY_MAP_FAILED");

    case VK_ERROR_INCOMPATIBLE_DRIVER:
      return absl::FailedPreconditionError("VK_ERROR_INCOMPATIBLE_DRIVER");

    case VK_ERROR_INVALID_DEVICE_ADDRESS_EXT:
      return absl::OutOfRangeError("VK_ERROR_INVALID_DEVICE_ADDRESS_EXT");

    default:
      return absl::UnknownError("unhandled VkResult");
  }
}

}  // namespace vulkan
}  // namespace uvkc
