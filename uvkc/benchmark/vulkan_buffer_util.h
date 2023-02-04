// Copyright 2020-2023 Google LLC
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

#ifndef UVKC_BENCHMARK_VULKAN_BUFFER_UTIL_H_
#define UVKC_BENCHMARK_VULKAN_BUFFER_UTIL_H_

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "uvkc/vulkan/buffer.h"
#include "uvkc/vulkan/device.h"

namespace uvkc {
namespace benchmark {

// Sets data for a |device_buffer| via a CPU staging buffer by invoking
// |staging_buffer_setter| on the pointer pointing to the start of the CPU
// staging buffer. |device_buffer| is expected to have
// VK_BUFFER_USAGE_TRANSFER_DST_BIT bit.
absl::Status SetDeviceBufferViaStagingBuffer(
    vulkan::Device *device, vulkan::Buffer *device_buffer,
    size_t buffer_size_in_bytes,
    const std::function<void(void *, size_t)> &staging_buffer_setter);

// Convenience overload of `SetDeviceBufferViaStagingBuffer` that passes
// in a span of type |ElemetType| to the getter |stagin_beffer_setter|.
template <typename ElementType>
absl::Status SetDeviceBufferViaStagingBuffer(
    vulkan::Device *device, vulkan::Buffer *device_buffer,
    size_t buffer_size_in_bytes,
    const std::function<void(absl::Span<ElementType>)> &staging_buffer_setter) {
  return SetDeviceBufferViaStagingBuffer(
      device, device_buffer, buffer_size_in_bytes,
      [&staging_buffer_setter](void *buffer, size_t size) {
        staging_buffer_setter(absl::MakeSpan(static_cast<ElementType *>(buffer),
                                             size / sizeof(ElementType)));
      });
}

// Get data from a |device_buffer| via a CPU staging buffer by invoking
// |staging_buffer_getter| on the pointer pointing to the start of the CPU
// staging buffer. |device_buffer| is expected to have
// VK_BUFFER_USAGE_TRANSFER_SRC_BIT bit.
absl::Status GetDeviceBufferViaStagingBuffer(
    vulkan::Device *device, vulkan::Buffer *device_buffer,
    size_t buffer_size_in_bytes,
    const std::function<void(void *, size_t)> &staging_buffer_getter);

// Convenience overload of `GetDeviceBufferViaStagingBuffer` that passes
// in a span of type |ElemetType| to the getter |stagin_beffer_getter|.
template <typename ElementType>
absl::Status GetDeviceBufferViaStagingBuffer(
    vulkan::Device *device, vulkan::Buffer *device_buffer,
    size_t buffer_size_in_bytes,
    const std::function<void(absl::Span<const ElementType>)>
        &staging_buffer_getter) {
  return GetDeviceBufferViaStagingBuffer(
      device, device_buffer, buffer_size_in_bytes,
      [&staging_buffer_getter](void *buffer, size_t size) {
        staging_buffer_getter(
            absl::MakeSpan(static_cast<const ElementType *>(buffer),
                           size / sizeof(ElementType)));
      });
}

}  // namespace benchmark
}  // namespace uvkc

#endif  // UVKC_BENCHMARK_VULKAN_BUFFER_UTIL_H_
