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

#ifndef UVKC_BENCHMARK_VULKAN_IMAGE_UTIL_H_
#define UVKC_BENCHMARK_VULKAN_IMAGE_UTIL_H_

#include "absl/status/status.h"
#include "uvkc/vulkan/device.h"
#include "uvkc/vulkan/image.h"

namespace uvkc {
namespace benchmark {

// Sets data for a |device_image| via a CPU staging buffer by invoking
// |staging_buffer_setter| on the pointer pointing to the start of the CPU
// staging buffer.
//
// |device_image| is expected to have VK_IMAGE_USAGE_TRANSFER_DST_BIT bit.
//
// This function will discard the existing content in the image and transition
// it into |to_layout|.
absl::Status SetDeviceImageViaStagingBuffer(
    vulkan::Device *device, vulkan::Image *device_image,
    VkExtent3D image_dimensions, VkImageLayout to_layout,
    size_t buffer_size_in_bytes,
    const std::function<void(void *, size_t)> &staging_buffer_setter);

// Get data from a |device_image| via a CPU staging buffer by invoking
// |staging_buffer_getter| on the pointer pointing to the start of the CPU
// staging buffer.
//
// |device_image| is expected to have VK_IMAGE_USAGE_TRANSFER_SRC_BIT bit.
//
// This function will transition the image from |from_layout| to
// VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL.
absl::Status GetDeviceImageViaStagingBuffer(
    vulkan::Device *device, vulkan::Image *device_image,
    VkExtent3D image_dimensions, VkImageLayout from_layout,
    size_t buffer_size_in_bytes,
    const std::function<void(void *, size_t)> &staging_buffer_getter);

}  // namespace benchmark
}  // namespace uvkc

#endif  // UVKC_BENCHMARK_VULKAN_IMAGE_UTIL_H_
