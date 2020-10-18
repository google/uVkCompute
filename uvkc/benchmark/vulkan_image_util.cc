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

#include "uvkc/benchmark/vulkan_image_util.h"

#include "uvkc/base/status.h"

namespace uvkc {
namespace benchmark {

absl::Status SetDeviceImageViaStagingBuffer(
    vulkan::Device *device, vulkan::Image *device_image,
    VkExtent3D image_dimensions, VkImageLayout to_layout,
    size_t buffer_size_in_bytes,
    const std::function<void(void *, size_t)> &staging_buffer_setter) {
  // Create a staging buffer.
  UVKC_ASSIGN_OR_RETURN(
      auto staging_buffer,
      device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           buffer_size_in_bytes));
  UVKC_ASSIGN_OR_RETURN(void *src_staging_ptr,
                        staging_buffer->MapMemory(0, buffer_size_in_bytes));
  staging_buffer_setter(src_staging_ptr, buffer_size_in_bytes);
  staging_buffer->UnmapMemory();

  // Copy the data to the device.
  UVKC_ASSIGN_OR_RETURN(auto cmdbuffer, device->AllocateCommandBuffer());
  UVKC_RETURN_IF_ERROR(cmdbuffer->Begin());
  UVKC_RETURN_IF_ERROR(
      cmdbuffer->TransitionImageLayout(*device_image, VK_IMAGE_LAYOUT_UNDEFINED,
                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
  cmdbuffer->CopyBufferToImage(*staging_buffer, 0, *device_image,
                               image_dimensions);
  UVKC_RETURN_IF_ERROR(cmdbuffer->TransitionImageLayout(
      *device_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, to_layout));
  UVKC_RETURN_IF_ERROR(cmdbuffer->End());
  UVKC_RETURN_IF_ERROR(device->QueueSubmitAndWait(*cmdbuffer));

  return absl::OkStatus();
}

absl::Status GetDeviceImageViaStagingBuffer(
    vulkan::Device *device, vulkan::Image *device_image,
    VkExtent3D image_dimensions, VkImageLayout from_layout,
    size_t buffer_size_in_bytes,
    const std::function<void(void *, size_t)> &staging_buffer_getter) {
  // Create a staging buffer.
  UVKC_ASSIGN_OR_RETURN(
      auto staging_buffer,
      device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           buffer_size_in_bytes));

  // Copy the data from the device.
  UVKC_ASSIGN_OR_RETURN(auto cmdbuffer, device->AllocateCommandBuffer());
  UVKC_RETURN_IF_ERROR(cmdbuffer->Begin());
  UVKC_RETURN_IF_ERROR(cmdbuffer->TransitionImageLayout(
      *device_image, from_layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL));
  cmdbuffer->CopyImageToBuffer(*device_image, image_dimensions, *staging_buffer,
                               0);
  UVKC_RETURN_IF_ERROR(cmdbuffer->End());
  UVKC_RETURN_IF_ERROR(device->QueueSubmitAndWait(*cmdbuffer));

  UVKC_ASSIGN_OR_RETURN(void *dst_staging_ptr,
                        staging_buffer->MapMemory(0, buffer_size_in_bytes));
  staging_buffer_getter(dst_staging_ptr, buffer_size_in_bytes);
  staging_buffer->UnmapMemory();

  return absl::OkStatus();
}

}  // namespace benchmark
}  // namespace uvkc
