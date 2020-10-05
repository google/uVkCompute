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

#include "uvkc/vulkan/buffer.h"

#include "absl/status/status.h"
#include "uvkc/vulkan/status_util.h"

namespace uvkc {
namespace vulkan {

Buffer::Buffer(VkDevice device, VkDeviceMemory memory, VkBuffer buffer,
               const DynamicSymbols &symbols)
    : buffer_(buffer), device_(device), memory_(memory), symbols_(symbols) {}

Buffer::~Buffer() {
  symbols_.vkDestroyBuffer(device_, buffer_, /*pAllocator=*/nullptr);
  symbols_.vkFreeMemory(device_, memory_, /*pAllocator=*/nullptr);
}

VkBuffer Buffer::buffer() const { return buffer_; }

absl::StatusOr<void *> Buffer::MapMemory(size_t offset, size_t size) {
  void *data = nullptr;
  VK_RETURN_IF_ERROR(
      symbols_.vkMapMemory(device_, memory_, offset, size, /*flags=*/0, &data));
  return data;
}

void Buffer::UnmapMemory() { symbols_.vkUnmapMemory(device_, memory_); }

}  // namespace vulkan
}  // namespace uvkc
