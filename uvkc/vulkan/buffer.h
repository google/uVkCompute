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

#ifndef UVKC_VULKAN_BUFFER_H_
#define UVKC_VULKAN_BUFFER_H_

#include <vulkan/vulkan.h>

#include "absl/status/statusor.h"
#include "uvkc/vulkan/dynamic_symbols.h"

namespace uvkc {
namespace vulkan {

// A class representing a Vulkan buffer.
//
// This is just a simple wrapper around VkBuffer and its backing memory. It
// handles resource release at object destruction time.
class Buffer {
 public:
  // Wraps a Vulkan |buffer| and its backing |memory| from |device| and manages
  // deallocation of the |memory| and freeing of the |buffer|.
  Buffer(VkDevice device, VkDeviceMemory memory, VkBuffer buffer,
         const DynamicSymbols &symbols);

  ~Buffer();

  // Returns the VkBuffer handle.
  VkBuffer buffer() const;

  // Gets a CPU accessible memory address for the current buffer.
  absl::StatusOr<void *> MapMemory(size_t offset, size_t size);

  // Invalidate the CPU accessible memory address for the current buffer.
  void UnmapMemory();

 private:
  VkBuffer buffer_;

  VkDevice device_;
  VkDeviceMemory memory_;

  const DynamicSymbols &symbols_;
};

}  // namespace vulkan
}  // namespace uvkc

#endif  // UVKC_VULKAN_BUFFER_H_
