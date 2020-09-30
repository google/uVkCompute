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

#ifndef UVKC_VULKAN_COMMAND_BUFFER_H_
#define UVKC_VULKAN_COMMAND_BUFFER_H_

#include <vulkan/vulkan.h>

#include <memory>
#include <unordered_map>

#include "absl/status/statusor.h"
#include "uvkc/vulkan/buffer.h"
#include "uvkc/vulkan/pipeline.h"

namespace uvkc {
namespace vulkan {

// A class representing a Vulkan command buffer.
//
// Objects from this class do not reset the Vulkan command buffers at
// destruction time; the pool is expected to release them all together.
class CommandBuffer {
 public:
  // Wraps a |command_buffer| from |device|.
  CommandBuffer(VkDevice device, VkCommandBuffer command_buffer);

  ~CommandBuffer();

  // Returns the VkCommandBuffer handle.
  VkCommandBuffer command_buffer() const;

  // Begins command buffer recording.
  absl::Status Begin();

  // Ends command buffer recording.
  absl::Status End();

  // Records a command to copy the |src_buffer| to |dst_buffer|.
  void CopyBuffer(const Buffer &src_buffer, size_t src_offset,
                  const Buffer &dst_buffer, size_t dst_offset, size_t length);

  // A struct containing bound descriptor set information.
  struct BoundDescriptorSet {
    uint32_t index;
    VkDescriptorSet set;
  };

  // Records a command to bind the compute |pipeline| and resource descriptor
  // sets recorded in |bound_descriptor_sets| into this command buffer.
  void BindPipelineAndDescriptorSets(
      const Pipeline &pipeline,
      absl::Span<const BoundDescriptorSet> bound_descriptor_sets);

  // Records a dispatch command.
  void Dispatch(uint32_t x, uint32_t y, uint32_t z);

 private:
  VkCommandBuffer command_buffer_;

  VkDevice device_;
};

}  // namespace vulkan
}  // namespace uvkc

#endif  // UVKC_VULKAN_COMMAND_BUFFER_H_
