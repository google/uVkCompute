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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "uvkc/vulkan/buffer.h"
#include "uvkc/vulkan/dynamic_symbols.h"
#include "uvkc/vulkan/image.h"
#include "uvkc/vulkan/pipeline.h"
#include "uvkc/vulkan/timestamp_query_pool.h"

namespace uvkc {
namespace vulkan {

// A class representing a Vulkan command buffer.
//
// Objects from this class do not reset the Vulkan command buffers at
// destruction time; the pool is expected to release them all together.
class CommandBuffer {
 public:
  // Wraps a |command_buffer| from |device|.
  CommandBuffer(VkDevice device, VkCommandBuffer command_buffer,
                const DynamicSymbols &symbols);

  ~CommandBuffer();

  // Returns the VkCommandBuffer handle.
  VkCommandBuffer command_buffer() const;

  // Begins command buffer recording.
  absl::Status Begin();

  // Ends command buffer recording.
  absl::Status End();

  // Resets this command buffer to its initial state.
  absl::Status Reset();

  // Records a command to copy the |src_buffer| to |dst_buffer|.
  void CopyBuffer(const Buffer &src_buffer, size_t src_offset,
                  const Buffer &dst_buffer, size_t dst_offset, size_t length);

  // Records a command to copy the tightly packed data starting at |src_offset|
  // of the |src_buffer| to |dst_image|. The |dst_image| should be of
  // VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL.
  void CopyBufferToImage(const Buffer &src_buffer, size_t src_offset,
                         const Image &dst_image, VkExtent3D image_dimensions);

  // Records a command to copy the |src_image|'s data into a tightly packed
  // |dst_buffer| starting at |dst_offset|.  The |src_image| should be of
  // VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL.
  void CopyImageToBuffer(const Image &src_image, VkExtent3D image_dimensions,
                         const Buffer &dst_buffer, size_t dst_offset);

  // Performs image layout transition from |from_layout| to |to_layout| of the
  // given |image|.
  absl::Status TransitionImageLayout(const Image &image,
                                     VkImageLayout from_layout,
                                     VkImageLayout to_layout);

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

  // Records a command to reset the given timestamp |query_pool|.
  void ResetQueryPool(const TimestampQueryPool &query_pool);

  // Records a command to write the timestamp at the given |pipeline_stage| to
  // the query with |query_index| in the |query_pool|.
  void WriteTimestamp(const TimestampQueryPool &query_pool,
                      VkPipelineStageFlagBits pipeline_stage,
                      uint32_t query_index);

  // Records a dispatch command.
  void Dispatch(uint32_t x, uint32_t y, uint32_t z);

  // Records a pipeline barrier that synchronizes shader read from a compute
  // shader with shader write from a previous compute shader.
  void DispatchBarrier();

 private:
  VkCommandBuffer command_buffer_;

  VkDevice device_;

  const DynamicSymbols &symbols_;
};

}  // namespace vulkan
}  // namespace uvkc

#endif  // UVKC_VULKAN_COMMAND_BUFFER_H_
