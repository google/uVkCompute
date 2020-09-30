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

#include "uvkc/vulkan/command_buffer.h"

#include "absl/status/status.h"
#include "uvkc/vulkan/status_util.h"

namespace uvkc {
namespace vulkan {

CommandBuffer::CommandBuffer(VkDevice device, VkCommandBuffer command_buffer)
    : command_buffer_(command_buffer), device_(device) {}

CommandBuffer::~CommandBuffer() = default;

VkCommandBuffer CommandBuffer::command_buffer() const {
  return command_buffer_;
}

absl::Status CommandBuffer::Begin() {
  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.pNext = nullptr;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  begin_info.pInheritanceInfo = nullptr;
  return VkResultToStatus(vkBeginCommandBuffer(command_buffer_, &begin_info));
}

absl::Status CommandBuffer::End() {
  return VkResultToStatus(vkEndCommandBuffer(command_buffer_));
}

void CommandBuffer::CopyBuffer(const Buffer &src_buffer, size_t src_offset,
                               const Buffer &dst_buffer, size_t dst_offset,
                               size_t length) {
  VkBufferCopy region = {};
  region.srcOffset = src_offset;
  region.dstOffset = dst_offset;
  region.size = length;
  vkCmdCopyBuffer(command_buffer_, src_buffer.buffer(), dst_buffer.buffer(),
                  /*regionCount=*/1, &region);
}

void CommandBuffer::BindPipelineAndDescriptorSets(
    const Pipeline &pipeline,
    absl::Span<const BoundDescriptorSet> bound_descriptor_sets) {
  vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipeline.pipeline());

  for (const auto &descriptor_set : bound_descriptor_sets) {
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline.pipeline_layout(), descriptor_set.index,
                            /*descriptorSetCount=*/1,
                            /*pDescriptorSets=*/&descriptor_set.set,
                            /*dynamicOffsetCount=*/0,
                            /*pDynamicOffsets=*/nullptr);
  }
}

void CommandBuffer::Dispatch(uint32_t x, uint32_t y, uint32_t z) {
  vkCmdDispatch(command_buffer_, x, y, z);
}

}  // namespace vulkan
}  // namespace uvkc
