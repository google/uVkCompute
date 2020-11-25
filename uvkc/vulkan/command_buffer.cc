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
#include "absl/strings/str_cat.h"
#include "uvkc/vulkan/status_util.h"
#include "uvkc/vulkan/timestamp_query_pool.h"

namespace uvkc {
namespace vulkan {

CommandBuffer::CommandBuffer(VkDevice device, VkCommandBuffer command_buffer,
                             const DynamicSymbols &symbols)
    : command_buffer_(command_buffer), device_(device), symbols_(symbols) {}

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
  return VkResultToStatus(
      symbols_.vkBeginCommandBuffer(command_buffer_, &begin_info));
}

absl::Status CommandBuffer::End() {
  return VkResultToStatus(symbols_.vkEndCommandBuffer(command_buffer_));
}

absl::Status CommandBuffer::Reset() {
  // We don't release the resources when resetting the command buffer. The
  // assumption behind this is that the command buffer will be used in some sort
  // of benchmarking loop so each iteration/recording requires the same
  // resource.
  return VkResultToStatus(
      symbols_.vkResetCommandBuffer(command_buffer_, /*flags=*/0));
}

void CommandBuffer::CopyBuffer(const Buffer &src_buffer, size_t src_offset,
                               const Buffer &dst_buffer, size_t dst_offset,
                               size_t length) {
  VkBufferCopy region = {};
  region.srcOffset = src_offset;
  region.dstOffset = dst_offset;
  region.size = length;
  symbols_.vkCmdCopyBuffer(command_buffer_, src_buffer.buffer(),
                           dst_buffer.buffer(),
                           /*regionCount=*/1, &region);
}

void CommandBuffer::CopyBufferToImage(const Buffer &src_buffer,
                                      size_t src_offset, const Image &dst_image,
                                      VkExtent3D image_dimensions) {
  VkBufferImageCopy region = {};
  region.bufferOffset = src_offset;
  // Indicate the buffer is tightly packed
  region.bufferRowLength = region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = image_dimensions;

  symbols_.vkCmdCopyBufferToImage(command_buffer_, src_buffer.buffer(),
                                  dst_image.image(),
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  /*regionCount=*/1, &region);
}

void CommandBuffer::CopyImageToBuffer(const Image &src_image,
                                      VkExtent3D image_dimensions,
                                      const Buffer &dst_buffer,
                                      size_t dst_offset) {
  VkBufferImageCopy region = {};
  region.bufferOffset = dst_offset;
  // Indicate the buffer is tightly packed
  region.bufferRowLength = region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = image_dimensions;

  symbols_.vkCmdCopyImageToBuffer(command_buffer_, src_image.image(),
                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  dst_buffer.buffer(),
                                  /*regionCount=*/1, &region);
}

absl::Status CommandBuffer::TransitionImageLayout(const Image &image,
                                                  VkImageLayout from_layout,
                                                  VkImageLayout to_layout) {
  VkImageMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = from_layout;
  barrier.newLayout = to_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image.image();
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  VkPipelineStageFlags src_stage;
  VkPipelineStageFlags dst_stage;
  if (from_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
      to_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    // Uploading data to the image after creation
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (from_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             to_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    // Prepare shader image after uploading data
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  } else if (from_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
             to_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
    // Download data after shader usage
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    src_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else {
    return absl::UnimplementedError(absl::StrCat(
        "image layout transition from ", from_layout, " to ", to_layout));
  }

  symbols_.vkCmdPipelineBarrier(command_buffer_, src_stage, dst_stage, 0, 0,
                                nullptr, 0, nullptr, 1, &barrier);
  return absl::OkStatus();
}

void CommandBuffer::BindPipelineAndDescriptorSets(
    const Pipeline &pipeline,
    absl::Span<const BoundDescriptorSet> bound_descriptor_sets) {
  symbols_.vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                             pipeline.pipeline());

  for (const auto &descriptor_set : bound_descriptor_sets) {
    symbols_.vkCmdBindDescriptorSets(
        command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline.pipeline_layout(), descriptor_set.index,
        /*descriptorSetCount=*/1,
        /*pDescriptorSets=*/&descriptor_set.set,
        /*dynamicOffsetCount=*/0,
        /*pDynamicOffsets=*/nullptr);
  }
}

void CommandBuffer::ResetQueryPool(const TimestampQueryPool &query_pool) {
  symbols_.vkCmdResetQueryPool(command_buffer_, query_pool.query_pool(),
                               /*firstQuery=*/0,
                               /*queryCount=*/query_pool.query_count());
}

void CommandBuffer::WriteTimestamp(const TimestampQueryPool &query_pool,
                                   VkPipelineStageFlagBits pipeline_stage,
                                   uint32_t query_index) {
  symbols_.vkCmdWriteTimestamp(command_buffer_, pipeline_stage,
                               query_pool.query_pool(), query_index);
}

void CommandBuffer::Dispatch(uint32_t x, uint32_t y, uint32_t z) {
  symbols_.vkCmdDispatch(command_buffer_, x, y, z);
}

void CommandBuffer::DispatchBarrier() {
  VkMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  symbols_.vkCmdPipelineBarrier(command_buffer_,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                &barrier, 0, nullptr, 0, nullptr);
}

}  // namespace vulkan
}  // namespace uvkc
