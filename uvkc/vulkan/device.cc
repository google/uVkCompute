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

#include "uvkc/vulkan/device.h"

#include <cstdint>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "uvkc/base/status.h"
#include "uvkc/vulkan/status_util.h"

namespace uvkc {
namespace vulkan {

absl::StatusOr<std::unique_ptr<Device>> Device::Create(
    VkPhysicalDevice physical_device, uint32_t queue_family_index,
    VkDevice device) {
  VkCommandPoolCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  create_info.queueFamilyIndex = queue_family_index;

  VkCommandPool command_pool = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(vkCreateCommandPool(
      device, &create_info, /*pAllocator=*/nullptr, &command_pool));

  return absl::WrapUnique(
      new Device(device, physical_device, queue_family_index, command_pool));
}

Device::~Device() {
  vkDeviceWaitIdle(device_);
  vkDestroyCommandPool(device_, command_pool_, /*pAllocator=*/nullptr);
  vkDestroyDevice(device_, /*pAllocator=*/nullptr);
}

absl::StatusOr<std::unique_ptr<Buffer>> Device::CreateBuffer(
    VkBufferUsageFlags usage_flags, VkMemoryPropertyFlags memory_flags,
    VkDeviceSize size_in_bytes) {
  VkBufferCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  create_info.size = size_in_bytes;
  create_info.usage = usage_flags;
  create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkBuffer buffer = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(
      vkCreateBuffer(device_, &create_info, /*pAllocator=*/nullptr, &buffer));

  // Get memory requirements for the buffer
  VkMemoryRequirements memory_requirements;
  vkGetBufferMemoryRequirements(device_, buffer, &memory_requirements);

  // Allocate memory for the buffer
  VkMemoryAllocateInfo allocate_info = {};
  allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate_info.pNext = nullptr;
  allocate_info.allocationSize = memory_requirements.size;
  UVKC_ASSIGN_OR_RETURN(
      allocate_info.memoryTypeIndex,
      SelectMemoryType(memory_requirements.memoryTypeBits, memory_flags));

  VkDeviceMemory memory = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(vkAllocateMemory(device_, &allocate_info,
                                      /*pAlloator=*/nullptr, &memory));

  // Bind the memory to the buffer
  VK_RETURN_IF_ERROR(
      vkBindBufferMemory(device_, buffer, memory, /*memoryOffset=*/0));

  return std::make_unique<Buffer>(device_, memory, buffer);
}

absl::StatusOr<std::unique_ptr<ShaderModule>> Device::CreateShaderModule(
    const uint32_t *spirv_data, size_t spirv_size) {
  return ShaderModule::Create(device_, spirv_data, spirv_size);
}

absl::StatusOr<std::unique_ptr<Pipeline>> Device::CreatePipeline(
    const ShaderModule &shader_module, const char *entry_point,
    absl::Span<Pipeline::SpecConstant> spec_constants) {
  return Pipeline::Create(device_, shader_module, entry_point, spec_constants);
}

absl::StatusOr<std::unique_ptr<DescriptorPool>> Device::CreateDescriptorPool(
    const ShaderModule &shader_module) {
  auto pool_sizes = shader_module.CalculateDescriptorPoolSize();
  return DescriptorPool::Create(device_, shader_module.num_sets(),
                                {pool_sizes.data(), pool_sizes.size()});
}

absl::Status Device::AttachBufferToDescriptor(
    const ShaderModule &shader_module,
    const std::unordered_map<VkDescriptorSetLayout, VkDescriptorSet>
        &layout_set_map,
    absl::Span<const Device::BoundBuffer> bound_buffers) {
  std::vector<VkDescriptorBufferInfo> buffer_infos(bound_buffers.size());
  std::vector<VkWriteDescriptorSet> write_sets(bound_buffers.size());
  for (int i = 0; i < bound_buffers.size(); ++i) {
    const auto &descriptor = bound_buffers[i];
    auto &info = buffer_infos[i];
    auto &write = write_sets[i];

    info.buffer = descriptor.buffer->buffer();
    info.offset = 0;
    info.range = VK_WHOLE_SIZE;

    UVKC_ASSIGN_OR_RETURN(VkDescriptorSetLayout set_layout,
                          shader_module.GetDescriptorSetLayout(descriptor.set));
    UVKC_ASSIGN_OR_RETURN(const auto *binding_info,
                          shader_module.GetDescriptorSetLayoutBinding(
                              descriptor.set, descriptor.binding));

    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.pNext = nullptr;
    write.dstSet = layout_set_map.at(set_layout);
    write.dstBinding = descriptor.binding;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = binding_info->descriptorType;
    write.pImageInfo = nullptr;
    write.pBufferInfo = &info;
    write.pTexelBufferView = nullptr;
  }

  vkUpdateDescriptorSets(device_, write_sets.size(), write_sets.data(),
                         /*descriptorCopyCount=*/0,
                         /*pDescriptorCopies=*/nullptr);
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<CommandBuffer>> Device::AllocateCommandBuffer() {
  VkCommandBufferAllocateInfo allocate_info = {};
  allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocate_info.pNext = nullptr;
  allocate_info.commandPool = command_pool_;
  allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate_info.commandBufferCount = 1;

  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(
      vkAllocateCommandBuffers(device_, &allocate_info, &command_buffer));
  return std::make_unique<CommandBuffer>(device_, command_buffer);
}

absl::Status Device::QueueSubmitAndWait(const CommandBuffer &command_buffer) {
  VkFenceCreateInfo fence_create_info = {};
  fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_create_info.pNext = nullptr;
  fence_create_info.flags = 0;

  VkFence fence = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(vkCreateFence(device_, &fence_create_info,
                                   /*pALlocator=*/nullptr, &fence));

  VkCommandBuffer cmdbuf = command_buffer.command_buffer();
  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cmdbuf;

  VK_RETURN_IF_ERROR(vkQueueSubmit(queue_, 1, &submit_info, fence));

  VK_RETURN_IF_ERROR(vkWaitForFences(device_, /*fenceCount=*/1, &fence,
                                     /*waitAll=*/true, /*timeout=*/UINT64_MAX));

  vkDestroyFence(device_, fence, /*pAllocator=*/nullptr);
  return absl::OkStatus();
}

Device::Device(VkDevice device, VkPhysicalDevice physical_device,
               uint32_t queue_family_index, VkCommandPool command_pool)
    : device_(device),
      physical_device_(physical_device),
      memory_properties_(),
      queue_(VK_NULL_HANDLE),
      queue_family_index_(queue_family_index),
      command_pool_(command_pool) {
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties_);
  vkGetDeviceQueue(device_, queue_family_index_, 0, &queue_);
}

absl::StatusOr<uint32_t> Device::SelectMemoryType(
    uint32_t supported_memory_types,
    VkMemoryPropertyFlags desired_memory_properties) {
  for (int i = 0; i < memory_properties_.memoryTypeCount; ++i) {
    if ((supported_memory_types & (1 << i)) &&
        ((memory_properties_.memoryTypes[i].propertyFlags &
          desired_memory_properties) == desired_memory_properties))
      return i;
  }
  return absl::UnavailableError("cannot find memory type with required bits");
}

}  // namespace vulkan
}  // namespace uvkc
