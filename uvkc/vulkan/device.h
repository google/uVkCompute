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

#ifndef UVKC_VULKAN_DEVICE_H_
#define UVKC_VULKAN_DEVICE_H_

#include <vulkan/vulkan.h>

#include <memory>
#include <unordered_map>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "uvkc/vulkan/buffer.h"
#include "uvkc/vulkan/command_buffer.h"
#include "uvkc/vulkan/descriptor_pool.h"
#include "uvkc/vulkan/dynamic_symbols.h"
#include "uvkc/vulkan/image.h"
#include "uvkc/vulkan/pipeline.h"
#include "uvkc/vulkan/shader_module.h"
#include "uvkc/vulkan/timestamp_query_pool.h"

namespace uvkc {
namespace vulkan {

// A class representing a Vulkan logical device.
//
// This is the main interaction points with the Vulkan system. All resource
// creation and workload dispatching are expected to happen with this class.
//
// Command buffers allocated from this device can be returned back to the pool
// individually.
class Device {
 public:
  // Wraps a logical |device| from |physical_device| of |queue_family_index|.
  static absl::StatusOr<std::unique_ptr<Device>> Create(
      VkPhysicalDevice physical_device, uint32_t queue_family_index,
      uint32_t valid_timestamp_bits, uint32_t nanoseconds_per_timestamp_value,
      VkDevice device, const DynamicSymbols &symbols);

  ~Device();

  // Creates a buffer of |size_in_bytes| for the specified usage as indicated by
  // |usage_flags| and memory properties as indicated in |memory_flags|.
  absl::StatusOr<std::unique_ptr<Buffer>> CreateBuffer(
      VkBufferUsageFlags usage_flags, VkMemoryPropertyFlags memory_flags,
      VkDeviceSize size_in_bytes);

  // Creates an image for the specified usage as indicated by |usage_flags| and
  // memory properties as indicated in |memory_flags|.
  absl::StatusOr<std::unique_ptr<Image>> CreateImage(
      VkImageUsageFlags usage_flags, VkMemoryPropertyFlags memory_flags,
      VkImageType image_type, VkFormat image_format, VkExtent3D dimensions,
      VkImageTiling image_tiling, VkImageViewType view_type);

  // Creates a sampler that performs nearest filtering and clipping to edge for
  // U/V/W coordinate. The sampler does not supprot anisotropic filtering and
  // comparison.
  absl::StatusOr<std::unique_ptr<Sampler>> CreateSampler();

  // Creates a shader module from the SPIR-V code starting at |spirv_data| and
  // of |spirv_size| 32-bit integers.
  absl::StatusOr<std::unique_ptr<ShaderModule>> CreateShaderModule(
      const uint32_t *spirv_data, size_t spirv_size);

  // Creates a compute pipeline calling |entry_point| in the given
  // |shader_module| and specializes the pipeline with |spec_constants|.
  absl::StatusOr<std::unique_ptr<Pipeline>> CreatePipeline(
      const ShaderModule &shader_module, const char *entry_point,
      absl::Span<Pipeline::SpecConstant> spec_constants);

  // Creates a descriptor pool with enough resources matching the pipeline
  // layout of the given |shader_module|.
  absl::StatusOr<std::unique_ptr<DescriptorPool>> CreateDescriptorPool(
      const ShaderModule &shader_module);

  // A |buffer| and its bound descriptor |set| and binding| numbers.
  struct BoundBuffer {
    const Buffer *buffer;
    uint32_t set;
    uint32_t binding;
  };

  // Attaches buffers to descriptors for use in dispatching the given
  // |shader_module|.
  absl::Status AttachBufferToDescriptor(
      const ShaderModule &shader_module,
      const std::unordered_map<VkDescriptorSetLayout, VkDescriptorSet>
          &layout_set_map,
      absl::Span<const BoundBuffer> bound_buffers);

  // An |image| and its bound descriptor |set| and binding| numbers.
  struct BoundImage {
    const Image *image;
    const Sampler *sampler;
    uint32_t set;
    uint32_t binding;
  };

  // Attaches images to descriptors for use in dispatching the given
  // |shader_module|.
  absl::Status AttachImageToDescriptor(
      const ShaderModule &shader_module,
      const std::unordered_map<VkDescriptorSetLayout, VkDescriptorSet>
          &layout_set_map,
      absl::Span<const BoundImage> bound_images);

  // Allocates a primary command buffer.
  absl::StatusOr<std::unique_ptr<CommandBuffer>> AllocateCommandBuffer();

  // Resets the command pool and recycles all the sources from all the command
  // buffers allocated from this device thus far.
  absl::Status ResetCommandPool();

  // Creates a query pool for managing |query_count| timestamp queries.
  absl::StatusOr<std::unique_ptr<TimestampQueryPool>> CreateTimestampQueryPool(
      uint32_t query_count);

  // Submits the given |command_buffer| to the queue.
  absl::Status QueueSubmitAndWait(const CommandBuffer &command_buffer);

 private:
  Device(VkDevice device, VkPhysicalDevice physical_device,
         uint32_t queue_family_index, uint32_t valid_timestamp_bits,
         uint32_t nanoseconds_per_timestamp_value, VkCommandPool command_pool,
         const DynamicSymbols &symbols);

  // Selects a memory type among |supported_memory_types| that statisfies
  // |desired_memory_properties| and returns its array index in
  // VkPhysicalDeviceMemoryProperties.
  absl::StatusOr<uint32_t> SelectMemoryType(
      uint32_t supported_memory_types,
      VkMemoryPropertyFlags desired_memory_properties);

  // Allocates Vulkan memory with the given |memory_flags| according to
  // |memory_requirements|.
  absl::StatusOr<VkDeviceMemory> AllocateMemory(
      VkMemoryRequirements memory_requirements,
      VkMemoryPropertyFlags memory_flags);

  VkDevice device_;

  VkPhysicalDevice physical_device_;
  VkPhysicalDeviceMemoryProperties memory_properties_;

  VkQueue queue_;
  uint32_t queue_family_index_;
  uint32_t valid_timestamp_bits_;
  uint32_t nanoseconds_per_timestamp_value_;

  VkCommandPool command_pool_;

  const DynamicSymbols &symbols_;
};

}  // namespace vulkan
}  // namespace uvkc

#endif  // UVKC_VULKAN_DEVICE_H_
