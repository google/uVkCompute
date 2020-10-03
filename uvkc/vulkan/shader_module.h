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

#ifndef UVKC_VULKAN_SHADER_MODULE_H_
#define UVKC_VULKAN_SHADER_MODULE_H_

#include <vulkan/vulkan.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "uvkc/vulkan/dynamic_symbols.h"
#include "uvkc/vulkan/pipeline_util.h"

namespace uvkc {
namespace vulkan {

// A class representing a Vulkan shader module.
//
// The shader module is expected to only contain one entry point.
//
// In addition to creating the VkShaderModule object from the given SPIR-V code,
// this class also performs reflection over the SPIR-V code to understand the
// pipeline layout requirements.
class ShaderModule {
 public:
  // Creates a Vulkan shader module from SPIR-V code starting at |spirv_data|
  // with |spirv_size| 32-bit integers and creates descriptor set layout objects
  // for each descriptor set in the shader module.
  static absl::StatusOr<std::unique_ptr<ShaderModule>> Create(
      VkDevice device, const uint32_t *spirv_data, size_t spirv_size,
      const DynamicSymbols &symbols);

  ~ShaderModule();

  // Returns the VkShaderModule handle.
  VkShaderModule shader_module() const;

  // Returns the number of sets used in this shader module.
  uint32_t num_sets() const;

  // Returns all descriptor set layout objects for this shader module.
  absl::Span<const VkDescriptorSetLayout> descriptor_set_layouts() const;

  // Returns the VkDescriptorSetLayout for the given descriptor |set|.
  absl::StatusOr<VkDescriptorSetLayout> GetDescriptorSetLayout(
      uint32_t set) const;

  // Returns a map from descriptor set numbers to the corresponding layout
  // objects.
  std::unordered_map<uint32_t, VkDescriptorSetLayout>
  GetDescriptorSetLayoutMap() const;

  // Returns the VkDescriptorSetLayoutBinding for the given descriptor |set| and
  // |binding|.
  absl::StatusOr<const VkDescriptorSetLayoutBinding *>
  GetDescriptorSetLayoutBinding(uint32_t set, uint32_t binding) const;

  // Calculates minimal pool size requirements for each descriptor type used in
  // this shader module.
  std::vector<VkDescriptorPoolSize> CalculateDescriptorPoolSize() const;

 private:
  ShaderModule(VkShaderModule module, VkDevice device,
               std::vector<VkDescriptorSetLayout> vk_set_layouts,
               PipelineLayout pipeline_layout, const DynamicSymbols &symbols);

  VkShaderModule shader_module_;

  VkDevice device_;

  // Vulkan descriptor set layouts for all used descriptor sets in the shader
  // module. It matches 1:1 to the pipeline_layout_.set_layouts array.
  std::vector<VkDescriptorSetLayout> vk_set_layouts_;
  PipelineLayout pipeline_layout_;

  const DynamicSymbols &symbols_;
};

}  // namespace vulkan
}  // namespace uvkc

#endif  // UVKC_VULKAN_SHADER_MODULE_H_
