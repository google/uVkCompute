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

#include "uvkc/vulkan/shader_module.h"

#include <unordered_map>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "uvkc/vulkan/status_util.h"

namespace uvkc {
namespace vulkan {

absl::StatusOr<std::unique_ptr<ShaderModule>> ShaderModule::Create(
    VkDevice device, const uint32_t *spirv_data, size_t spirv_size,
    const DynamicSymbols &symbols) {
  // Create the VkShaderModule object for the given SPIR-V code
  VkShaderModuleCreateInfo module_create_info = {};
  module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  module_create_info.pNext = nullptr;
  module_create_info.flags = 0;
  module_create_info.codeSize = spirv_size * sizeof(uint32_t);
  module_create_info.pCode = spirv_data;

  VkShaderModule shader_module = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(symbols.vkCreateShaderModule(device, &module_create_info,
                                                  /*pAllocator=*/nullptr,
                                                  &shader_module));

  // Reflect on the SPIR-V code to get pipeline layout information
  UVKC_ASSIGN_OR_RETURN(PipelineLayout pipeline_layout,
                        ReflectSpirvPipelineLayout(spirv_data, spirv_size));

  // Create all VkDescriptorSetLayout objects
  size_t num_sets = pipeline_layout.set_layouts.size();
  std::vector<VkDescriptorSetLayout> vk_set_layout(num_sets);
  for (int i = 0; i < num_sets; ++i) {
    VK_RETURN_IF_ERROR(symbols.vkCreateDescriptorSetLayout(
        device, &pipeline_layout.set_layouts[i].create_info,
        /*pAllocator=*/nullptr, &vk_set_layout[i]));
  }

  return absl::WrapUnique(
      new ShaderModule(shader_module, device, std::move(vk_set_layout),
                       std::move(pipeline_layout), symbols));
}

ShaderModule::~ShaderModule() {
  for (const auto &set_layout : vk_set_layouts_) {
    symbols_.vkDestroyDescriptorSetLayout(device_, set_layout,
                                          /*pAllocator=*/nullptr);
  }
  symbols_.vkDestroyShaderModule(device_, shader_module_,
                                 /*pAllocator=*/nullptr);
}

VkShaderModule ShaderModule::shader_module() const { return shader_module_; }

uint32_t ShaderModule::num_sets() const { return vk_set_layouts_.size(); }

absl::Span<const VkDescriptorSetLayout> ShaderModule::descriptor_set_layouts()
    const {
  return {vk_set_layouts_.data(), vk_set_layouts_.size()};
}

absl::StatusOr<VkDescriptorSetLayout> ShaderModule::GetDescriptorSetLayout(
    uint32_t set) const {
  for (int i = 0; i < vk_set_layouts_.size(); ++i) {
    if (pipeline_layout_.set_layouts[i].set_number == set)
      return vk_set_layouts_[i];
  }
  return absl::InvalidArgumentError(
      absl::StrCat("cannot find set layout object for set #", set));
}

std::unordered_map<uint32_t, VkDescriptorSetLayout>
ShaderModule::GetDescriptorSetLayoutMap() const {
  std::unordered_map<uint32_t, VkDescriptorSetLayout> layout_map;
  for (int i = 0; i < vk_set_layouts_.size(); ++i) {
    layout_map[pipeline_layout_.set_layouts[i].set_number] = vk_set_layouts_[i];
  }
  return layout_map;
}

absl::StatusOr<const VkDescriptorSetLayoutBinding *>
ShaderModule::GetDescriptorSetLayoutBinding(uint32_t set,
                                            uint32_t binding) const {
  for (const auto &set_layout : pipeline_layout_.set_layouts) {
    if (set_layout.set_number != set) continue;
    for (const auto &set_binding : set_layout.bindings) {
      if (set_binding.binding == binding) return &set_binding;
    }
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "cannot find binding info for set #", set, " and binding #", binding));
}

std::vector<VkDescriptorPoolSize> ShaderModule::CalculateDescriptorPoolSize()
    const {
  std::unordered_map<VkDescriptorType, uint32_t> descriptor_counts;
  for (const auto &set_layout : pipeline_layout_.set_layouts) {
    for (const auto &binding : set_layout.bindings) {
      descriptor_counts[binding.descriptorType] += binding.descriptorCount;
    }
  }

  std::vector<VkDescriptorPoolSize> pool_sizes(descriptor_counts.size());
  size_t index = 0;
  for (const auto &descriptor_count : descriptor_counts) {
    pool_sizes[index].type = descriptor_count.first;
    pool_sizes[index].descriptorCount = descriptor_count.second;
    ++index;
  }

  return pool_sizes;
}

ShaderModule::ShaderModule(VkShaderModule module, VkDevice device,
                           std::vector<VkDescriptorSetLayout> vk_set_layouts,
                           PipelineLayout pipeline_layout,
                           const DynamicSymbols &symbols)
    : shader_module_(module),
      device_(device),
      vk_set_layouts_(std::move(vk_set_layouts)),
      pipeline_layout_(std::move(pipeline_layout)),
      symbols_(symbols) {}

}  // namespace vulkan
}  // namespace uvkc
