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

#include "uvkc/vulkan/pipeline_util.h"

#include "spirv_reflect.h"

namespace uvkc {
namespace vulkan {

absl::StatusOr<PipelineLayout> ReflectSpirvPipelineLayout(
    const uint32_t *spirv_data, size_t spirv_size) {
  SpvReflectShaderModule module = {};
  SpvReflectResult result = spvReflectCreateShaderModule(
      spirv_size * sizeof(uint32_t), spirv_data, &module);
  if (result != SPV_REFLECT_RESULT_SUCCESS) {
    return absl::InternalError("failed to reflect on SPIR-V binary module");
  }

  uint32_t count = 0;
  result = spvReflectEnumerateDescriptorSets(&module, &count, nullptr);
  if (result != SPV_REFLECT_RESULT_SUCCESS) {
    return absl::InternalError("failed to enumerate descriptor sets");
  }

  std::vector<SpvReflectDescriptorSet *> sets(count);
  result = spvReflectEnumerateDescriptorSets(&module, &count, sets.data());
  if (result != SPV_REFLECT_RESULT_SUCCESS) {
    return absl::InternalError("failed to enumerate descriptor sets");
  }

  std::vector<PipelineLayout::DescriptorSetLayout> set_layouts(sets.size());
  for (int set_index = 0; set_index < sets.size(); ++set_index) {
    const SpvReflectDescriptorSet &set_reflection = *(sets[set_index]);

    PipelineLayout::DescriptorSetLayout &layout = set_layouts[set_index];
    layout.set_number = set_reflection.set;
    layout.create_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout.create_info.pNext = nullptr;
    layout.create_info.flags = 0;

    layout.bindings.resize(set_reflection.binding_count);
    for (int binding_index = 0; binding_index < set_reflection.binding_count;
         ++binding_index) {
      const SpvReflectDescriptorBinding &binding_reflection =
          *(set_reflection.bindings[binding_index]);

      VkDescriptorSetLayoutBinding &layout_binding =
          layout.bindings[binding_index];
      layout_binding.binding = binding_reflection.binding;
      layout_binding.descriptorType =
          static_cast<VkDescriptorType>(binding_reflection.descriptor_type);
      layout_binding.descriptorCount = 1;
      for (int dim_index = 0; dim_index < binding_reflection.array.dims_count;
           ++dim_index) {
        layout_binding.descriptorCount *=
            binding_reflection.array.dims[dim_index];
      }
      layout_binding.stageFlags =
          static_cast<VkShaderStageFlagBits>(module.shader_stage);
      layout_binding.pImmutableSamplers = nullptr;
    }

    layout.create_info.bindingCount = set_reflection.binding_count;
    layout.create_info.pBindings = layout.bindings.data();
  }

  spvReflectDestroyShaderModule(&module);

  return PipelineLayout{std::move(set_layouts)};
}

}  // namespace vulkan
}  // namespace uvkc
