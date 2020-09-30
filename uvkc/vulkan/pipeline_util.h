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

#ifndef UVKC_VULKAN_PIPELINE_UTIL_H_
#define UVKC_VULKAN_PIPELINE_UTIL_H_

#include <vulkan/vulkan.h>

#include <vector>

#include "absl/status/statusor.h"

namespace uvkc {
namespace vulkan {

// A struct describing Vulkan pipeline layout
struct PipelineLayout {
  struct DescriptorSetLayout {
    uint32_t set_number;
    VkDescriptorSetLayoutCreateInfo create_info;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
  };

  std::vector<DescriptorSetLayout> set_layouts;
};

// Reflects on the SPIR-V code starting at |spirv_data| with |spirv_size|
// 32-bit integers and returns the pipeline layout information.
absl::StatusOr<PipelineLayout> ReflectSpirvPipelineLayout(
    const uint32_t *spirv_data, size_t spirv_size);

}  // namespace vulkan
}  // namespace uvkc

#endif  // UVKC_VULKAN_PIPELINE_UTIL_H_
