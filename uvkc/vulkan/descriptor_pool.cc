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

#include "uvkc/vulkan/descriptor_pool.h"

#include <vector>

#include "absl/memory/memory.h"
#include "uvkc/vulkan/status_util.h"

namespace uvkc {
namespace vulkan {

absl::StatusOr<std::unique_ptr<DescriptorPool>> DescriptorPool::Create(
    VkDevice device, uint32_t max_sets,
    absl::Span<VkDescriptorPoolSize> descriptor_counts,
    const DynamicSymbols &symbols) {
  VkDescriptorPoolCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  create_info.maxSets = max_sets;
  create_info.poolSizeCount = descriptor_counts.size();
  create_info.pPoolSizes = descriptor_counts.data();

  VkDescriptorPool pool = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(symbols.vkCreateDescriptorPool(device, &create_info,
                                                    /*pAllocator=*/nullptr,
                                                    &pool));

  return absl::WrapUnique(new DescriptorPool(pool, device, symbols));
}

DescriptorPool::~DescriptorPool() {
  symbols_.vkDestroyDescriptorPool(device_, pool_, /*pALlocator=*/nullptr);
}

absl::StatusOr<std::unordered_map<VkDescriptorSetLayout, VkDescriptorSet>>
DescriptorPool::AllocateDescriptorSets(
    absl::Span<const VkDescriptorSetLayout> set_layouts) {
  VkDescriptorSetAllocateInfo allocate_info = {};
  allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocate_info.pNext = nullptr;
  allocate_info.descriptorPool = pool_;
  allocate_info.descriptorSetCount = set_layouts.size();
  allocate_info.pSetLayouts = set_layouts.data();

  std::vector<VkDescriptorSet> sets(set_layouts.size());
  VK_RETURN_IF_ERROR(
      symbols_.vkAllocateDescriptorSets(device_, &allocate_info, sets.data()));
  std::unordered_map<VkDescriptorSetLayout, VkDescriptorSet> layout_set_map;
  for (int i = 0; i < set_layouts.size(); ++i) {
    layout_set_map[set_layouts[i]] = sets[i];
  }

  return layout_set_map;
}

DescriptorPool::DescriptorPool(VkDescriptorPool pool, VkDevice device,
                               const DynamicSymbols &symbols)
    : pool_(pool), device_(device), symbols_(symbols) {}

}  // namespace vulkan
}  // namespace uvkc
