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

#ifndef UVKC_VULKAN_DESCRIPTOR_POOL_H_
#define UVKC_VULKAN_DESCRIPTOR_POOL_H_

#include <vulkan/vulkan.h>

#include <memory>
#include <unordered_map>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "uvkc/vulkan/dynamic_symbols.h"

namespace uvkc {
namespace vulkan {

// A class representing a Vulkan descriptor pool.
//
// Individual descriptors allocated from this pool cannot be returned alone; the
// pool is expected to be reset as a whole.
class DescriptorPool {
 public:
  // Creates a descriptor pool allowing |max_sets| and maximal number of
  // descriptors for each descriptor type as specified in |descriptor_counts|
  // from |device|.
  static absl::StatusOr<std::unique_ptr<DescriptorPool>> Create(
      VkDevice device, uint32_t max_sets,
      absl::Span<VkDescriptorPoolSize> descriptor_counts,
      const DynamicSymbols &symbols);

  ~DescriptorPool();

  // Allocates descriptor sets following the given |set_layouts| and returns the
  // mapping from the layout to the concrete set object.
  absl::StatusOr<std::unordered_map<VkDescriptorSetLayout, VkDescriptorSet>>
  AllocateDescriptorSets(absl::Span<const VkDescriptorSetLayout> set_layouts);

 private:
  DescriptorPool(VkDescriptorPool pool, VkDevice device,
                 const DynamicSymbols &symbols);

  VkDescriptorPool pool_;

  VkDevice device_;

  const DynamicSymbols &symbols_;
};

}  // namespace vulkan
}  // namespace uvkc

#endif  // UVKC_VULKAN_DESCRIPTOR_POOL_H_
