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

#ifndef UVKC_VULKAN_PIPELINE_H_
#define UVKC_VULKAN_PIPELINE_H_

#include <vulkan/vulkan.h>

#include <memory>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "uvkc/vulkan/dynamic_symbols.h"
#include "uvkc/vulkan/shader_module.h"

namespace uvkc {
namespace vulkan {

// A class representing a Vulkan compute pipeline.
class Pipeline {
 public:
  // A struct representing a specialization constant.
  struct SpecConstant {
    uint32_t id;
    enum class Type {
      s32,
      u32,
      f32,
    } type;
    union {
      int32_t s32;
      uint32_t u32;
      float f32;
    } value;

    size_t size() const;
  };

  // Creates a Vulkan compute pipeline using the given |entry_point| in the
  // |shader_module|, with the provided |spec_constants|.
  static absl::StatusOr<std::unique_ptr<Pipeline>> Create(
      VkDevice device, const ShaderModule &shader_module,
      const char *entry_point, absl::Span<SpecConstant> spec_constants,
      const DynamicSymbols &symbols);

  ~Pipeline();

  // Returns the VkPipeline handle.
  VkPipeline pipeline() const;

  // Returns the VkPipelineLayout used for this pipeline.
  VkPipelineLayout pipeline_layout() const;

 private:
  Pipeline(VkPipeline pipeline, VkDevice device, VkPipelineLayout layout,
           const DynamicSymbols &symbols);

  VkPipeline pipeline_;

  VkDevice device_;
  VkPipelineLayout pipeline_layout_;

  const DynamicSymbols &symbols_;
};

}  // namespace vulkan
}  // namespace uvkc

#endif  // UVKC_VULKAN_PIPELINE_H_
