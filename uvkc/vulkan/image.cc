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

#include "uvkc/vulkan/image.h"

#include "absl/status/status.h"
#include "uvkc/vulkan/status_util.h"

namespace uvkc {
namespace vulkan {

Image::Image(VkDevice device, VkDeviceMemory memory, VkImage image,
             VkImageView image_view, const DynamicSymbols &symbols)
    : image_(image),
      image_view_(image_view),
      device_(device),
      memory_(memory),
      symbols_(symbols) {}

Image::~Image() {
  symbols_.vkDestroyImageView(device_, image_view_, /*pAllocator=*/nullptr);
  symbols_.vkDestroyImage(device_, image_, /*pAllocator=*/nullptr);
  symbols_.vkFreeMemory(device_, memory_, /*pAllocator=*/nullptr);
}

VkImage Image::image() const { return image_; }

VkImageView Image::image_view() const { return image_view_; }

Sampler::Sampler(VkDevice device, VkSampler sampler,
                 const DynamicSymbols &symbols)
    : sampler_(sampler), device_(device), symbols_(symbols) {}

Sampler::~Sampler() {
  symbols_.vkDestroySampler(device_, sampler_, /*pAllocator=*/nullptr);
}

VkSampler Sampler::sampler() const { return sampler_; }

}  // namespace vulkan
}  // namespace uvkc
