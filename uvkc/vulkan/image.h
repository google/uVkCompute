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

#ifndef UVKC_VULKAN_IMAGE_H_
#define UVKC_VULKAN_IMAGE_H_

#include <vulkan/vulkan.h>

#include "absl/status/statusor.h"
#include "uvkc/vulkan/dynamic_symbols.h"

namespace uvkc {
namespace vulkan {

// A class representing a Vulkan image.
//
// This is just a simple wrapper around VkImage, its view, and its backing
// memory. It handles resource release at object destruction time.
class Image {
 public:
  // Wraps a Vulkan |image| and its backing |memory| from |device| and manages
  // deallocation of the |memory| and freeing of the |image|.
  Image(VkDevice device, VkDeviceMemory memory, VkImage image,
        VkImageView image_view, const DynamicSymbols &symbols);

  ~Image();

  // Returns the VkBuffer handle.
  VkImage image() const;

  VkImageView image_view() const;

 private:
  VkImage image_;

  VkImageView image_view_;

  VkDevice device_;
  VkDeviceMemory memory_;

  const DynamicSymbols &symbols_;
};

// A class representing a Vulkan sampler.
//
// This is just a simple wrapper around VkSampler. It handles resource release
// at object destruction time.
class Sampler {
 public:
  Sampler(VkDevice device, VkSampler sampler, const DynamicSymbols &symbols);

  ~Sampler();

  VkSampler sampler() const;

 private:
  VkSampler sampler_;

  VkDevice device_;

  const DynamicSymbols &symbols_;
};

}  // namespace vulkan
}  // namespace uvkc

#endif  // UVKC_VULKAN_IMAGE_H_
