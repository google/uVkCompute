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

#ifndef UVKC_VULKAN_DRIVER_H_
#define UVKC_VULKAN_DRIVER_H_

#include <vulkan/vulkan.h>

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "uvkc/vulkan/device.h"
#include "uvkc/vulkan/dynamic_symbols.h"

namespace uvkc {
namespace vulkan {

// A class representing a Vulkan driver.
//
// This class is the beginning of all interaction with the Vulkan system. It
// handles VkInstance creation, VkPhysicalDevice enumeration, and VkDevice
// creation.
class Driver {
 public:
  // Creates a Vulkan driver for an application with the given |app_name|.
  static absl::StatusOr<std::unique_ptr<Driver>> Create(
      const char *app_name, DynamicSymbols *symbols);

  ~Driver();

  struct PhysicalDeviceInfo {
    VkPhysicalDevice handle;
    VkPhysicalDeviceProperties v10_properties;
    VkPhysicalDeviceSubgroupProperties subgroup_properties;
  };

  // Enumerates all available physical devices on system.
  absl::StatusOr<std::vector<PhysicalDeviceInfo>> EnumeratePhysicalDevices();

  // Creates a logical device from the given |physical_device| with the ability
  // to use a queue of the given |queue_flags|.
  absl::StatusOr<std::unique_ptr<Device>> CreateDevice(
      const PhysicalDeviceInfo &physical_device, VkQueueFlags queue_flags);

 private:
  explicit Driver(VkInstance instance, const DynamicSymbols &symbols);

  VkInstance instance_;

  const DynamicSymbols &symbols_;
};

}  // namespace vulkan
}  // namespace uvkc

#endif  // UVKC_VULKAN_DRIVER_H_
