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

#include "uvkc/benchmark/vulkan_context.h"

#include "uvkc/base/status.h"

namespace uvkc {
namespace benchmark {

VulkanContext::VulkanContext(
    std::unique_ptr<vulkan::DynamicSymbols> symbols,
    std::unique_ptr<vulkan::Driver> driver,
    std::vector<vulkan::Driver::PhysicalDeviceInfo> physical_devices,
    std::vector<std::unique_ptr<vulkan::Device>> devices)
    : symbols(std::move(symbols)),
      driver(std::move(driver)),
      physical_devices(std::move(physical_devices)),
      devices(std::move(devices)),
      latency_measure({LatencyMeasureMode::kSystemSubmit, 0.}) {}

absl::StatusOr<std::unique_ptr<VulkanContext>> CreateDefaultVulkanContext(
    const char *app_name) {
  UVKC_ASSIGN_OR_RETURN(auto symbols,
                        vulkan::DynamicSymbols::CreateFromSystemLoader());
  UVKC_ASSIGN_OR_RETURN(auto driver,
                        vulkan::Driver::Create(app_name, symbols.get()));
  UVKC_ASSIGN_OR_RETURN(auto physical_devices,
                        driver->EnumeratePhysicalDevices());

  std::vector<std::unique_ptr<vulkan::Device>> devices;
  for (const auto &physical_device : physical_devices) {
    UVKC_ASSIGN_OR_RETURN(
        auto device,
        driver->CreateDevice(physical_device, VK_QUEUE_COMPUTE_BIT));
    devices.push_back(std::move(device));
  }
  return std::make_unique<VulkanContext>(std::move(symbols), std::move(driver),
                                         std::move(physical_devices),
                                         std::move(devices));
}

}  // namespace benchmark
}  // namespace uvkc
