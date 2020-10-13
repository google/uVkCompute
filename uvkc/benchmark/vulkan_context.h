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

#ifndef UVKC_BENCHMARK_VULKAN_CONTEXT_H_
#define UVKC_BENCHMARK_VULKAN_CONTEXT_H_

#include <memory>

#include "absl/status/statusor.h"
#include "uvkc/vulkan/device.h"
#include "uvkc/vulkan/driver.h"
#include "uvkc/vulkan/dynamic_symbols.h"

namespace uvkc {
namespace benchmark {

enum class LatencyMeasureMode {
  // time spent from queue submit to returning from queue wait
  kSystemSubmit,
  // system_submit subtracted by time for void dispatch
  kSystemDispatch,
  // Timestamp difference measured on GPU
  kGpuTimestamp,
};

struct LatencyMeasure {
  LatencyMeasureMode mode;
  double overhead_seconds;
};

// A struct for holding the Vulkan application context for benchmarks.
//
// This struct is meant to meant to contain Vulkan object handles that share
// among multiple benchmarks, for example, the Vulkan driver and device.
//
// TODO: add a way to support benchmark-specific data.
struct VulkanContext {
  std::unique_ptr<vulkan::DynamicSymbols> symbols;
  std::unique_ptr<vulkan::Driver> driver;

  std::vector<vulkan::Driver::PhysicalDeviceInfo> physical_devices;
  std::vector<std::unique_ptr<vulkan::Device>> devices;

  LatencyMeasure latency_measure;

  VulkanContext(
      std::unique_ptr<vulkan::DynamicSymbols> symbols,
      std::unique_ptr<vulkan::Driver> driver,
      std::vector<vulkan::Driver::PhysicalDeviceInfo> physical_devices,
      std::vector<std::unique_ptr<vulkan::Device>> devices);
};

// Creates the default Vulkan application context where we create a logical
// device with one compute queue for each available physical device.
absl::StatusOr<std::unique_ptr<VulkanContext>> CreateDefaultVulkanContext(
    const char *app_name);

}  // namespace benchmark
}  // namespace uvkc

#endif  // UVKC_BENCHMARK_VULKAN_CONTEXT_H_
