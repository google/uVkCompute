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

#include <memory>

#include "benchmark/benchmark.h"
#include "uvkc/benchmark/dispatch_void_shader.h"
#include "uvkc/benchmark/main.h"
#include "uvkc/benchmark/status_util.h"
#include "uvkc/benchmark/vulkan_context.h"
#include "uvkc/vulkan/device.h"

static const char kBenchmarkName[] = "dispatch_void_shader";

namespace uvkc {
namespace benchmark {

absl::StatusOr<std::unique_ptr<VulkanContext>> CreateVulkanContext() {
  return CreateDefaultVulkanContext(kBenchmarkName);
}

bool RegisterVulkanOverheadBenchmark(
    const vulkan::Driver::PhysicalDeviceInfo &physical_device,
    vulkan::Device *device, double *overhead_seconds) {
  return false;
}

void RegisterVulkanBenchmarks(
    const vulkan::Driver::PhysicalDeviceInfo &physical_device,
    vulkan::Device *device, const LatencyMeasure *latency_measure) {
  BM_CHECK_EQ(latency_measure->mode,
              ::uvkc::benchmark::LatencyMeasureMode::kSystemSubmit)
      << kBenchmarkName << " only supports system_submit latency measure mode";

  double void_dispatch_latency_seconds = 0;
  const char *gpu_name = physical_device.v10_properties.deviceName;
  RegisterDispatchVoidShaderBenchmark(gpu_name, device,
                                      &void_dispatch_latency_seconds);
}

}  // namespace benchmark
}  // namespace uvkc
