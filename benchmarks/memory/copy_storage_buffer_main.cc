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

#include "benchmarks/memory/copy_storage_buffer.h"
#include "uvkc/benchmark/main.h"
#include "uvkc/vulkan/device.h"

static const char kBenchmarkName[] = "copy_storage_buffer";

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
  const char *gpu_name = physical_device.v10_properties.deviceName;

  for (int shift = 20; shift < 26; ++shift) {  // Number of bytes: 1M -> 32M
    int num_bytes = 1 << shift;
    for (auto element_type :
         {StorageBufferElementType::Float, StorageBufferElementType::Float4}) {
      double avg_latency_seconds = 0;
      RegisterCopyStorageBufferBenchmark(
          gpu_name, device, num_bytes, element_type, latency_measure->mode,
          &latency_measure->overhead_seconds, &avg_latency_seconds);
    }
  }
}

}  // namespace benchmark
}  // namespace uvkc
