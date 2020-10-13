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

#ifndef UVKC_BENCHMARK_MAIN_H_
#define UVKC_BENCHMARK_MAIN_H_

#include "absl/status/statusor.h"
#include "uvkc/benchmark/vulkan_context.h"
#include "uvkc/vulkan/device.h"
#include "uvkc/vulkan/driver.h"

namespace uvkc {
namespace benchmark {

// Creates a Vulkan application context for the current benchmark binary.
//
// The context is expected to hold Vulkan objects that can be shared among
// multiple benchmarks, e.g., the Vulkan driver and device.
//
// The context will be created before running all benchmarks and it will persist
// during the lifetime of all benchmarks.
//
// Normally the benchmark just need to call CreateDefaultVulkanContext() with
// the proper Vulkan application name.
absl::StatusOr<std::unique_ptr<VulkanContext>> CreateVulkanContext();

// Registers a benchmark for evaluating the overhead that should be subtracted
// from the normal benchmark latency. Returns true if a benchmark is registered;
// returns false if to use the default overhead latency benchmark (that is, void
// shader dispatch).
//
// This is only used for LatencyMesaureMode::kSystemDispatch.
bool RegisterVulkanOverheadBenchmark(
    const vulkan::Driver::PhysicalDeviceInfo &physical_device,
    vulkan::Device *device, double *overhead_seconds);

// Registers all Vulkan benchmarks for the current benchmark binary.
//
// The |overhead_seconds| field in |latency_measure| should subtracted from the
// latency measured by the registered benchmarks for
// LatencyMeasureMode::kSystemDispatch.
void RegisterVulkanBenchmarks(
    const vulkan::Driver::PhysicalDeviceInfo &physical_device,
    vulkan::Device *device,
    // Use pointer here to avoid copy the value at benchmark registration time
    const LatencyMeasure *latency_measure);

}  // namespace benchmark
}  // namespace uvkc

#endif  // UVKC_BENCHMARK_MAIN_H_
