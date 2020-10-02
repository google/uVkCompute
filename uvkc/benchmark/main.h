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

namespace uvkc {
namespace benchmark {

// Creates a Vulkan application context for the current benchmark binary.
//
// The context is expected to hold Vulkan objects that can be shared among
// multiple benchmarks, e.g., the Vulkan driver and device.
//
// The context will be created before running all benchmarks and it will persist
// during the lifetime of all benchmarks.
absl::StatusOr<std::unique_ptr<VulkanContext>> CreateVulkanContext();

// Registers all Vulkan benchmarks for the current benchmark binary.
void RegisterVulkanBenchmarks(VulkanContext *vulkan_context);

}  // namespace benchmark
}  // namespace uvkc

#endif  // UVKC_BENCHMARK_MAIN_H_
