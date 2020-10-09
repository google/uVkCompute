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

#ifndef UVKC_BENCHMARK_DISPATCH_VOID_SHADER_H_
#define UVKC_BENCHMARK_DISPATCH_VOID_SHADER_H_

#include "uvkc/vulkan/device.h"

namespace uvkc {
namespace benchmark {

// Regisers a benchmark that measures the average latency of dispatching a void
// shader to the given |device| with the given |gpu_name|. Writes the average
// latency to |avg_latency_seconds| after benchmarking.
void RegisterDispatchVoidShaderBenchmark(const char *gpu_name,
                                         vulkan::Device *device,
                                         double *avg_latency_seconds);

}  // namespace benchmark
}  // namespace uvkc

#endif  // UVKC_BENCHMARK_DISPATCH_VOID_SHADER_H_
