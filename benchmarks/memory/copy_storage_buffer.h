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

#ifndef BENCHMARKS_MEMORY_COPY_STORAGE_BUFFER_H_
#define BENCHMARKS_MEMORY_COPY_STORAGE_BUFFER_H_

#include "uvkc/benchmark/vulkan_context.h"
#include "uvkc/vulkan/device.h"

namespace uvkc {
namespace benchmark {

enum class StorageBufferElementType {
  Float,
  Float4,
};

// Regisers a benchmark that measures the average latency of copying the data
// from a storage buffer at (set#0, binding#0) to another one at (set#0,
// binding#1) on |device| with the given |gpu_name|. Writes the average latency
// to |avg_latency_seconds| after benchmarking.
void RegisterCopyStorageBufferBenchmark(const char *gpu_name,
                                        vulkan::Device *device,
                                        size_t buffer_num_bytes,
                                        StorageBufferElementType element_type,
                                        LatencyMeasureMode latency_measure_mode,
                                        const double *overhead_latency_seconds,
                                        double *avg_latency_seconds);

}  // namespace benchmark
}  // namespace uvkc

#endif  // BENCHMARKS_MEMORY_COPY_STORAGE_BUFFER_H_
