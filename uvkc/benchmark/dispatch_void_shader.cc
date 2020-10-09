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

#include <chrono>
#include <memory>
#include <numeric>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "benchmark/benchmark.h"
#include "uvkc/benchmark/status_util.h"
#include "uvkc/vulkan/device.h"

static uint32_t kShaderCode[] = {
#include "void_shader_spirv_instance.inc"
};

static void DispatchVoidShader(::benchmark::State &state,
                               ::uvkc::vulkan::Device *device,
                               double *avg_latency_seconds) {
  //===-------------------------------------------------------------------===/
  // Create shader module and pipeline
  //===-------------------------------------------------------------------===/

  BM_CHECK_OK_AND_ASSIGN(
      auto shader_module,
      device->CreateShaderModule(kShaderCode,
                                 sizeof(kShaderCode) / sizeof(uint32_t)));

  BM_CHECK_OK_AND_ASSIGN(auto pipeline,
                         device->CreatePipeline(*shader_module, "main",
                                                /*spec_constants=*/{}));

  //===-------------------------------------------------------------------===/
  // Benchmarking
  //===-------------------------------------------------------------------===/

  BM_CHECK_OK_AND_ASSIGN(auto cmdbuf, device->AllocateCommandBuffer());
  double total_seconds = 0;
  for (auto _ : state) {
    BM_CHECK_OK(cmdbuf->Begin());
    cmdbuf->BindPipelineAndDescriptorSets(*pipeline,
                                          /*bound_descriptor_sets=*/{});
    cmdbuf->Dispatch(1, 1, 1);
    BM_CHECK_OK(cmdbuf->End());
    auto start_time = std::chrono::high_resolution_clock::now();
    BM_CHECK_OK(device->QueueSubmitAndWait(*cmdbuf));
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);
    state.SetIterationTime(elapsed_seconds.count());
    total_seconds += elapsed_seconds.count();
    BM_CHECK_OK(cmdbuf->Reset());
  }
  *avg_latency_seconds = total_seconds / state.iterations();

  // Reset the command pool to release all command buffers in the benchmarking
  // loop to avoid draining GPU resources.
  BM_CHECK_OK(device->ResetCommandPool());
}

namespace uvkc {
namespace benchmark {

void RegisterDispatchVoidShaderBenchmark(const char *gpu_name,
                                         vulkan::Device *device,
                                         double *avg_latency_seconds) {
  std::string test_name = absl::StrCat(gpu_name, "/dispatch_void_shader");
  ::benchmark::RegisterBenchmark(test_name.c_str(), DispatchVoidShader, device,
                                 avg_latency_seconds)
      ->UseManualTime()
      ->Unit(::benchmark::kMicrosecond);
}

}  // namespace benchmark
}  // namespace uvkc
