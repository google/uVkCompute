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
#include <functional>
#include <memory>
#include <numeric>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "benchmark/benchmark.h"
#include "benchmarks/memory/copy_storage_buffer.h"
#include "uvkc/benchmark/main.h"
#include "uvkc/benchmark/status_util.h"
#include "uvkc/benchmark/vulkan_buffer_util.h"
#include "uvkc/benchmark/vulkan_context.h"
#include "uvkc/vulkan/device.h"
#include "uvkc/vulkan/pipeline.h"
#include "uvkc/vulkan/timestamp_query_pool.h"

using ::uvkc::benchmark::LatencyMeasureMode;
using ::uvkc::vulkan::Pipeline;

static const char kBenchmarkName[] = "subgroup_arthmetic";

static uint32_t kAddLoopCode[] = {
#include "subgroup_add_loop_spirv_instance.inc"
};

static uint32_t kAddIntrinsicCode[] = {
#include "subgroup_add_intrinsic_spirv_instance.inc"
};

static uint32_t kMulLoopCode[] = {
#include "subgroup_mul_loop_spirv_instance.inc"
};

static uint32_t kMulIntrinsicCode[] = {
#include "subgroup_mul_intrinsic_spirv_instance.inc"
};

enum class Arithmetic { Add, Mul };

struct ShaderCode {
  const char *name;       // Test case name
  const uint32_t *code;   // SPIR-V code
  size_t code_num_bytes;  // Number of bytes for SPIR-V code
  Arithmetic op;
};

static ShaderCode kShaderCodeCases[] = {
    // clang-format off
    {"add/loop", kAddLoopCode, sizeof(kAddLoopCode), Arithmetic::Add},
    {"add/intrinsic", kAddIntrinsicCode, sizeof(kAddIntrinsicCode), Arithmetic::Add},
    {"mul/loop", kMulLoopCode, sizeof(kMulLoopCode), Arithmetic::Mul},
    {"mul/intrinsic", kMulIntrinsicCode, sizeof(kMulIntrinsicCode), Arithmetic::Mul},
    // clang-format on
};

static uint32_t kWorkgroupSize = 64;

static void CalculateSubgroupArithmetic(
    ::benchmark::State &state, ::uvkc::vulkan::Device *device,
    const ::uvkc::benchmark::LatencyMeasure *latency_measure,
    const uint32_t *code, size_t code_num_words, int num_elements,
    uint32_t subgroup_size, Arithmetic arith_op) {
  size_t buffer_num_bytes = num_elements * sizeof(float);

  //===-------------------------------------------------------------------===/
  // Create shader module, pipeline, and descriptor sets
  //===-------------------------------------------------------------------===/

  BM_CHECK_OK_AND_ASSIGN(auto shader_module,
                         device->CreateShaderModule(code, code_num_words));

  ::uvkc::vulkan::Pipeline::SpecConstant spec_constant = {
      /*id=*/0, Pipeline::SpecConstant::Type::s32, num_elements};
  BM_CHECK_OK_AND_ASSIGN(
      auto pipeline, device->CreatePipeline(*shader_module, "main",
                                            absl::MakeSpan(&spec_constant, 1)));

  BM_CHECK_OK_AND_ASSIGN(auto descriptor_pool,
                         device->CreateDescriptorPool(*shader_module));
  BM_CHECK_OK_AND_ASSIGN(auto layout_set_map,
                         descriptor_pool->AllocateDescriptorSets(
                             shader_module->descriptor_set_layouts()));

  //===-------------------------------------------------------------------===/
  // Create buffers
  //===-------------------------------------------------------------------===/

  BM_CHECK_OK_AND_ASSIGN(
      auto src_buffer,
      device->CreateBuffer(
          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer_num_bytes));
  BM_CHECK_OK_AND_ASSIGN(
      auto dst_buffer,
      device->CreateBuffer(
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer_num_bytes));

  //===-------------------------------------------------------------------===/
  // Set source buffer data
  //===-------------------------------------------------------------------===/

  // +: fill the whole buffer as 1.0f.
  // *: fill with alternating subgroup_size and (1 / subgroup_size).
  BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
      device, src_buffer.get(), buffer_num_bytes,
      [arith_op, subgroup_size](void *ptr, size_t num_bytes) {
        float *src_float_buffer = reinterpret_cast<float *>(ptr);
        switch (arith_op) {
          case Arithmetic::Add: {
            for (int i = 0; i < num_bytes / sizeof(float); ++i) {
              src_float_buffer[i] = 1.0f;
            }
          } break;
          case Arithmetic::Mul: {
            for (int i = 0; i < num_bytes / sizeof(float); i += 2) {
              src_float_buffer[i] = subgroup_size;
              src_float_buffer[i + 1] = 1.0f / subgroup_size;
            }
          } break;
        }
      }));

  //===-------------------------------------------------------------------===/
  // Dispatch
  //===-------------------------------------------------------------------===/

  std::vector<::uvkc::vulkan::Device::BoundBuffer> bound_buffers = {
      {src_buffer.get(), /*set=*/0, /*binding=*/0},
      {dst_buffer.get(), /*set=*/0, /*binding=*/1},
  };
  BM_CHECK_OK(device->AttachBufferToDescriptor(
      *shader_module, layout_set_map,
      {bound_buffers.data(), bound_buffers.size()}));

  BM_CHECK_EQ(shader_module->descriptor_set_layouts().size(), 1)
      << "unexpected number of descriptor sets";
  auto descriptor_set_layout = shader_module->descriptor_set_layouts().front();

  std::vector<::uvkc::vulkan::CommandBuffer::BoundDescriptorSet>
      bound_descriptor_sets(1);
  bound_descriptor_sets[0].index = 0;
  bound_descriptor_sets[0].set = layout_set_map.at(descriptor_set_layout);
  BM_CHECK_OK_AND_ASSIGN(auto dispatch_cmdbuf, device->AllocateCommandBuffer());

  BM_CHECK_OK(dispatch_cmdbuf->Begin());
  dispatch_cmdbuf->BindPipelineAndDescriptorSets(
      *pipeline, {bound_descriptor_sets.data(), bound_descriptor_sets.size()});
  dispatch_cmdbuf->Dispatch(num_elements / kWorkgroupSize, 1, 1);
  BM_CHECK_OK(dispatch_cmdbuf->End());
  BM_CHECK_OK(device->QueueSubmitAndWait(*dispatch_cmdbuf));

  //===-------------------------------------------------------------------===/
  // Verify destination buffer data
  //===-------------------------------------------------------------------===/

  BM_CHECK_OK(::uvkc::benchmark::GetDeviceBufferViaStagingBuffer(
      device, dst_buffer.get(), buffer_num_bytes,
      [arith_op, subgroup_size](void *ptr, size_t num_bytes) {
        float *dst_float_buffer = reinterpret_cast<float *>(ptr);
        switch (arith_op) {
          case Arithmetic::Add: {
            for (int i = 0; i < num_bytes / sizeof(float); ++i) {
              float expected_value = 1.0f;
              if (i % subgroup_size == 0) {
                expected_value = subgroup_size;
              }

              BM_CHECK_EQ(dst_float_buffer[i], expected_value)
                  << "destination buffer element #" << i
                  << " has incorrect value: expected to be " << expected_value
                  << " but found " << dst_float_buffer[i];
            }
          } break;
          case Arithmetic::Mul: {
            for (int i = 0; i < num_bytes / sizeof(float); ++i) {
              float expected_value = 0.0f;
              if (i % subgroup_size == 0) {
                expected_value = 1.0f;
              } else if (i % 2 == 0) {
                expected_value = subgroup_size;
              } else {
                expected_value = 1.0f / subgroup_size;
              }

              BM_CHECK_EQ(dst_float_buffer[i], expected_value)
                  << "destination buffer element #" << i
                  << " has incorrect value: expected to be " << expected_value
                  << " but found " << dst_float_buffer[i];
            }
          } break;
        }
      }));

  //===-------------------------------------------------------------------===/
  // Benchmarking
  //===-------------------------------------------------------------------===/

  std::unique_ptr<::uvkc::vulkan::TimestampQueryPool> query_pool;
  bool use_timestamp =
      latency_measure->mode == LatencyMeasureMode::kGpuTimestamp;
  if (use_timestamp) {
    BM_CHECK_OK_AND_ASSIGN(query_pool, device->CreateTimestampQueryPool(2));
  }

  BM_CHECK_OK_AND_ASSIGN(auto cmdbuf, device->AllocateCommandBuffer());
  for (auto _ : state) {
    BM_CHECK_OK(cmdbuf->Begin());
    if (use_timestamp) cmdbuf->ResetQueryPool(*query_pool);

    cmdbuf->BindPipelineAndDescriptorSets(
        *pipeline,
        {bound_descriptor_sets.data(), bound_descriptor_sets.size()});

    if (use_timestamp) {
      cmdbuf->WriteTimestamp(*query_pool, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0);
    }

    cmdbuf->Dispatch(num_elements / kWorkgroupSize, 1, 1);

    if (use_timestamp) {
      cmdbuf->WriteTimestamp(*query_pool, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             1);
    }

    BM_CHECK_OK(cmdbuf->End());

    auto start_time = std::chrono::high_resolution_clock::now();
    BM_CHECK_OK(device->QueueSubmitAndWait(*cmdbuf));
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);

    switch (latency_measure->mode) {
      case LatencyMeasureMode::kSystemDispatch: {
        state.SetIterationTime(elapsed_seconds.count() -
                               latency_measure->overhead_seconds);
      } break;
      case LatencyMeasureMode::kSystemSubmit: {
        state.SetIterationTime(elapsed_seconds.count());
      } break;
      case LatencyMeasureMode::kGpuTimestamp: {
        BM_CHECK_OK_AND_ASSIGN(
            double timestamp_seconds,
            query_pool->CalculateElapsedSecondsBetween(0, 1));
        state.SetIterationTime(timestamp_seconds);
      } break;
    }

    BM_CHECK_OK(cmdbuf->Reset());
  }
  state.counters["FLOps"] =
      ::benchmark::Counter(num_elements,
                           ::benchmark::Counter::kIsIterationInvariant |
                               ::benchmark::Counter::kIsRate,
                           ::benchmark::Counter::kIs1000);

  // Reset the command pool to release all command buffers in the benchmarking
  // loop to avoid draining GPU resources.
  BM_CHECK_OK(device->ResetCommandPool());
}

static int kBufferNumElements = 1 << 20;  // 1M

namespace uvkc {
namespace benchmark {

absl::StatusOr<std::unique_ptr<VulkanContext>> CreateVulkanContext() {
  return CreateDefaultVulkanContext(kBenchmarkName);
}

bool RegisterVulkanOverheadBenchmark(
    const vulkan::Driver::PhysicalDeviceInfo &physical_device,
    vulkan::Device *device, double *overhead_seconds) {
  RegisterCopyStorageBufferBenchmark(physical_device.v10_properties.deviceName,
                                     device, kBufferNumElements * sizeof(float),
                                     StorageBufferElementType::Float,
                                     LatencyMeasureMode::kSystemSubmit,
                                     /*overhead_seconds=*/0, overhead_seconds);
  return true;
}

void RegisterVulkanBenchmarks(
    const vulkan::Driver::PhysicalDeviceInfo &physical_device,
    vulkan::Device *device, const LatencyMeasure *latency_measure) {
  const char *gpu_name = physical_device.v10_properties.deviceName;

  for (const auto &shader : kShaderCodeCases) {  // Loop/intrinsic shader
    std::string test_name =
        absl::StrCat(gpu_name, "/", shader.name, "/", kBufferNumElements);
    ::benchmark::RegisterBenchmark(
        test_name.c_str(), CalculateSubgroupArithmetic, device, latency_measure,
        shader.code, shader.code_num_bytes / sizeof(uint32_t),
        kBufferNumElements, physical_device.subgroup_properties.subgroupSize,
        shader.op)
        ->UseManualTime()
        ->Unit(::benchmark::kMicrosecond);
  }
}

}  // namespace benchmark
}  // namespace uvkc
