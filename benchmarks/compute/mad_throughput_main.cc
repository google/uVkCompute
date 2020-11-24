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
#include "uvkc/benchmark/fp16_util.h"
#include "uvkc/benchmark/main.h"
#include "uvkc/benchmark/status_util.h"
#include "uvkc/benchmark/vulkan_buffer_util.h"
#include "uvkc/benchmark/vulkan_context.h"
#include "uvkc/vulkan/device.h"
#include "uvkc/vulkan/pipeline.h"

using ::uvkc::benchmark::LatencyMeasureMode;
using namespace uvkc::benchmark;

static const char kBenchmarkName[] = "mad_throughput";

#include "mad_throughput_shader_spirv_permutation.inc"

struct ShaderCode {
  const char *name;       // Test case name
  const uint32_t *code;   // SPIR-V code
  size_t code_num_bytes;  // Number of bytes for SPIR-V code
  Precision precision;
};

ShaderCode kShaders[] = {
    {"mad_throughput_f32", TYPE_vec4, sizeof(TYPE_vec4), Precision::fp32},
    {"mad_throughput_f16", TYPE_f16vec4, sizeof(TYPE_f16vec4), Precision::fp16},
};

static void Throughput(::benchmark::State &state,
                       ::uvkc::vulkan::Device *device,
                       const ::uvkc::benchmark::LatencyMeasure *latency_measure,
                       const uint32_t *code, size_t code_num_words,
                       size_t num_element, int loop_count,
                       Precision precision) {
  //===-------------------------------------------------------------------===/
  // Create shader module, pipeline, and descriptor sets
  //===-------------------------------------------------------------------===/

  BM_CHECK_OK_AND_ASSIGN(auto shader_module,
                         device->CreateShaderModule(code, code_num_words));
  ::uvkc::vulkan::Pipeline::SpecConstant spec_constant = {};
  spec_constant.id = 0;
  spec_constant.type = ::uvkc::vulkan::Pipeline::SpecConstant::Type::s32;
  spec_constant.value.s32 = loop_count;
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
  const size_t src0_size = num_element * GetSize(precision);
  const size_t src1_size = num_element * GetSize(precision);
  const size_t dst_size = num_element * GetSize(precision);

  BM_CHECK_OK_AND_ASSIGN(
      auto src0_buffer,
      device->CreateBuffer(
          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, src0_size));
  BM_CHECK_OK_AND_ASSIGN(
      auto src1_buffer,
      device->CreateBuffer(
          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, src1_size));
  BM_CHECK_OK_AND_ASSIGN(
      auto dst_buffer,
      device->CreateBuffer(
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, dst_size));

  //===-------------------------------------------------------------------===/
  // Set source buffer data
  //===-------------------------------------------------------------------===/
  auto getSrc0 = [](size_t i) {
    float v = float((i % 9) + 1) * 0.1f;
    return v;
  };
  auto getSrc1 = [](size_t i) {
    float v = float((i % 5) + 1) * 1.f;
    return v;
  };
  if (precision == Precision::fp16) {
    BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
        device, src0_buffer.get(), src0_size, [&](void *ptr, size_t num_bytes) {
          uint16_t *src_float_buffer = reinterpret_cast<uint16_t *>(ptr);
          for (size_t i = 0; i < num_element; i++) {
            src_float_buffer[i] = fp16(getSrc0(i)).getValue();
          }
        }));

    BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
        device, src1_buffer.get(), src1_size, [&](void *ptr, size_t num_bytes) {
          uint16_t *src_float_buffer = reinterpret_cast<uint16_t *>(ptr);
          for (size_t i = 0; i < num_element; i++) {
            src_float_buffer[i] = fp16(getSrc1(i)).getValue();
          }
        }));
  } else if (precision == Precision::fp32) {
    BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
        device, src0_buffer.get(), src0_size, [&](void *ptr, size_t num_bytes) {
          float *src_float_buffer = reinterpret_cast<float *>(ptr);
          for (size_t i = 0; i < num_element; i++) {
            src_float_buffer[i] = getSrc0(i);
          }
        }));

    BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
        device, src1_buffer.get(), src1_size, [&](void *ptr, size_t num_bytes) {
          float *src_float_buffer = reinterpret_cast<float *>(ptr);
          for (size_t i = 0; i < num_element; i++) {
            src_float_buffer[i] = getSrc1(i);
          }
        }));
  }
  //===-------------------------------------------------------------------===/
  // Dispatch
  //===-------------------------------------------------------------------===/

  std::vector<::uvkc::vulkan::Device::BoundBuffer> bound_buffers = {
      {src0_buffer.get(), /*set=*/0, /*binding=*/0},
      {src1_buffer.get(), /*set=*/0, /*binding=*/1},
      {dst_buffer.get(), /*set=*/0, /*binding=*/2},
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
  dispatch_cmdbuf->Dispatch(num_element / (4 * 16), 1, 1);
  BM_CHECK_OK(dispatch_cmdbuf->End());
  BM_CHECK_OK(device->QueueSubmitAndWait(*dispatch_cmdbuf));

  //===-------------------------------------------------------------------===/
  // Verify destination buffer data
  //===-------------------------------------------------------------------===/

  if (precision == Precision::fp16) {
    BM_CHECK_OK(::uvkc::benchmark::GetDeviceBufferViaStagingBuffer(
        device, dst_buffer.get(), dst_size, [&](void *ptr, size_t num_bytes) {
          uint16_t *dst_float_buffer = reinterpret_cast<uint16_t *>(ptr);
          for (size_t i = 0; i < num_element; i++) {
            float limit = getSrc1(i) * (1.f / (1.f - getSrc0(i)));
            BM_CHECK_FLOAT_EQ(fp16(dst_float_buffer[i]).toFloat(), limit, 0.5f)
                << "destination buffer element #" << i
                << " has incorrect value: expected to be " << limit
                << " but found " << fp16(dst_float_buffer[i]).toFloat();
          }
        }));
  } else if (precision == Precision::fp32) {
    BM_CHECK_OK(::uvkc::benchmark::GetDeviceBufferViaStagingBuffer(
        device, dst_buffer.get(), dst_size, [&](void *ptr, size_t num_bytes) {
          float *dst_float_buffer = reinterpret_cast<float *>(ptr);
          for (size_t i = 0; i < num_element; i++) {
            float limit = getSrc1(i) * (1.f / (1.f - getSrc0(i)));
            BM_CHECK_FLOAT_EQ(dst_float_buffer[i], limit, 0.01f)
                << "destination buffer element #" << i
                << " has incorrect value: expected to be " << limit
                << " but found " << dst_float_buffer[i];
          }
        }));
  }

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

    cmdbuf->Dispatch(num_element / (4 * 16), 1, 1);

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

  double numOperation = double(num_element) * 2. /*fma*/ *
                        10. /*10 elements per loop iteration*/ *
                        double(loop_count);
  state.counters["FLOps"] =
      ::benchmark::Counter(numOperation,
                           ::benchmark::Counter::kIsIterationInvariant |
                               ::benchmark::Counter::kIsRate,
                           ::benchmark::Counter::kIs1000);

  // Reset the command pool to release all command buffers in the benchmarking
  // loop to avoid draining GPU resources.
  BM_CHECK_OK(device->ResetCommandPool());
}

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

  const size_t num_element = 1024 * 1024;
  const int min_loop_count = 100000;
  const int max_loop_count = min_loop_count * 2;
  for (const auto &shader : kShaders) {
    for (int loop_count = min_loop_count; loop_count <= max_loop_count;
         loop_count += min_loop_count) {
      std::string test_name = absl::StrCat(gpu_name, "/", shader.name, "/",
                                           num_element, "/", loop_count);
      ::benchmark::RegisterBenchmark(test_name.c_str(), Throughput, device,
                                     latency_measure, shader.code,
                                     shader.code_num_bytes / sizeof(uint32_t),
                                     num_element, loop_count, shader.precision)
          ->UseManualTime()
          ->Unit(::benchmark::kMicrosecond);
    }
  }
}

}  // namespace benchmark
}  // namespace uvkc
