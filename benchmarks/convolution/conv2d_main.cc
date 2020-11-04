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
#include "uvkc/benchmark/main.h"
#include "uvkc/benchmark/status_util.h"
#include "uvkc/benchmark/vulkan_buffer_util.h"
#include "uvkc/benchmark/vulkan_context.h"
#include "uvkc/vulkan/device.h"
#include "uvkc/vulkan/pipeline.h"

using ::uvkc::benchmark::LatencyMeasureMode;
using ::uvkc::vulkan::Pipeline;

static const char kBenchmarkName[] = "convolution";

#include "conv2d_tiled_shader_spirv_permutation.inc"

struct ShaderCode {
  const char *name;       // Test case name
  const uint32_t *code;   // SPIR-V code
  size_t code_num_bytes;  // Number of bytes for SPIR-V code
  int wg_tile_ow;         // Workgroup tile size along output width dimension
  int wg_tile_oc;         // Workgroup tile size along output channel dimension
};

#define SHADER_TILE(A, B)                             \
  {                                                   \
#A "x" #B, WG_TILE_OW_##A##_WG_TILE_OC_##B,       \
        sizeof(WG_TILE_OW_##A##_WG_TILE_OC_##B), A, B \
  }

static ShaderCode kShaderCodeCases[] = {
    // clang-format off
    SHADER_TILE(2, 32),  SHADER_TILE(2, 64),
    SHADER_TILE(4, 32),  SHADER_TILE(4, 64),
    SHADER_TILE(8, 32),  SHADER_TILE(8, 64),
    SHADER_TILE(16, 32), SHADER_TILE(16, 64),
    SHADER_TILE(32, 32), SHADER_TILE(32, 64),
    // clang-format on
};
#undef SHADER_TILE

static void Conv2D(::benchmark::State &state, ::uvkc::vulkan::Device *device,
                   const ::uvkc::benchmark::LatencyMeasure *latency_measure,
                   const uint32_t *code, size_t code_num_words, int input_h,
                   int input_w, int input_c, int filter_h, int filter_w,
                   int output_c, int stride_h, int stride_w, int wg_tile_ow,
                   int wg_tile_oc) {
  int output_h = (input_h - filter_h) / stride_h + 1;
  int output_w = (input_w - filter_w) / stride_w + 1;

  BM_CHECK_EQ(stride_h, 1) << "expected height stride to be 1";
  BM_CHECK_EQ(stride_w, 1) << "expected widht stride to be 1";
  BM_CHECK_EQ(input_c % 4, 0) << "expected input channel to be a multiple of 4";
  BM_CHECK_EQ(output_w % 2, 0) << "expected output width to be a multiple of 2";
  BM_CHECK_EQ(output_c % 32, 0)
      << "expected input channel to be a multiple of 32";

  //===---------------------------------------------------------------------===/
  // Create shader module, pipeline, and descriptor sets
  //===---------------------------------------------------------------------===/

  BM_CHECK_OK_AND_ASSIGN(auto shader_module,
                         device->CreateShaderModule(code, code_num_words));

  Pipeline::SpecConstant spec_constant[] = {
      {0, Pipeline::SpecConstant::Type::u32, output_h},
      {1, Pipeline::SpecConstant::Type::u32, output_w},
      {2, Pipeline::SpecConstant::Type::u32, output_c},
      {3, Pipeline::SpecConstant::Type::u32, input_h},
      {4, Pipeline::SpecConstant::Type::u32, input_w},
      {5, Pipeline::SpecConstant::Type::u32, input_c},
      {6, Pipeline::SpecConstant::Type::u32, filter_h},
      {7, Pipeline::SpecConstant::Type::u32, filter_w},
  };
  BM_CHECK_OK_AND_ASSIGN(
      auto pipeline, device->CreatePipeline(*shader_module, "main",
                                            absl::MakeSpan(spec_constant, 8)));

  BM_CHECK_OK_AND_ASSIGN(auto descriptor_pool,
                         device->CreateDescriptorPool(*shader_module));
  BM_CHECK_OK_AND_ASSIGN(auto layout_set_map,
                         descriptor_pool->AllocateDescriptorSets(
                             shader_module->descriptor_set_layouts()));

  //===---------------------------------------------------------------------===/
  // Create buffers
  //===---------------------------------------------------------------------===/

  size_t input_size = input_h * input_w * input_c * sizeof(float);
  size_t filter_size = filter_h * filter_w * input_c * output_c * sizeof(float);
  size_t output_size = output_h * output_w * output_c * sizeof(float);

  BM_CHECK_OK_AND_ASSIGN(
      auto input_buffer,
      device->CreateBuffer(
          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, input_size));
  BM_CHECK_OK_AND_ASSIGN(
      auto filter_buffer,
      device->CreateBuffer(
          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, filter_size));
  BM_CHECK_OK_AND_ASSIGN(
      auto output_buffer,
      device->CreateBuffer(
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, output_size));

  //===---------------------------------------------------------------------===/
  // Set source buffer data
  //===---------------------------------------------------------------------===/

  auto generateInputData = [](int h, int w, int c) {
    return float(h % 17) * 0.5f + float(w % 13) * 0.5f + float(c % 9) * 0.25f;
  };
  auto generateFilterData = [](int h, int w, int ic, int oc) {
    return float(h % 5) * 0.25f + float(w % 7) * 0.25f +
           float(ic % 21) * 0.25f + float(oc % 13) * 0.5f;
  };

  BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
      device, input_buffer.get(), input_size, [&](void *ptr, size_t num_bytes) {
        float *src_float_buffer = reinterpret_cast<float *>(ptr);
        for (int ih = 0; ih < input_h; ++ih) {
          for (int iw = 0; iw < input_w; ++iw) {
            for (int ic = 0; ic < input_c; ++ic) {
              int offset = ih * input_w * input_c + iw * input_c + ic;
              src_float_buffer[offset] = generateInputData(ih, iw, ic);
            }
          }
        }
      }));

  BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
      device, filter_buffer.get(), filter_size,
      [&](void *ptr, size_t num_bytes) {
        float *src_float_buffer = reinterpret_cast<float *>(ptr);
        for (int fh = 0; fh < filter_h; ++fh) {
          for (int fw = 0; fw < filter_w; ++fw) {
            for (int ic = 0; ic < input_c; ++ic) {
              for (int oc = 0; oc < output_c; ++oc) {
                int offset = fh * filter_w * input_c * output_c +
                             fw * input_c * output_c + ic * output_c + oc;
                src_float_buffer[offset] = generateFilterData(fh, fw, ic, oc);
              }
            }
          }
        }
      }));

  //===---------------------------------------------------------------------===/
  // Dispatch
  //===---------------------------------------------------------------------===/

  std::vector<::uvkc::vulkan::Device::BoundBuffer> bound_buffers{
      {input_buffer.get(), /*set=*/0, /*binding=*/0},
      {filter_buffer.get(), /*set=*/0, /*binding=*/1},
      {output_buffer.get(), /*set=*/0, /*binding=*/2},
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
  dispatch_cmdbuf->Dispatch(output_c / wg_tile_oc, output_w / wg_tile_ow,
                            output_h);
  BM_CHECK_OK(dispatch_cmdbuf->End());
  BM_CHECK_OK(device->QueueSubmitAndWait(*dispatch_cmdbuf));

  //===---------------------------------------------------------------------===/
  // Verify destination buffer data
  //===---------------------------------------------------------------------===/

  BM_CHECK_OK(::uvkc::benchmark::GetDeviceBufferViaStagingBuffer(
      device, output_buffer.get(), output_size,
      [&](void *ptr, size_t num_bytes) {
        float *dst_float_buffer = reinterpret_cast<float *>(ptr);
        for (int oh = 0; oh < output_h; ++oh) {
          for (int ow = 0; ow < output_w; ++ow) {
            for (int oc = 0; oc < output_c; ++oc) {
              float expected_value = 0.f;
              for (int fh = 0; fh < filter_h; ++fh) {
                for (int fw = 0; fw < filter_w; ++fw) {
                  for (int ic = 0; ic < input_c; ++ic) {
                    int ih = oh * stride_h + fh;
                    int iw = ow * stride_w + fw;
                    float input = generateInputData(ih, iw, ic);
                    float filter = generateFilterData(fh, fw, ic, oc);
                    expected_value += input * filter;
                  }
                }
              }

              int offset = oh * output_w * output_c + ow * output_c + oc;
              BM_CHECK_EQ(dst_float_buffer[offset], expected_value)
                  << "destination buffer element [" << oh << ", " << ow << ", "
                  << oc << "]"
                  << " has incorrect value: expected to be " << expected_value
                  << " but found " << dst_float_buffer[offset];
            }
          }
        }
      }));

  //===---------------------------------------------------------------------===/
  // Benchmarking
  //===---------------------------------------------------------------------===/

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

    cmdbuf->Dispatch(output_c / wg_tile_oc, output_w / wg_tile_ow, output_h);

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

  double num_operations =
      // For each output element:
      double(output_h) * double(output_w) * double(output_c) *
      // Convolution performs dot product of the filter's size.
      double(filter_h) * double(filter_w) * double(input_c) * 2;
  state.counters["FLOps"] =
      ::benchmark::Counter(num_operations,
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

  const int input_h = 256;
  const int input_w = 256;
  const int input_c = 16;
  const int filter_h = 3;
  const int filter_w = 3;
  const int output_c = 64;
  const int stride_h = 1;
  const int stride_w = 1;

  std::string workload_name =
      absl::StrCat("Input[1x", input_h, "x", input_w, "x", input_c, "]xFilter[",
                   filter_h, "x", filter_w, "x", input_c, "x", output_c, "]");

  for (const auto &shader : kShaderCodeCases) {
    std::string test_name =
        absl::StrCat(gpu_name, "/", workload_name, "/Tile[", shader.name, "]");
    ::benchmark::RegisterBenchmark(
        test_name.c_str(), Conv2D, device, latency_measure, shader.code,
        shader.code_num_bytes / sizeof(uint32_t), input_h, input_w, input_c,
        filter_h, filter_w, output_c, stride_h, stride_w, shader.wg_tile_ow,
        shader.wg_tile_oc)
        ->UseManualTime()
        ->Unit(::benchmark::kMicrosecond);
  }
}

}  // namespace benchmark
}  // namespace uvkc
