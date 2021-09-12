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

using ::uvkc::benchmark::fp16;
using ::uvkc::benchmark::GetSize;
using ::uvkc::benchmark::LatencyMeasureMode;
using ::uvkc::benchmark::Precision;
using ::uvkc::vulkan::Pipeline;

static const char kBenchmarkName[] = "2d_convolution";

struct ShaderCode {
  const uint32_t *code;   // SPIR-V code
  size_t code_num_bytes;  // Number of bytes for SPIR-V code
  int invocation_oh;      // # indices for each invocation along output height
  int invocation_ow;      // # indices for each invocation along output width
  int invocation_oc;      // # vectors for each invocation along output channel
  int wg_size_x;          // gl_WorkGroupSize.x
  int wg_size_y;          // gl_WorkGroupSize.y
  int wg_size_z;          // gl_WorkGroupSize.z
  int scalar_per_thread;  // Number of scalar elements each thread process
  Precision precision;
};

#define SHADER_TILE(X, Y, Z, OH, OW, OC, T, Precision)                                                \
  {                                                                                                   \
    WG_X_##X##_WG_Y_##Y##_WG_Z_##Z##_IVC_OH_##OH##_IVC_OW_##OW##_IVC_OC_##OC##_VEC4TYPE_##T,          \
        sizeof(                                                                                       \
            WG_X_##X##_WG_Y_##Y##_WG_Z_##Z##_IVC_OH_##OH##_IVC_OW_##OW##_IVC_OC_##OC##_VEC4TYPE_##T), \
        OH, OW, OC, X, Y, Z, 4, Precision                                                             \
  }
#define F16_SHADER_TILE(X, Y, Z, OH, OW, OC) \
  SHADER_TILE(X, Y, Z, OH, OW, OC, f16vec4, Precision::fp16)
#define F32_SHADER_TILE(X, Y, Z, OH, OW, OC) \
  SHADER_TILE(X, Y, Z, OH, OW, OC, vec4, Precision::fp32)

#define F16_SHADER_PACK(X, Y, Z, OH, OW, OC)                                           \
  {                                                                                    \
    WG_X_##X##_WG_Y_##Y##_WG_Z_##Z##_IVC_OH_##OH##_IVC_OW_##OW##_IVC_OC_##OC,          \
        sizeof(                                                                        \
            WG_X_##X##_WG_Y_##Y##_WG_Z_##Z##_IVC_OH_##OH##_IVC_OW_##OW##_IVC_OC_##OC), \
        OH, OW, OC, X, Y, Z, 8, Precision::fp16                                        \
  }

// clang-format off
#define F32_WORKGROUP_TILE(X, Y, Z) \
  F32_SHADER_TILE(X, Y, Z, 1, 1, 1), F32_SHADER_TILE(X, Y, Z, 1, 2, 1), F32_SHADER_TILE(X, Y, Z, 1, 4, 1), \
  F32_SHADER_TILE(X, Y, Z, 2, 1, 1), F32_SHADER_TILE(X, Y, Z, 2, 2, 1), F32_SHADER_TILE(X, Y, Z, 2, 4, 1), \
  F32_SHADER_TILE(X, Y, Z, 4, 1, 1), F32_SHADER_TILE(X, Y, Z, 4, 2, 1), F32_SHADER_TILE(X, Y, Z, 4, 4, 1)

#define F16_WORKGROUP_PACK(X, Y, Z) \
  F16_SHADER_PACK(X, Y, Z, 1, 1, 1), F16_SHADER_PACK(X, Y, Z, 1, 2, 1), F16_SHADER_PACK(X, Y, Z, 1, 4, 1), \
  F16_SHADER_PACK(X, Y, Z, 2, 1, 1), F16_SHADER_PACK(X, Y, Z, 2, 2, 1), F16_SHADER_PACK(X, Y, Z, 2, 4, 1), \
  F16_SHADER_PACK(X, Y, Z, 4, 1, 1), F16_SHADER_PACK(X, Y, Z, 4, 2, 1), F16_SHADER_PACK(X, Y, Z, 4, 4, 1)
// clang-format on

#if defined(UVKC_ADRENO)

#include "conv2d_f16_packed_shader_adreno_spirv_permutation.inc"
#include "conv2d_f32_tiled_shader_adreno_spirv_permutation.inc"

static ShaderCode kShaderCodeCases[] = {
    F32_WORKGROUP_TILE(64, 1, 1), F32_WORKGROUP_TILE(32, 2, 1),
    F32_WORKGROUP_TILE(16, 4, 1), F32_WORKGROUP_TILE(16, 2, 2),
    F32_WORKGROUP_TILE(8, 4, 2),  F32_WORKGROUP_TILE(4, 4, 4),

    F16_WORKGROUP_PACK(64, 1, 1), F16_WORKGROUP_PACK(32, 2, 1),
    F16_WORKGROUP_PACK(16, 4, 1), F16_WORKGROUP_PACK(16, 2, 2),
    F16_WORKGROUP_PACK(8, 4, 2),  F16_WORKGROUP_PACK(4, 4, 4),
};

#elif defined(UVKC_MALI_VALHALL)

#include "conv2d_f16_packed_shader_valhall_spirv_permutation.inc"
#include "conv2d_f32_tiled_shader_valhall_spirv_permutation.inc"

static ShaderCode kShaderCodeCases[] = {
    F32_WORKGROUP_TILE(16, 1, 1), F32_WORKGROUP_TILE(8, 2, 1),
    F32_WORKGROUP_TILE(4, 4, 1),  F32_WORKGROUP_TILE(4, 2, 2),

    F16_WORKGROUP_PACK(16, 1, 1), F16_WORKGROUP_PACK(8, 2, 1),
    F16_WORKGROUP_PACK(4, 4, 1),  F16_WORKGROUP_PACK(4, 2, 2),
};

#else
#error "unsupported GPU architecture"
#endif

#undef F16_WORKGROUP_PACK
#undef F32_WORKGROUP_TILE
#undef F16_SHADER_TILE
#undef F32_SHADER_TILE
#undef F16_SHADER_PACK
#undef SHADER_TILE

struct DataScaleCase {
  int input_h;
  int input_w;
  int input_c;
  int filter_h;
  int filter_w;
  int output_c;
  int stride_h;
  int stride_w;
};

static DataScaleCase kDataCases[] = {
    {258, 258, 16, 3, 3, 256, 1, 1},
    //{513, 513, 16, 3, 3, 64, 2, 2},
};

static void Conv2D(::benchmark::State &state, ::uvkc::vulkan::Device *device,
                   const ::uvkc::benchmark::LatencyMeasure *latency_measure,
                   const uint32_t *code, size_t code_num_words, int input_h,
                   int input_w, int input_c, int filter_h, int filter_w,
                   int output_c, int stride_h, int stride_w, int wg_size_x,
                   int wg_size_y, int wg_size_z, int wg_tile_oh, int wg_tile_ow,
                   int wg_tile_oc, int scalar_per_thread, Precision precision) {
  int output_h = (input_h - filter_h) / stride_h + 1;
  int output_w = (input_w - filter_w) / stride_w + 1;

  BM_CHECK_EQ(output_h % wg_tile_oh, 0)
      << "expected output height to be a multiple of workgroup tile size";
  BM_CHECK_EQ(output_w % wg_tile_ow, 0)
      << "expected output width to be a multiple of workgroup tile size";
  BM_CHECK_EQ(output_c % wg_tile_oc, 0)
      << "expected output channel to be a multiple of workgroup tile size";
  BM_CHECK_EQ(wg_tile_oh % wg_size_z, 0)
      << "expected workgroup tile size to be a multiple of workgroup size";
  BM_CHECK_EQ(wg_tile_ow % wg_size_y, 0)
      << "expected workgroup tile size to be a multiple of workgroup size";
  BM_CHECK_EQ(wg_tile_oc % (wg_size_x * scalar_per_thread), 0)
      << "expected workgroup tile size to be a multiple of workgroup size";

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
      {8, Pipeline::SpecConstant::Type::u32, stride_h},
      {9, Pipeline::SpecConstant::Type::u32, stride_w},
  };
  BM_CHECK_OK_AND_ASSIGN(
      auto pipeline, device->CreatePipeline(*shader_module, "main",
                                            absl::MakeSpan(spec_constant, 10)));

  BM_CHECK_OK_AND_ASSIGN(auto descriptor_pool,
                         device->CreateDescriptorPool(*shader_module));
  BM_CHECK_OK_AND_ASSIGN(auto layout_set_map,
                         descriptor_pool->AllocateDescriptorSets(
                             shader_module->descriptor_set_layouts()));

  //===---------------------------------------------------------------------===/
  // Create buffers
  //===---------------------------------------------------------------------===/

  size_t input_size = input_h * input_w * input_c * GetSize(precision);
  size_t filter_size =
      filter_h * filter_w * input_c * output_c * GetSize(precision);
  size_t output_size = output_h * output_w * output_c * GetSize(precision);

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

  auto generateInputData = [](int h, int w, int c) -> float {
    return ((h + w * 2 + c * 3) % 3) * 0.5f;
  };
  auto generateFilterData = [](int h, int w, int ic, int oc) -> float {
    return ((h + w * 2 + ic * 3 + oc * 4) % 3) * 0.5f;
  };

  if (precision == Precision::fp16) {
    BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
        device, input_buffer.get(), input_size,
        [&](void *ptr, size_t num_bytes) {
          uint16_t *src_float_buffer = reinterpret_cast<uint16_t *>(ptr);
          for (int ih = 0; ih < input_h; ++ih) {
            for (int iw = 0; iw < input_w; ++iw) {
              for (int ic = 0; ic < input_c; ++ic) {
                int offset = ih * input_w * input_c + iw * input_c + ic;
                src_float_buffer[offset] =
                    fp16(generateInputData(ih, iw, ic)).getValue();
              }
            }
          }
        }));

    BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
        device, filter_buffer.get(), filter_size,
        [&](void *ptr, size_t num_bytes) {
          uint16_t *src_float_buffer = reinterpret_cast<uint16_t *>(ptr);
          for (int fh = 0; fh < filter_h; ++fh) {
            for (int fw = 0; fw < filter_w; ++fw) {
              for (int ic = 0; ic < input_c; ++ic) {
                for (int oc = 0; oc < output_c; ++oc) {
                  int offset = fh * filter_w * input_c * output_c +
                               fw * input_c * output_c + ic * output_c + oc;
                  src_float_buffer[offset] =
                      fp16(generateFilterData(fh, fw, ic, oc)).getValue();
                }
              }
            }
          }
        }));
  } else if (precision == Precision::fp32) {
    BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
        device, input_buffer.get(), input_size,
        [&](void *ptr, size_t num_bytes) {
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
  }

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
                            output_h / wg_tile_oh);
  BM_CHECK_OK(dispatch_cmdbuf->End());
  BM_CHECK_OK(device->QueueSubmitAndWait(*dispatch_cmdbuf));

  //===---------------------------------------------------------------------===/
  // Verify destination buffer data
  //===---------------------------------------------------------------------===/

  if (precision == Precision::fp16) {
    BM_CHECK_OK(::uvkc::benchmark::GetDeviceBufferViaStagingBuffer(
        device, output_buffer.get(), output_size,
        [&](void *ptr, size_t num_bytes) {
          uint16_t *dst_float_buffer = reinterpret_cast<uint16_t *>(ptr);
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
                float gpu_value = fp16(dst_float_buffer[offset]).toFloat();
                BM_CHECK_FLOAT_EQ(gpu_value, expected_value, 0.25f)
                    << "destination buffer element [" << oh << ", " << ow
                    << ", " << oc << "]"
                    << " has incorrect value: expected to be " << expected_value
                    << " but found " << gpu_value;
              }
            }
          }
        }));
  } else if (precision == Precision::fp32) {
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
                    << "destination buffer element [" << oh << ", " << ow
                    << ", " << oc << "]"
                    << " has incorrect value: expected to be " << expected_value
                    << " but found " << dst_float_buffer[offset];
              }
            }
          }
        }));
  }

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

    cmdbuf->Dispatch(output_c / wg_tile_oc, output_w / wg_tile_ow,
                     output_h / wg_tile_oh);

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

  for (const auto &data : kDataCases) {
    std::string workload_name = absl::StrCat(
        "Input[1x", data.input_h, "x", data.input_w, "x", data.input_c,
        "]xFilter[", data.filter_h, "x", data.filter_w, "x", data.input_c, "x",
        data.output_c, "]/Stride[", data.stride_h, "x", data.stride_w, "]");

    for (const auto &shader : kShaderCodeCases) {
      int wg_tile_oh = shader.invocation_oh * shader.wg_size_z;
      int wg_tile_ow = shader.invocation_ow * shader.wg_size_y;
      int wg_tile_oc =
          shader.invocation_oc * shader.wg_size_x * shader.scalar_per_thread;

      // Make sure the output image is tilable to integral number of workgroups.
      if (data.output_c % wg_tile_oc != 0) continue;
      int output_w = (data.input_w - data.filter_w) / data.stride_w + 1;
      if (output_w % wg_tile_ow != 0) continue;
      int output_h = (data.input_h - data.filter_h) / data.stride_h + 1;
      if (output_h % wg_tile_oh != 0) continue;

      std::string shader_name = absl::StrCat(
          "Tile[", wg_tile_oh, "x", wg_tile_ow, "x", wg_tile_oc, "]/WGSize[",
          shader.wg_size_x, "x", shader.wg_size_y, "x", shader.wg_size_z, "]/",
          (shader.precision == Precision::fp16 ? "f16" : "f32"));

      std::string test_name =
          absl::StrCat(gpu_name, "/", workload_name, "/", shader_name);

      ::benchmark::RegisterBenchmark(
          test_name.c_str(), Conv2D, device, latency_measure, shader.code,
          shader.code_num_bytes / sizeof(uint32_t), data.input_h, data.input_w,
          data.input_c, data.filter_h, data.filter_w, data.output_c,
          data.stride_h, data.stride_w, shader.wg_size_x, shader.wg_size_y,
          shader.wg_size_z, wg_tile_oh, wg_tile_ow, wg_tile_oc,
          shader.scalar_per_thread, shader.precision)
          ->UseManualTime()
          ->Unit(::benchmark::kMicrosecond);
    }
  }
}

}  // namespace benchmark
}  // namespace uvkc
