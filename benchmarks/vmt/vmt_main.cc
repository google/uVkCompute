// Copyright 2020-2023 Google LLC
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
#include <cstdint>
#include <memory>
#include <numeric>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "benchmark/benchmark.h"
#include "uvkc/base/log.h"
#include "uvkc/benchmark/data_type_util.h"
#include "uvkc/benchmark/main.h"
#include "uvkc/benchmark/status_util.h"
#include "uvkc/benchmark/vulkan_buffer_util.h"
#include "uvkc/benchmark/vulkan_context.h"
#include "uvkc/vulkan/device.h"
#include "uvkc/vulkan/pipeline.h"

using namespace uvkc::benchmark;
using ::uvkc::benchmark::LatencyMeasureMode;
using ::uvkc::vulkan::Pipeline;

static const char kBenchmarkName[] = "vmt";

struct ShaderCode {
  const char *name;                 // Shader case name
  absl::Span<const uint32_t> code;  // SPIR-V code
  int N0;
  int K0;
  int wg_size_x;
  int wg_size_y;
  DataType input_type;   // LHS & RHS element type
  DataType output_type;  // Output/Result matrix element type
};

#define SHADER_I8(N0, K0, X, Y)                                               \
  ShaderCode {                                                                \
    "Tile[" #N0 "x" #K0 "]", vmt_i8::N0_##N0##_K0_##K0##_WG_X_##X##_WG_Y_##Y, \
        N0, K0, X, Y, DataType::i8, DataType::i32                             \
  }

#define WORKGROUP_TILE_N_I8(X, Y, N0) \
  SHADER_I8(N0, 8, X, Y), SHADER_I8(N0, 8, X, Y)

#if defined(UVKC_RDNA3)

namespace vmt_i8 {
#include "vmt_i8_shader_rdna3_spirv_permutation.inc"
}

static ShaderCode kShaderCodeCases[] = {
    WORKGROUP_TILE_N_I8(64, 1, 1), WORKGROUP_TILE_N_I8(64, 1, 2),
    WORKGROUP_TILE_N_I8(64, 1, 4), WORKGROUP_TILE_N_I8(64, 2, 2),
    WORKGROUP_TILE_N_I8(64, 2, 4), WORKGROUP_TILE_N_I8(64, 4, 4),
};

#elif defined(UVKC_PROMOTE_RDNA3)

namespace vmt_i8 {
#include "vmt_promote_lhs_i8_shader_rdna3_spirv_permutation.inc"
}

static ShaderCode kShaderCodeCases[] = {
    WORKGROUP_TILE_N_I8(64, 1, 1), WORKGROUP_TILE_N_I8(64, 1, 2),
    WORKGROUP_TILE_N_I8(64, 1, 4), WORKGROUP_TILE_N_I8(64, 2, 2),
    WORKGROUP_TILE_N_I8(64, 2, 4), WORKGROUP_TILE_N_I8(64, 4, 4),
};

#else
#error "unsupported GPU architecture/strategy"
#endif

/// Fills the 2D matrix with values produced by the |generator| function.
template <typename GeneratorFn>
static void FillBuffer(DataType data_type, void *raw_buffer, size_t num_bytes,
                       unsigned dim_1, unsigned dim_2, GeneratorFn generator) {
  auto fill = [&](auto traits) {
    using Traits = decltype(traits);
    using StorageType = typename Traits::storage_type;
    using RuntimeType = typename Traits::storage_type;
    auto buffer = absl::MakeSpan(static_cast<StorageType *>(raw_buffer),
                                 num_bytes / GetSize(data_type));

    for (int i = 0; i < dim_1; ++i) {
      for (int j = 0; j < dim_2; ++j) {
        buffer[j + i * dim_1] =
            static_cast<StorageType>(RuntimeType(generator(i, j)));
      }
    }
  };

  InvokeWithTraits(data_type, fill);
}

/// Checks that the output vector calculated by the shader is contains the
/// same values as runtime vecmat with values defined by |lhs| and |rhs|.
template <DataType OutputType, DataType InputType, typename Generator1Fn,
          typename Generator2Fn>
static void CheckOutput(const ShaderCode &shader, void *raw_buffer,
                        size_t num_bytes, unsigned N, unsigned K,
                        Generator1Fn lhs, Generator2Fn rhs) {
  using OutputTraits = DataTypeTraits<OutputType>;
  using OutputStorageType = typename OutputTraits::storage_type;
  using OutputRuntimeType = typename OutputTraits::runtime_type;
  using InputTraits = DataTypeTraits<InputType>;
  using InputRuntimeType = typename InputTraits::runtime_type;

  auto output =
      absl::MakeConstSpan(static_cast<OutputStorageType *>(raw_buffer),
                          num_bytes / GetSize(OutputType));
  for (int j = 0; j < N; ++j) {
    OutputRuntimeType acc(0.0f);
    for (int k = 0; k < K; ++k) {
      acc += OutputRuntimeType(InputRuntimeType(lhs(0, k))) *
             OutputRuntimeType(InputRuntimeType(rhs(j, k)));
    }

    OutputRuntimeType gpuValue(output[j]);
    BM_CHECK_EQ(gpuValue, acc)
        << "destination buffer element (" << j << ")"
        << " has incorrect value: expected to be " << acc << " but found "
        << gpuValue << "\n\t^ In shader: " << shader.name << ", "
        << GetName(shader.input_type) << "->" << GetName(shader.output_type);
  }
}

static void Vmt(::benchmark::State &state, ::uvkc::vulkan::Device *device,
                const ::uvkc::benchmark::LatencyMeasure *latency_measure,
                const ShaderCode &shader, int N, int K) {
  //===-------------------------------------------------------------------===/
  // Create shader module, pipeline, and descriptor sets
  //===-------------------------------------------------------------------===/

  BM_CHECK_OK_AND_ASSIGN(
      auto shader_module,
      device->CreateShaderModule(shader.code.data(), shader.code.size()));

  ::uvkc::vulkan::Pipeline::SpecConstant spec_constant[2] = {
      {/*id=*/0, Pipeline::SpecConstant::Type::s32, N},
      {/*id=*/1, Pipeline::SpecConstant::Type::s32, K},
  };
  BM_CHECK_OK_AND_ASSIGN(
      auto pipeline,
      device->CreatePipeline(*shader_module, "main", spec_constant));

  BM_CHECK_OK_AND_ASSIGN(auto descriptor_pool,
                         device->CreateDescriptorPool(*shader_module));
  BM_CHECK_OK_AND_ASSIGN(auto layout_set_map,
                         descriptor_pool->AllocateDescriptorSets(
                             shader_module->descriptor_set_layouts()));

  //===-------------------------------------------------------------------===/
  // Create buffers
  //===-------------------------------------------------------------------===/
  DataType input_type = shader.input_type;
  DataType output_type = shader.output_type;
  const size_t src0_size = K * GetSize(input_type);
  const size_t src1_size = K * N * GetSize(input_type);
  const size_t dst_size = N * GetSize(output_type);

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

  VkExtent3D dimensions1 = {uint32_t(N / 8), uint32_t(K), 1};
  BM_CHECK_OK_AND_ASSIGN(
      auto src_image1,
      device->CreateImage(
          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_TYPE_2D,
          VK_FORMAT_R32G32B32A32_SFLOAT, dimensions1, VK_IMAGE_TILING_OPTIMAL,
          VK_IMAGE_VIEW_TYPE_2D));
  BM_CHECK_OK_AND_ASSIGN(auto src_sampler1, device->CreateSampler());

  //===-------------------------------------------------------------------===/
  // Set source buffer data
  //===-------------------------------------------------------------------===/
  auto getLhs = [K](int i, int j) {
    float v = ((float)((i * K + j) % 5) - 1.0f) / 2.0f;
    return v;
  };
  auto getRhs = [K](int i, int j) {
    float v = ((float)((i * K + j) % 7) - 1.0f) / 2.0f;
    return v;
  };

  BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
      device, src0_buffer.get(), src0_size, [&](void *ptr, size_t num_bytes) {
        FillBuffer(input_type, ptr, num_bytes, 1, K, getLhs);
      }));

  // In vmt, the RHS is input is transposed, which makes the matrix
  // column-major.
  BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
      device, src1_buffer.get(), src1_size, [&](void *ptr, size_t num_bytes) {
        FillBuffer(input_type, ptr, num_bytes, N, K, getRhs);
      }));

  //===-------------------------------------------------------------------===/
  // Clear the output buffer data set by the previous benchmark run
  //===-------------------------------------------------------------------===/

  BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
      device, dst_buffer.get(), dst_size, [&](void *ptr, size_t num_bytes) {
        FillBuffer(output_type, ptr, num_bytes, 1, N,
                   [](int, int) { return 0.0f; });
      }));

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
      << "unexpected number of descriptor sets (" << shader.name << ")";
  auto descriptor_set_layout = shader_module->descriptor_set_layouts().front();

  std::vector<::uvkc::vulkan::CommandBuffer::BoundDescriptorSet>
      bound_descriptor_sets(1);
  bound_descriptor_sets[0].index = 0;
  bound_descriptor_sets[0].set = layout_set_map.at(descriptor_set_layout);
  BM_CHECK_OK_AND_ASSIGN(auto dispatch_cmdbuf, device->AllocateCommandBuffer());

  BM_CHECK_OK(dispatch_cmdbuf->Begin());
  dispatch_cmdbuf->BindPipelineAndDescriptorSets(
      *pipeline, {bound_descriptor_sets.data(), bound_descriptor_sets.size()});
  // Each workgroup processes N0 rows with S0 subgroups per row.
  dispatch_cmdbuf->Dispatch(N / shader.N0, 1, 1);
  BM_CHECK_OK(dispatch_cmdbuf->End());
  BM_CHECK_OK(device->QueueSubmitAndWait(*dispatch_cmdbuf));

  //===-------------------------------------------------------------------===/
  // Verify destination buffer data
  //===-------------------------------------------------------------------===/

  if (output_type == DataType::i32) {
    BM_CHECK_OK(::uvkc::benchmark::GetDeviceBufferViaStagingBuffer(
        device, dst_buffer.get(), dst_size, [&](void *ptr, size_t num_bytes) {
          if (input_type == DataType::i8) {
            CheckOutput<DataType::i32, DataType::i8>(shader, ptr, num_bytes, N,
                                                     K, getLhs, getRhs);
          } else {
            BM_CHECK(false) << "Unhandled input type";
          }
        }));
  } else {
    BM_CHECK(false) << "Unhandled output type";
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

    cmdbuf->Dispatch(N / shader.N0, 1, 1);

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

  double numOperation = double(N) * double(K) * 2.;
  state.counters["Ops"] =
      ::benchmark::Counter(numOperation,
                           ::benchmark::Counter::kIsIterationInvariant |
                               ::benchmark::Counter::kIsRate,
                           ::benchmark::Counter::kIs1000);

  // Reset the command pool to release all command buffers in the benchmarking
  // loop to avoid draining GPU resources.
  BM_CHECK_OK(device->ResetCommandPool());
}

// Returns true iff |a| is a multiple of |b|.
static bool isMultipleOf(int a, int b) { return a >= b && a % b == 0; }

namespace uvkc::benchmark {

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

  const int N = 4096;
  const int K = 4096;

  for (const ShaderCode &shader : kShaderCodeCases) {
    std::string vecmat_size = absl::StrCat(N, "x", K);
    std::string tiling_scheme = absl::StrCat(shader.N0, "x", shader.K0);
    BM_CHECK(isMultipleOf(N, shader.N0))
        << "Incompatible tiling scheme: " << tiling_scheme;
    BM_CHECK(isMultipleOf(K, shader.K0))
        << "Incompatible tiling scheme: " << tiling_scheme;
    BM_CHECK(isMultipleOf(shader.K0, 4))
        << "Incompatible tiling scheme: " << tiling_scheme;

    std::string workgroup_size =
        absl::StrCat(shader.wg_size_x, "x", shader.wg_size_y, "x1");
    std::string type_info = absl::StrCat(GetName(shader.input_type), "->",
                                         GetName(shader.output_type));
    std::string test_name =
        absl::StrCat(gpu_name, "/vmt[", vecmat_size, "]/", type_info, "/",
                     shader.name, "/Workgroup[", workgroup_size, "]");
    ::benchmark::RegisterBenchmark(test_name.c_str(), Vmt, device,
                                   latency_measure, shader, N, K)
        ->UseManualTime()
        ->Unit(::benchmark::kMicrosecond);
  }
}

}  // namespace uvkc::benchmark
