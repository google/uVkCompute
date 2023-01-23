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
#include "absl/types/span.h"
#include "benchmark/benchmark.h"
#include "uvkc/benchmark/data_type_util.h"
#include "uvkc/benchmark/main.h"
#include "uvkc/benchmark/status_util.h"
#include "uvkc/benchmark/vulkan_buffer_util.h"
#include "uvkc/benchmark/vulkan_context.h"
#include "uvkc/benchmark/vulkan_image_util.h"
#include "uvkc/vulkan/device.h"
#include "uvkc/vulkan/pipeline.h"

using namespace uvkc::benchmark;
using ::uvkc::benchmark::LatencyMeasureMode;
using ::uvkc::vulkan::Pipeline;

static const char kBenchmarkName[] = "matmul_tiled";

struct ShaderCode {
  const char *name;                 // Shader case name
  absl::Span<const uint32_t> code;  // SPIR-V code
  bool texture;                     // Whether texture is used for the RHS
  int tileM;
  int tileN;
  int tileK;
  int wg_size_x;
  int wg_size_y;
  DataType input_type;   // LHS & RHS element type
  DataType output_type;  // Output/Result matrix element type
};

#define SHADER_TILE_F32(M, N, K, X, Y)                                  \
  ShaderCode {                                                          \
    "Tile[" #M "x" #N "x" #K "]",                                       \
        matmul_tiled_f32::                                              \
            TILE_M_##M##_TILE_N_##N##_TILE_K_##K##_WG_X_##X##_WG_Y_##Y, \
        false, M, N, K, X, Y, DataType::fp32, DataType::fp32            \
  }

#define SHADER_TILE_F16_TEX(M, N, K, T, X, Y)                                         \
  ShaderCode {                                                                        \
    "Tile[" #M "x" #N "x" #K "]/Texture=" #T,                                         \
        matmul_tiled_f16::                                                            \
            TILE_M_##M##_TILE_N_##N##_TILE_K_##K##_TEXTURE_##T##_WG_X_##X##_WG_Y_##Y, \
        static_cast<bool>(T), M, N, K, X, Y, DataType::fp16, DataType::fp16           \
  }

#define SHADER_TILE_I8(M, N, K, X, Y)                                   \
  ShaderCode {                                                          \
    "Tile[" #M "x" #N "x" #K "]",                                       \
        matmul_tiled_i8::                                               \
            TILE_M_##M##_TILE_N_##N##_TILE_K_##K##_WG_X_##X##_WG_Y_##Y, \
        false, M, N, K, X, Y, DataType::i8, DataType::i32               \
  }

// Try with and without texture.
#define SHADER_TILE_F16(M, N, K, X, Y) \
  SHADER_TILE_F16_TEX(M, N, K, 1, X, Y), SHADER_TILE_F16_TEX(M, N, K, 0, X, Y)

#define WORKGROUP_TILE_N_F16(X, Y, N)                                  \
  SHADER_TILE_F16(2, N, 4, X, Y), SHADER_TILE_F16(4, N, 4, X, Y),      \
      SHADER_TILE_F16(8, N, 4, X, Y), SHADER_TILE_F16(16, N, 4, X, Y), \
      SHADER_TILE_F16(32, N, 4, X, Y), SHADER_TILE_F16(2, N, 8, X, Y), \
      SHADER_TILE_F16(4, N, 8, X, Y), SHADER_TILE_F16(8, N, 8, X, Y),  \
      SHADER_TILE_F16(16, N, 8, X, Y), SHADER_TILE_F16(32, N, 8, X, Y)

#define WORKGROUP_TILE_N_F32(X, Y, N)                                  \
  SHADER_TILE_F32(2, N, 4, X, Y), SHADER_TILE_F32(4, N, 4, X, Y),      \
      SHADER_TILE_F32(8, N, 4, X, Y), SHADER_TILE_F32(16, N, 4, X, Y), \
      SHADER_TILE_F32(32, N, 4, X, Y), SHADER_TILE_F32(2, N, 8, X, Y), \
      SHADER_TILE_F32(4, N, 8, X, Y), SHADER_TILE_F32(8, N, 8, X, Y),  \
      SHADER_TILE_F32(16, N, 8, X, Y), SHADER_TILE_F32(32, N, 8, X, Y)

#define WORKGROUP_TILE_N_I8(X, Y, N)                                 \
  SHADER_TILE_I8(2, N, 4, X, Y), SHADER_TILE_I8(4, N, 4, X, Y),      \
      SHADER_TILE_I8(8, N, 4, X, Y), SHADER_TILE_I8(16, N, 4, X, Y), \
      SHADER_TILE_I8(32, N, 4, X, Y), SHADER_TILE_I8(2, N, 8, X, Y), \
      SHADER_TILE_I8(4, N, 8, X, Y), SHADER_TILE_I8(8, N, 8, X, Y),  \
      SHADER_TILE_I8(16, N, 8, X, Y), SHADER_TILE_I8(32, N, 8, X, Y)

#if defined(UVKC_ADRENO)

namespace matmul_tiled_f32 {
#include "matmul_tiled_shader_f32_adreno_spirv_permutation.inc"
}

namespace matmul_tiled_f16 {
#include "matmul_tiled_shader_f16_adreno_spirv_permutation.inc"
}

namespace matmul_tiled_i8 {
#include "matmul_tiled_shader_i8_adreno_spirv_permutation.inc"
}

static ShaderCode kShaderCodeCases[] = {
    WORKGROUP_TILE_N_F32(32, 2, 128), WORKGROUP_TILE_N_F32(32, 2, 256),
    WORKGROUP_TILE_N_F16(32, 2, 128), WORKGROUP_TILE_N_F16(32, 2, 256),
    WORKGROUP_TILE_N_I8(32, 2, 128),
};

#elif defined(UVKC_MALI_VALHALL)

namespace matmul_tiled_f32 {
#include "matmul_tiled_shader_f32_valhall_spirv_permutation.inc"
}
namespace matmul_tiled_f16 {
#include "matmul_tiled_shader_f16_valhall_spirv_permutation.inc"
}

namespace matmul_tiled_i8 {
#include "matmul_tiled_shader_i8_valhall_spirv_permutation.inc"
}

static ShaderCode kShaderCodeCases[] = {
    WORKGROUP_TILE_N_F32(16, 1, 64), WORKGROUP_TILE_N_F32(16, 1, 128),
    WORKGROUP_TILE_N_F16(8, 2, 64),  WORKGROUP_TILE_N_F16(8, 2, 128),
    WORKGROUP_TILE_N_I8(16, 1, 64),  WORKGROUP_TILE_N_I8(16, 1, 128),
};

#else
#error "unsupported GPU architecture"
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

/// Checks that the output 2D matrix calculated by the shader is contains the
/// same values as runtime matmul of matrices with values defined by |lhs| and
/// |rhs|.
template <DataType OutputType, DataType InputType, typename Generator1Fn,
          typename Generator2Fn>
static void CheckOutput(const ShaderCode &shader, void *raw_buffer,
                        size_t num_bytes, unsigned M, unsigned N, unsigned K,
                        Generator1Fn lhs, Generator2Fn rhs) {
  using OutputTraits = DataTypeTraits<OutputType>;
  using OutputStorageType = typename OutputTraits::storage_type;
  using OutputRuntimeType = typename OutputTraits::runtime_type;
  using InputTraits = DataTypeTraits<InputType>;
  using InputRuntimeType = typename InputTraits::runtime_type;

  auto output =
      absl::MakeConstSpan(static_cast<OutputStorageType *>(raw_buffer),
                          num_bytes / GetSize(OutputType));
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      OutputRuntimeType acc(0.0f);
      for (int k = 0; k < K; ++k) {
        acc += OutputRuntimeType(InputRuntimeType(lhs(i, k))) *
               OutputRuntimeType(InputRuntimeType(rhs(k, j)));
      }

      OutputRuntimeType gpuValue(output[i * N + j]);
      BM_CHECK_EQ(gpuValue, acc)
          << "destination buffer element (" << i << "," << j << ")"
          << " has incorrect value: expected to be " << acc << " but found "
          << gpuValue << "\n\t^ In shader: " << shader.name << ", "
          << GetName(shader.input_type) << "->" << GetName(shader.output_type);
    }
  }
}

static void MatMul(::benchmark::State &state, ::uvkc::vulkan::Device *device,
                   const ::uvkc::benchmark::LatencyMeasure *latency_measure,
                   const ShaderCode &shader, int M, int N, int K) {
  //===-------------------------------------------------------------------===/
  // Create shader module, pipeline, and descriptor sets
  //===-------------------------------------------------------------------===/

  BM_CHECK_OK_AND_ASSIGN(
      auto shader_module,
      device->CreateShaderModule(shader.code.data(), shader.code.size()));

  ::uvkc::vulkan::Pipeline::SpecConstant spec_constant[3] = {
      {/*id=*/0, Pipeline::SpecConstant::Type::s32, M},
      {/*id=*/1, Pipeline::SpecConstant::Type::s32, N},
      {/*id=*/2, Pipeline::SpecConstant::Type::s32, K},
  };
  BM_CHECK_OK_AND_ASSIGN(
      auto pipeline, device->CreatePipeline(*shader_module, "main",
                                            absl::MakeSpan(spec_constant, 3)));

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
  const size_t src0_size = M * K * GetSize(input_type);
  const size_t src1_size = K * N * GetSize(input_type);
  const size_t dst_size = M * N * GetSize(output_type);

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
  auto getSrc0 = [K](int i, int j) {
    float v = ((float)((i + j * K) % 5) - 1.0f) / 2.0f;
    return v;
  };
  auto getSrc1 = [N](int i, int j) {
    float v = ((float)((i + j * N) % 7) - 1.0f) / 2.0f;
    return v;
  };

  BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
      device, src0_buffer.get(), src0_size, [&](void *ptr, size_t num_bytes) {
        FillBuffer(input_type, ptr, num_bytes, M, K, getSrc0);
      }));

  BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
      device, src1_buffer.get(), src1_size, [&](void *ptr, size_t num_bytes) {
        FillBuffer(input_type, ptr, num_bytes, K, N, getSrc1);
      }));

  if (shader.texture) {
    BM_CHECK_OK(::uvkc::benchmark::SetDeviceImageViaStagingBuffer(
        device, src_image1.get(), dimensions1,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, src1_size,
        [&](void *ptr, size_t num_bytes) {
          FillBuffer(input_type, ptr, num_bytes, K, N, getSrc1);
        }));
  }

  //===-------------------------------------------------------------------===/
  // Dispatch
  //===-------------------------------------------------------------------===/
  std::vector<::uvkc::vulkan::Device::BoundBuffer> bound_buffers;
  if (shader.texture) {
    // For simplicity always bind the B matrix as both texture and buffer.
    std::vector<::uvkc::vulkan::Device::BoundImage> bound_images = {
        {src_image1.get(), src_sampler1.get(), /*set=*/0, /*binding=*/3}};
    BM_CHECK_OK(device->AttachImageToDescriptor(
        *shader_module, layout_set_map,
        {bound_images.data(), bound_images.size()}));
    bound_buffers = {
        {src0_buffer.get(), /*set=*/0, /*binding=*/0},
        {dst_buffer.get(), /*set=*/0, /*binding=*/2},
    };
  } else {
    bound_buffers = {
        {src0_buffer.get(), /*set=*/0, /*binding=*/0},
        {src1_buffer.get(), /*set=*/0, /*binding=*/1},
        {dst_buffer.get(), /*set=*/0, /*binding=*/2},
    };
  }
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
  dispatch_cmdbuf->Dispatch(N / shader.tileN, M / shader.tileM, 1);
  BM_CHECK_OK(dispatch_cmdbuf->End());
  BM_CHECK_OK(device->QueueSubmitAndWait(*dispatch_cmdbuf));

  //===-------------------------------------------------------------------===/
  // Verify destination buffer data
  //===-------------------------------------------------------------------===/

  if (output_type == DataType::fp16) {
    BM_CHECK_OK(::uvkc::benchmark::GetDeviceBufferViaStagingBuffer(
        device, dst_buffer.get(), dst_size, [&](void *ptr, size_t num_bytes) {
          CheckOutput<DataType::fp16, DataType::fp16>(shader, ptr, num_bytes, M,
                                                      N, K, getSrc0, getSrc1);
        }));
  } else if (output_type == DataType::fp32) {
    BM_CHECK_OK(::uvkc::benchmark::GetDeviceBufferViaStagingBuffer(
        device, dst_buffer.get(), dst_size, [&](void *ptr, size_t num_bytes) {
          CheckOutput<DataType::fp32, DataType::fp32>(shader, ptr, num_bytes, M,
                                                      N, K, getSrc0, getSrc1);
        }));
  } else if (output_type == DataType::i32) {
    BM_CHECK_OK(::uvkc::benchmark::GetDeviceBufferViaStagingBuffer(
        device, dst_buffer.get(), dst_size, [&](void *ptr, size_t num_bytes) {
          CheckOutput<DataType::i32, DataType::i8>(shader, ptr, num_bytes, M, N,
                                                   K, getSrc0, getSrc1);
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

    cmdbuf->Dispatch(N / shader.tileN, M / shader.tileM, 1);

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

  double numOperation = double(N) * double(M) * double(K) * 2.;
  state.counters["FLOps"] =
      ::benchmark::Counter(numOperation,
                           ::benchmark::Counter::kIsIterationInvariant |
                               ::benchmark::Counter::kIsRate,
                           ::benchmark::Counter::kIs1000);

  // Reset the command pool to release all command buffers in the benchmarking
  // loop to avoid draining GPU resources.
  BM_CHECK_OK(device->ResetCommandPool());
}

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

  const int M = 1024;
  const int N = 1024;
  const int K = 1024;

  for (DataType input_type : {DataType::fp32, DataType::i8, DataType::fp16}) {
    for (const ShaderCode &shader : kShaderCodeCases) {
      if (shader.input_type != input_type) continue;
      int paddM = (M + shader.tileM - 1) / shader.tileM * shader.tileM;
      int paddN = (N + shader.tileN - 1) / shader.tileN * shader.tileN;
      std::string matmul_size = absl::StrCat(M, "x", N, "x", K);
      std::string tiling_scheme =
          absl::StrCat(shader.tileM, "x", shader.tileN, "x", shader.tileK);
      std::string workgroup_size =
          absl::StrCat(shader.wg_size_x, "x", shader.wg_size_y, "x1");
      std::string type_info = absl::StrCat(GetName(shader.input_type), "->",
                                           GetName(shader.output_type));
      std::string test_name =
          absl::StrCat(gpu_name, "/Matmul[", matmul_size, "]/", type_info, "/",
                       shader.name, "/Workgroup[", workgroup_size, "]");
      ::benchmark::RegisterBenchmark(test_name.c_str(), MatMul, device,
                                     latency_measure, shader, paddM, paddN, K)
          ->UseManualTime()
          ->Unit(::benchmark::kMicrosecond);
    }
  }
}

}  // namespace uvkc::benchmark
