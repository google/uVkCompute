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

static const char kBenchmarkName[] = "copy_storage_buffer_scalar";

static uint32_t kScalarShaderCode[] = {
#include "copy_storage_buffer_scalar_shader_spirv_instance.inc"
};

static uint32_t kVectorShaderCode[] = {
#include "copy_storage_buffer_vector_shader_spirv_instance.inc"
};

struct ShaderCode {
  const uint32_t *code;
  size_t size;
  const char *name;
};

static ShaderCode kShaderCodeCases[] = {
    {kScalarShaderCode, sizeof(kScalarShaderCode) / sizeof(uint32_t), "scalar"},
    {kVectorShaderCode, sizeof(kVectorShaderCode) / sizeof(uint32_t), "vector"},
};

static void CopyStorageBufferScalar(::benchmark::State &state,
                                    ::uvkc::vulkan::Device *device,
                                    const uint32_t *code, size_t size,
                                    int num_elements) {
  size_t buffer_size = num_elements * sizeof(float);

  //===-------------------------------------------------------------------===/
  // Create shader module, pipeline, and descriptor sets
  //===-------------------------------------------------------------------===/

  BM_CHECK_OK_AND_ASSIGN(auto shader_module,
                         device->CreateShaderModule(code, size));

  ::uvkc::vulkan::Pipeline::SpecConstant spec_constant = {};
  spec_constant.id = 0;
  spec_constant.type = ::uvkc::vulkan::Pipeline::SpecConstant::Type::s32;
  spec_constant.value.s32 = num_elements;
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
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer_size));
  BM_CHECK_OK_AND_ASSIGN(
      auto dst_buffer,
      device->CreateBuffer(
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer_size));

  //===-------------------------------------------------------------------===/
  // Set source buffer data
  //===-------------------------------------------------------------------===/

  BM_CHECK_OK(::uvkc::benchmark::SetDeviceBufferViaStagingBuffer(
      device, src_buffer.get(), buffer_size, [](void *ptr, size_t num_bytes) {
        float *src_float_buffer = reinterpret_cast<float *>(ptr);
        std::iota(src_float_buffer,
                  src_float_buffer + num_bytes / sizeof(float), 0.0f);
      }));

  //===-------------------------------------------------------------------===/
  // Dispatch
  //===-------------------------------------------------------------------===/

  std::vector<::uvkc::vulkan::Device::BoundBuffer> bound_buffers(2);
  bound_buffers[0].buffer = src_buffer.get();
  bound_buffers[0].set = 0;
  bound_buffers[0].binding = 0;
  bound_buffers[1].buffer = dst_buffer.get();
  bound_buffers[1].set = 0;
  bound_buffers[1].binding = 1;
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
  dispatch_cmdbuf->Dispatch(num_elements / 32, 1, 1);
  BM_CHECK_OK(dispatch_cmdbuf->End());
  BM_CHECK_OK(device->QueueSubmitAndWait(*dispatch_cmdbuf));

  //===-------------------------------------------------------------------===/
  // Verify destination buffer data
  //===-------------------------------------------------------------------===/

  BM_CHECK_OK(::uvkc::benchmark::GetDeviceBufferViaStagingBuffer(
      device, dst_buffer.get(), buffer_size, [](void *ptr, size_t num_bytes) {
        float *dst_float_buffer = reinterpret_cast<float *>(ptr);
        for (int i = 0; i < num_bytes / sizeof(float); ++i) {
          BM_CHECK_EQ(dst_float_buffer[i], i)
              << "destination buffer element #" << i
              << " has incorrect value: expected to be " << i << " but found "
              << dst_float_buffer[i];
        }
      }));

  //===-------------------------------------------------------------------===/
  // Benchmarking
  //===-------------------------------------------------------------------===/
  for (auto _ : state) {
    BM_CHECK_OK_AND_ASSIGN(auto cmdbuf, device->AllocateCommandBuffer());

    BM_CHECK_OK(cmdbuf->Begin());
    cmdbuf->BindPipelineAndDescriptorSets(
        *pipeline,
        {bound_descriptor_sets.data(), bound_descriptor_sets.size()});
    cmdbuf->Dispatch(num_elements / 32, 1, 1);
    BM_CHECK_OK(cmdbuf->End());
    auto start_time = std::chrono::high_resolution_clock::now();
    BM_CHECK_OK(device->QueueSubmitAndWait(*cmdbuf));
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);
    state.SetIterationTime(elapsed_seconds.count());
  }
  state.SetBytesProcessed(buffer_size);
}

namespace uvkc {
namespace benchmark {

absl::StatusOr<std::unique_ptr<VulkanContext>> CreateVulkanContext() {
  return CreateDefaultVulkanContext(kBenchmarkName);
}

void RegisterVulkanBenchmarks(VulkanContext *context) {
  for (int di = 0; di < context->devices.size(); ++di) {  // GPU
    const char *gpu_name = context->physical_devices[di].properties.deviceName;
    for (const auto &shader : kShaderCodeCases) {  // Scalar/vector shader
      for (int shift = 20; shift < 26; ++shift) {  // Number of bytes: 1M -> 32M
        int num_bytes = 1 << shift;
        std::string test_name =
            absl::StrCat(gpu_name, "/", shader.name, "/", num_bytes);
        ::benchmark::RegisterBenchmark(
            test_name.c_str(), CopyStorageBufferScalar,
            context->devices[di].get(), shader.code, shader.size, num_bytes / 4)
            ->UseManualTime()
            ->Unit(::benchmark::kMicrosecond);
      }
    }
  }
}

}  // namespace benchmark
}  // namespace uvkc
