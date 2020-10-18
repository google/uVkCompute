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
#include "uvkc/benchmark/vulkan_image_util.h"
#include "uvkc/vulkan/device.h"
#include "uvkc/vulkan/pipeline.h"
#include "uvkc/vulkan/timestamp_query_pool.h"

using ::uvkc::benchmark::LatencyMeasureMode;

static const char kBenchmarkName[] = "copy_image_to_buffer";

static uint32_t kShaderCode[] = {
#include "copy_sampled_image_to_storage_buffer_shader_spirv_instance.inc"
};

struct ShaderCodeCase {
  const char *name;       // Test case name
  const uint32_t *code;   // SPIR-V code
  size_t code_num_bytes;  // Number of bytes for SPIR-V code
};

static ShaderCodeCase kShaderCodeCases[] = {
    {"ImageLoad", kShaderCode, sizeof(kShaderCode)},
};

static void CopyImageToBuffer(
    ::benchmark::State &state, ::uvkc::vulkan::Device *device,
    ::uvkc::benchmark::LatencyMeasureMode latency_measure_mode,
    const double *overhead_latency_seconds, const uint32_t *code,
    size_t code_num_words, uint32_t image_width, uint32_t image_height) {
  uint32_t buffer_num_bytes = image_width * image_height * sizeof(float);

  //===-------------------------------------------------------------------===/
  // Create shader module, pipeline, and descriptor sets
  //===-------------------------------------------------------------------===/

  BM_CHECK_OK_AND_ASSIGN(auto shader_module,
                         device->CreateShaderModule(code, code_num_words));

  BM_CHECK_OK_AND_ASSIGN(
      auto pipeline,
      device->CreatePipeline(*shader_module, "main", /*spec_constant=*/{}));

  BM_CHECK_OK_AND_ASSIGN(auto descriptor_pool,
                         device->CreateDescriptorPool(*shader_module));
  BM_CHECK_OK_AND_ASSIGN(auto layout_set_map,
                         descriptor_pool->AllocateDescriptorSets(
                             shader_module->descriptor_set_layouts()));

  //===-------------------------------------------------------------------===/
  // Create images/samplers/buffers
  //===-------------------------------------------------------------------===/

  VkExtent3D dimensions = {image_width, image_height, 1};
  BM_CHECK_OK_AND_ASSIGN(
      auto src_image,
      device->CreateImage(
          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_TYPE_2D,
          VK_FORMAT_R32_SFLOAT, dimensions, VK_IMAGE_TILING_OPTIMAL,
          VK_IMAGE_VIEW_TYPE_2D));
  BM_CHECK_OK_AND_ASSIGN(auto src_sampler, device->CreateSampler());
  BM_CHECK_OK_AND_ASSIGN(
      auto dst_buffer,
      device->CreateBuffer(
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer_num_bytes));

  //===-------------------------------------------------------------------===/
  // Set source buffer data
  //===-------------------------------------------------------------------===/

  BM_CHECK_OK(::uvkc::benchmark::SetDeviceImageViaStagingBuffer(
      device, src_image.get(), dimensions,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, buffer_num_bytes,
      [](void *ptr, size_t num_bytes) {
        float *src_float_buffer = reinterpret_cast<float *>(ptr);
        std::iota(src_float_buffer,
                  src_float_buffer + num_bytes / sizeof(float), 0.0f);
      }));

  //===-------------------------------------------------------------------===/
  // Dispatch
  //===-------------------------------------------------------------------===/

  std::vector<::uvkc::vulkan::Device::BoundImage> bound_images = {
      {src_image.get(), src_sampler.get(), /*set=*/0, /*binding=*/0}};
  BM_CHECK_OK(device->AttachImageToDescriptor(
      *shader_module, layout_set_map,
      {bound_images.data(), bound_images.size()}));

  std::vector<::uvkc::vulkan::Device::BoundBuffer> bound_buffers = {
      {dst_buffer.get(), /*set=*/0, /*binding=*/1}};
  BM_CHECK_OK(device->AttachBufferToDescriptor(
      *shader_module, layout_set_map,
      {bound_buffers.data(), bound_buffers.size()}));

  BM_CHECK_EQ(shader_module->descriptor_set_layouts().size(), 1)
      << "unexpected number of descriptor sets";
  auto descriptor_set_layout = shader_module->descriptor_set_layouts().front();

  std::vector<::uvkc::vulkan::CommandBuffer::BoundDescriptorSet>
      bound_descriptor_sets = {
          {/*index=*/0, /*set=*/layout_set_map.at(descriptor_set_layout)}};
  BM_CHECK_OK_AND_ASSIGN(auto dispatch_cmdbuf, device->AllocateCommandBuffer());

  BM_CHECK_OK(dispatch_cmdbuf->Begin());
  dispatch_cmdbuf->BindPipelineAndDescriptorSets(
      *pipeline, {bound_descriptor_sets.data(), bound_descriptor_sets.size()});
  dispatch_cmdbuf->Dispatch(image_width / 16, image_height / 16, 1);
  BM_CHECK_OK(dispatch_cmdbuf->End());
  BM_CHECK_OK(device->QueueSubmitAndWait(*dispatch_cmdbuf));

  //===-------------------------------------------------------------------===/
  // Verify destination buffer data
  //===-------------------------------------------------------------------===/

  BM_CHECK_OK(::uvkc::benchmark::GetDeviceBufferViaStagingBuffer(
      device, dst_buffer.get(), buffer_num_bytes,
      [](void *ptr, size_t num_bytes) {
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

  std::unique_ptr<::uvkc::vulkan::TimestampQueryPool> query_pool;
  bool use_timestamp =
      latency_measure_mode == LatencyMeasureMode::kGpuTimestamp;
  if (use_timestamp) {
    BM_CHECK_OK_AND_ASSIGN(query_pool, device->CreateTimestampQueryPool(2));
  }

  BM_CHECK_OK_AND_ASSIGN(auto cmdbuf, device->AllocateCommandBuffer());
  double total_seconds = 0;
  for (auto _ : state) {
    BM_CHECK_OK(cmdbuf->Begin());
    if (use_timestamp) cmdbuf->ResetQueryPool(*query_pool);

    cmdbuf->BindPipelineAndDescriptorSets(
        *pipeline,
        {bound_descriptor_sets.data(), bound_descriptor_sets.size()});

    if (use_timestamp) {
      cmdbuf->WriteTimestamp(*query_pool, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0);
    }

    cmdbuf->Dispatch(image_width / 16, image_height / 16, 1);

    if (use_timestamp) {
      cmdbuf->WriteTimestamp(*query_pool, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             1);
    }

    BM_CHECK_OK(cmdbuf->End());

    auto start_time = std::chrono::high_resolution_clock::now();
    BM_CHECK_OK(device->QueueSubmitAndWait(*cmdbuf));
    auto end_time = std::chrono::high_resolution_clock::now();
    auto cpu_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);

    double iteration_seconds = 0;
    switch (latency_measure_mode) {
      case LatencyMeasureMode::kSystemDispatch: {
        iteration_seconds = cpu_seconds.count() - *overhead_latency_seconds;
      } break;
      case LatencyMeasureMode::kSystemSubmit: {
        iteration_seconds = cpu_seconds.count();
      } break;
      case LatencyMeasureMode::kGpuTimestamp: {
        BM_CHECK_OK_AND_ASSIGN(
            iteration_seconds,
            query_pool->CalculateElapsedSecondsBetween(0, 1));
      } break;
    }
    state.SetIterationTime(iteration_seconds);
    total_seconds += iteration_seconds;

    BM_CHECK_OK(cmdbuf->Reset());
  }
  state.SetBytesProcessed(state.iterations() * buffer_num_bytes * 2);  // R + W

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

  // TODO: respect VkImageFormatProperties::maxExtent
  for (uint32_t width = (1 << 10); width < (1 << 13); width <<= 1) {  // 1/2/4K
    for (uint32_t height = (1 << 10); height < (1 << 13); height <<= 1) {
      for (const auto &shader : kShaderCodeCases) {
        std::string test_name =
            absl::StrCat(gpu_name, "/", shader.name, "/", width, "x", height);
        ::benchmark::RegisterBenchmark(
            test_name.c_str(), CopyImageToBuffer, device, latency_measure->mode,
            &latency_measure->overhead_seconds, shader.code,
            shader.code_num_bytes / sizeof(uint32_t), width, height)
            ->UseManualTime()
            ->Unit(::benchmark::kMicrosecond);
      }
    }
  }
}

}  // namespace benchmark
}  // namespace uvkc
