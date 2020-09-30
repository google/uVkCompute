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
#include <iostream>
#include <numeric>

#include "absl/status/status.h"
#include "uvkc/base/status.h"
#include "uvkc/vulkan/device.h"
#include "uvkc/vulkan/driver.h"

static const char kBenchmarkName[] = "copy_storage_buffer_scalar";

static uint32_t kShaderCode[] = {
#include "copy_storage_buffer_scalar_shader_spirv_code.inc"
};

static absl::Status RunBenchmark() {
  //===---------------------------------------------------------------------===/
  // Create driver and devices
  //===---------------------------------------------------------------------===/

  UVKC_ASSIGN_OR_RETURN(auto driver,
                        uvkc::vulkan::Driver::Create(kBenchmarkName));
  UVKC_ASSIGN_OR_RETURN(auto physical_devices,
                        driver->EnumeratePhysicalDevices());

  std::cout << "Benchmarking: " << kBenchmarkName << "\n";

  for (const auto &physical_device : physical_devices) {
    std::cout << "  ---\n";
    std::cout << "  [GPU] " << physical_device.properties.deviceName << "\n";

    uint32_t num_elements = 64;
    size_t buffer_size = num_elements * sizeof(float);

    UVKC_ASSIGN_OR_RETURN(
        auto device,
        driver->CreateDevice(physical_device.handle, VK_QUEUE_COMPUTE_BIT));

    //===-------------------------------------------------------------------===/
    // Create shader module, pipeline, and descriptor sets
    //===-------------------------------------------------------------------===/

    UVKC_ASSIGN_OR_RETURN(
        auto shader_module,
        device->CreateShaderModule(kShaderCode,
                                   sizeof(kShaderCode) / sizeof(uint32_t)));
    UVKC_ASSIGN_OR_RETURN(
        auto pipeline,
        device->CreatePipeline(*shader_module, "main", /*spec_constants=*/{}));

    UVKC_ASSIGN_OR_RETURN(auto descriptor_pool,
                          device->CreateDescriptorPool(*shader_module));
    UVKC_ASSIGN_OR_RETURN(auto layout_set_map,
                          descriptor_pool->AllocateDescriptorSets(
                              shader_module->descriptor_set_layouts()));

    //===-------------------------------------------------------------------===/
    // Create buffers
    //===-------------------------------------------------------------------===/

    UVKC_ASSIGN_OR_RETURN(
        auto src_staging_buffer,
        device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             buffer_size));
    UVKC_ASSIGN_OR_RETURN(
        auto src_buffer,
        device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer_size));
    UVKC_ASSIGN_OR_RETURN(
        auto dst_buffer,
        device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer_size));
    UVKC_ASSIGN_OR_RETURN(
        auto dst_staging_buffer,
        device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             buffer_size));

    //===-------------------------------------------------------------------===/
    // Copy data from host staging buffer to device buffer
    //===-------------------------------------------------------------------===/

    UVKC_ASSIGN_OR_RETURN(void *src_staging_ptr,
                          src_staging_buffer->MapMemory(0, buffer_size));
    float *src_float_buffer = reinterpret_cast<float *>(src_staging_ptr);
    std::iota(src_float_buffer, src_float_buffer + num_elements, 1.0f);
    src_staging_buffer->UnmapMemory();

    UVKC_ASSIGN_OR_RETURN(auto src_copy_cmdbuf,
                          device->AllocateCommandBuffer());
    UVKC_RETURN_IF_ERROR(src_copy_cmdbuf->Begin());
    src_copy_cmdbuf->CopyBuffer(*src_staging_buffer, 0, *src_buffer, 0,
                                buffer_size);
    UVKC_RETURN_IF_ERROR(src_copy_cmdbuf->End());
    UVKC_RETURN_IF_ERROR(device->QueueSubmitAndWait(*src_copy_cmdbuf));

    //===-------------------------------------------------------------------===/
    // Dispatch
    //===-------------------------------------------------------------------===/

    std::vector<uvkc::vulkan::Device::BoundBuffer> bound_buffers(2);
    bound_buffers[0].buffer = src_buffer.get();
    bound_buffers[0].set = 0;
    bound_buffers[0].binding = 0;
    bound_buffers[1].buffer = dst_buffer.get();
    bound_buffers[1].set = 0;
    bound_buffers[1].binding = 1;
    UVKC_RETURN_IF_ERROR(device->AttachBufferToDescriptor(
        *shader_module, layout_set_map,
        {bound_buffers.data(), bound_buffers.size()}));

    if (shader_module->descriptor_set_layouts().size() != 1) {
      return absl::InternalError("unexpected number of descriptor sets");
    }
    auto descriptor_set_layout =
        shader_module->descriptor_set_layouts().front();

    std::vector<uvkc::vulkan::CommandBuffer::BoundDescriptorSet>
        bound_descriptor_sets(1);
    bound_descriptor_sets[0].index = 0;
    bound_descriptor_sets[0].set = layout_set_map.at(descriptor_set_layout);
    UVKC_ASSIGN_OR_RETURN(auto dispatch_cmdbuf,
                          device->AllocateCommandBuffer());

    UVKC_RETURN_IF_ERROR(dispatch_cmdbuf->Begin());
    dispatch_cmdbuf->BindPipelineAndDescriptorSets(
        *pipeline,
        {bound_descriptor_sets.data(), bound_descriptor_sets.size()});
    dispatch_cmdbuf->Dispatch(num_elements / 32, 1, 1);
    UVKC_RETURN_IF_ERROR(dispatch_cmdbuf->End());
    auto start_time = std::chrono::high_resolution_clock::now();
    UVKC_RETURN_IF_ERROR(device->QueueSubmitAndWait(*dispatch_cmdbuf));
    auto end_time = std::chrono::high_resolution_clock::now();
    uint64_t elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              end_time - start_time)
                              .count();
    std::cout << "  [time] " << elapsed_us << "us\n";

    //===-------------------------------------------------------------------===/
    // Copy data from device buffer to host staging buffer
    //===-------------------------------------------------------------------===/

    UVKC_ASSIGN_OR_RETURN(auto dst_copy_cmdbuf,
                          device->AllocateCommandBuffer());
    UVKC_RETURN_IF_ERROR(dst_copy_cmdbuf->Begin());
    dst_copy_cmdbuf->CopyBuffer(*dst_buffer, 0, *dst_staging_buffer, 0,
                                buffer_size);
    UVKC_RETURN_IF_ERROR(dst_copy_cmdbuf->End());
    UVKC_RETURN_IF_ERROR(device->QueueSubmitAndWait(*dst_copy_cmdbuf));

    UVKC_ASSIGN_OR_RETURN(void *dst_staging_ptr,
                          dst_staging_buffer->MapMemory(0, buffer_size));
    float *dst_float_buffer = reinterpret_cast<float *>(dst_staging_ptr);
    std::cout << "  [buffer]";
    for (int i = 0; i < num_elements; ++i) {
      std::cout << " " << dst_float_buffer[i];
    }
    std::cout << "\n";
    dst_staging_buffer->UnmapMemory();
  }

  return absl::OkStatus();
}

int main(int argc, char **argv) {
  auto status = RunBenchmark();
  if (!status.ok()) {
    std::cerr << status;
  }

  return 0;
}
