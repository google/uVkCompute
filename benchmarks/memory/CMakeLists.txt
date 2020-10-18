# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

uvkc_glsl_shader_instance(
  NAME
    copy_storage_buffer_scalar_shader
  SRC
    "copy_storage_buffer_scalar.glsl"
)

uvkc_glsl_shader_instance(
  NAME
    copy_storage_buffer_vector_shader
  SRC
    "copy_storage_buffer_vector.glsl"
)

uvkc_cc_library(
  NAME
    copy_storage_buffer_lib
  HDRS
    "copy_storage_buffer.h"
  SRCS
    "copy_storage_buffer.cc"
  DEPS
    ::copy_storage_buffer_scalar_shader
    ::copy_storage_buffer_vector_shader
    benchmark::benchmark
    uvkc::benchmark::vulkan_buffer_util
)

uvkc_cc_binary(
  NAME
    copy_storage_buffer
  SRCS
    "copy_storage_buffer_main.cc"
  DEPS
    ::copy_storage_buffer_lib
    uvkc::benchmark::main
)

uvkc_glsl_shader_instance(
  NAME
    copy_sampled_image_to_storage_buffer_shader
  SRC
    "copy_sampled_image_to_storage_buffer.glsl"
)

uvkc_cc_binary(
  NAME
    copy_sampled_image_to_storage_buffer
  SRCS
    "copy_sampled_image_to_storage_buffer_main.cc"
  DEPS
    ::copy_sampled_image_to_storage_buffer_shader
    benchmark::benchmark
    uvkc::benchmark::main
    uvkc::benchmark::vulkan_buffer_util
    uvkc::benchmark::vulkan_image_util
)
