# Copyright 2020-2023 Google LLC
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

uvkc_cc_library(
  NAME
    core
  HDRS
    "data_type_util.h"
    "status_util.h"
    "vulkan_buffer_util.h"
    "vulkan_context.h"
    "vulkan_image_util.h"
  SRCS
    "data_type_util.cc"
    "status_util.cc"
    "vulkan_buffer_util.cc"
    "vulkan_context.cc"
    "vulkan_image_util.cc"
  DEPS
    absl::status
    absl::statusor
    uvkc::base::log
    uvkc::vulkan::buffer
    uvkc::vulkan::device
    uvkc::vulkan::driver
    uvkc::vulkan::image
)

uvkc_glsl_shader_instance(
  NAME
    void_shader
  SRC
    "void_shader.glsl"
)

uvkc_cc_library(
  NAME
    dispatch_void_shader
  SRCS
    "dispatch_void_shader.cc"
  DEPS
    ::void_shader
    benchmark::benchmark
    uvkc::vulkan::device
    uvkc::vulkan::driver
)

uvkc_cc_library(
  NAME
    main
  HDRS
    "main.h"
  SRCS
    "main.cc"
  DEPS
    ::core
    ::dispatch_void_shader
    absl::flags
    absl::flags_parse
    benchmark::benchmark
    uvkc::base::log
)
