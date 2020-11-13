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
    fp16_util
  HDRS
    "fp16_util.h"
  SRCS
    "fp16_util.cc"
)

uvkc_cc_library(
  NAME
    main
  HDRS
    "main.h"
  SRCS
    "main.cc"
  DEPS
    ::dispatch_void_shader
    ::status_util
    ::vulkan_context
    absl::flags
    absl::flags_parse
    benchmark::benchmark
    uvkc::base::log
)

uvkc_cc_library(
  NAME
    status_util
  HDRS
    "status_util.h"
  SRCS
    "status_util.cc"
  DEPS
    absl::status
    absl::statusor
    uvkc::base::log
)

uvkc_glsl_shader_instance(
  NAME
    void_shader
  SRC
    "void_shader.glsl"
)

uvkc_cc_library(
  NAME
    vulkan_buffer_util
  HDRS
    "vulkan_buffer_util.h"
  SRCS
    "vulkan_buffer_util.cc"
  DEPS
    absl::status
    uvkc::vulkan::buffer
    uvkc::vulkan::device
)

uvkc_cc_library(
  NAME
    vulkan_context
  HDRS
    "vulkan_context.h"
  SRCS
    "vulkan_context.cc"
  DEPS
    absl::statusor
    uvkc::vulkan::device
    uvkc::vulkan::driver
)

uvkc_cc_library(
  NAME
    vulkan_image_util
  HDRS
    "vulkan_image_util.h"
  SRCS
    "vulkan_image_util.cc"
  DEPS
    absl::status
    uvkc::vulkan::device
    uvkc::vulkan::image
)
