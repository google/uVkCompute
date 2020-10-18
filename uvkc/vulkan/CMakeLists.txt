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
    buffer
  HDRS
    "buffer.h"
  SRCS
    "buffer.cc"
  COPTS
    -DVK_NO_PROTOTYPES
  DEPS
    ::dynamic_symbols
    Vulkan::Vulkan
)

uvkc_cc_library(
  NAME
    command_buffer
  HDRS
    "command_buffer.h"
  SRCS
    "command_buffer.cc"
  COPTS
    -DVK_NO_PROTOTYPES
  DEPS
    ::dynamic_symbols
    ::status_util
    ::pipeline
    absl::statusor
    Vulkan::Vulkan
)

uvkc_cc_library(
  NAME
    dynamic_symbols
  HDRS
    "dynamic_symbols.h"
  SRCS
    "dynamic_symbols.cc"
  COPTS
    -DVK_NO_PROTOTYPES
  DEPS
    absl::span
    absl::statusor
    absl::strings
    uvkc::android::vulkan_icd_symbol
    uvkc::base::dynamic_library
    Vulkan::Vulkan
)

uvkc_cc_library(
  NAME
    descriptor_pool
  HDRS
    "descriptor_pool.h"
  SRCS
    "descriptor_pool.cc"
  COPTS
    -DVK_NO_PROTOTYPES
  DEPS
    ::dynamic_symbols
    ::status_util
    absl::span
    absl::status
    absl::statusor
    absl::strings
    Vulkan::Vulkan
)

uvkc_cc_library(
  NAME
    device
  HDRS
    "device.h"
  SRCS
    "device.cc"
  COPTS
    -DVK_NO_PROTOTYPES
  DEPS
    ::buffer
    ::command_buffer
    ::descriptor_pool
    ::dynamic_symbols
    ::image
    ::pipeline
    ::shader_module
    ::timestamp_query_pool
    absl::statusor
    Vulkan::Vulkan
)

uvkc_cc_library(
  NAME
    driver
  HDRS
    "driver.h"
  SRCS
    "driver.cc"
  COPTS
    -DVK_NO_PROTOTYPES
  DEPS
    ::device
    ::dynamic_symbols
    ::status_util
    absl::statusor
    Vulkan::Vulkan
)

uvkc_cc_library(
  NAME
    image
  HDRS
    "image.h"
  SRCS
    "image.cc"
  COPTS
    -DVK_NO_PROTOTYPES
  DEPS
    ::dynamic_symbols
    Vulkan::Vulkan
)

uvkc_cc_library(
  NAME
    pipeline
  HDRS
    "pipeline.h"
  SRCS
    "pipeline.cc"
  COPTS
    -DVK_NO_PROTOTYPES
  DEPS
    ::dynamic_symbols
    ::shader_module
    ::status_util
    absl::span
    absl::statusor
    Vulkan::Vulkan
)

uvkc_cc_library(
  NAME
    pipeline_util
  HDRS
    "pipeline_util.h"
  SRCS
    "pipeline_util.cc"
    "${UVKC_SOURCE_ROOT}/third_party/SPIRV-Reflect/spirv_reflect.h"
    "${UVKC_SOURCE_ROOT}/third_party/SPIRV-Reflect/spirv_reflect.c"
  INCLUDES
    "${UVKC_SOURCE_ROOT}/third_party/SPIRV-Reflect"
  COPTS
    -DVK_NO_PROTOTYPES
  DEPS
    Vulkan::Vulkan
)

uvkc_cc_library(
  NAME
    shader_module
  HDRS
    "shader_module.h"
  SRCS
    "shader_module.cc"
  COPTS
    -DVK_NO_PROTOTYPES
  DEPS
    ::dynamic_symbols
    ::pipeline_util
    ::status_util
    absl::span
    absl::statusor
    Vulkan::Vulkan
)

uvkc_cc_library(
  NAME
    status_util
  HDRS
    "status_util.h"
  SRCS
    "status_util.cc"
  COPTS
    -DVK_NO_PROTOTYPES
  DEPS
    absl::status
    Vulkan::Vulkan
)

uvkc_cc_library(
  NAME
    timestamp_query_pool
  HDRS
    "timestamp_query_pool.h"
  SRCS
    "timestamp_query_pool.cc"
  COPTS
    -DVK_NO_PROTOTYPES
  DEPS
    ::dynamic_symbols
    absl::statusor
    Vulkan::Vulkan
)
