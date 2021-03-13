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

cmake_minimum_required(VERSION 3.13)

#-------------------------------------------------------------------------------
# Overall project settings
#-------------------------------------------------------------------------------

project(uVkCompute CXX C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(UVKC_SOURCE_ROOT ${PROJECT_SOURCE_DIR})

#-------------------------------------------------------------------------------
# CMake definitions
#-------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH
  ${UVKC_SOURCE_ROOT}/cmake/
  ${UVKC_SOURCE_ROOT}/third_party/abseil-cpp/absl/copts/
  )

include(AbseilConfigureCopts)

# By default Abseil strips string literals on mobile platforms, which means
# we cannot run binaries via command-line with proper options. Turn off
# the stripping.
if(ANDROID)
  add_definitions(-DABSL_FLAGS_STRIP_NAMES=0)
endif()

include(uvkc_find_vulkan_toolchain)

include(uvkc_utils)
include(uvkc_cc_library)
include(uvkc_cc_binary)
include(uvkc_glsl_shader_instance)
include(uvkc_glsl_shader_permutation)

#-------------------------------------------------------------------------------
# Third-party dependencies
#-------------------------------------------------------------------------------

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_subdirectory(third_party)

#-------------------------------------------------------------------------------
# Core libraries
#-------------------------------------------------------------------------------

add_subdirectory(uvkc/base)
add_subdirectory(uvkc/android)
add_subdirectory(uvkc/benchmark)
add_subdirectory(uvkc/vulkan)

#-------------------------------------------------------------------------------
# Utility tools
#-------------------------------------------------------------------------------

add_subdirectory(tools)

#-------------------------------------------------------------------------------
# Benchmark binaries
#-------------------------------------------------------------------------------

add_subdirectory(benchmarks)
