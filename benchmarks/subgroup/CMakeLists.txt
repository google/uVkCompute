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
    subgroup_add_loop
  SRC
    "subgroup_arithmetic_loop.glsl"
  GLSLC_ARGS
    "--target-env=vulkan1.1"
    "-DARITHMETIC_ADD"
)

uvkc_glsl_shader_instance(
  NAME
    subgroup_add_intrinsic
  SRC
    "subgroup_arithmetic_intrinsic.glsl"
  GLSLC_ARGS
    "--target-env=vulkan1.1"
    "-DARITHMETIC_ADD"
)

uvkc_glsl_shader_instance(
  NAME
    subgroup_mul_loop
  SRC
    "subgroup_arithmetic_loop.glsl"
  GLSLC_ARGS
    "--target-env=vulkan1.1"
    "-DARITHMETIC_MUL"
)

uvkc_glsl_shader_instance(
  NAME
    subgroup_mul_intrinsic
  SRC
    "subgroup_arithmetic_intrinsic.glsl"
  GLSLC_ARGS
    "--target-env=vulkan1.1"
    "-DARITHMETIC_MUL"
)

uvkc_cc_binary(
  NAME
    subgroup_arithmetic
  SRCS
    "subgroup_arithmetic_main.cc"
  DEPS
    ::subgroup_add_intrinsic
    ::subgroup_add_loop
    ::subgroup_mul_intrinsic
    ::subgroup_mul_loop
    benchmark::benchmark
    benchmarks::memory::copy_storage_buffer_lib
    uvkc::benchmark::main
    uvkc::benchmark::vulkan_buffer_util
)

