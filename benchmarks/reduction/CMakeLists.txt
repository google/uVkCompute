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

uvkc_glsl_shader_permutation(
  NAME
    tree_reduce_loop_shader
  SRC
    "tree_reduce_loop.glsl"
  PERMUTATION
    "BATCH_SIZE=[16|32|64|128]"
)

uvkc_glsl_shader_permutation(
  NAME
    tree_reduce_subgroup_shader
  SRC
    "tree_reduce_subgroup.glsl"
  PERMUTATION
    "BATCH_SIZE=[16|32|64|128]"
  GLSLC_ARGS
    "--target-env=vulkan1.1"
)

uvkc_cc_binary(
  NAME
    tree_reduce
  SRCS
    "tree_reduce_main.cc"
  DEPS
    ::tree_reduce_loop_shader
    ::tree_reduce_subgroup_shader
    benchmark::benchmark
    uvkc::benchmark::main
    uvkc::benchmark::vulkan_buffer_util
)
