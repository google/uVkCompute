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
    matmul_tiled_shader
  SRC
    "matmul_tiled.glsl"
  PERMUTATION
    "TILE_M=[2|4|8|16|32]"
    "TILE_N=[64|128]"
    "TILE_K=[4|8]"
)

uvkc_glsl_shader_permutation(
  NAME
    matmul_tiled_shader_f16
  SRC
    "matmul_tiled_fp16.glsl"
  PERMUTATION
    "TILE_M=[2|4|8|16|32]"
    "TILE_N=[64|128]"
    "TILE_K=[4|8]"
    "TEXTURE=[1|0]"
)

uvkc_cc_binary(
  NAME
    matmul_tiled
  SRCS
    "matmul_tiled_main.cc"
  DEPS
    ::matmul_tiled_shader
    ::matmul_tiled_shader_f16
    benchmark::benchmark
    uvkc::benchmark::fp16_util
    uvkc::benchmark::main
    uvkc::benchmark::vulkan_buffer_util
    uvkc::benchmark::vulkan_image_util
)

