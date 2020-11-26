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
    conv2d_tiled_shader
  SRC
    "conv2d_tiled.glsl"
  PERMUTATION
    "WG_TILE_OW=[2|4|8|16|32]"
    "WG_TILE_OC=[32|64]"
)

uvkc_cc_binary(
  NAME
    conv2d
  SRCS
    "conv2d_main.cc"
  DEPS
    ::conv2d_tiled_shader
    benchmark::benchmark
    uvkc::benchmark::main
    uvkc::benchmark::vulkan_buffer_util
)

uvkc_glsl_shader_permutation(
  NAME
    depthwise_conv2d_tiled_shader
  SRC
    "depthwise_conv2d_tiled.glsl"
  PERMUTATION
    "WG_TILE_OH=[1|2|4|8]"
    "WG_TILE_OW=[1|2|4|8]"
    "WG_TILE_OC=[16|32|64]"
)

uvkc_cc_binary(
  NAME
    depthwise_conv2d
  SRCS
    "depthwise_conv2d_main.cc"
  DEPS
    ::depthwise_conv2d_tiled_shader
    benchmark::benchmark
    uvkc::benchmark::main
    uvkc::benchmark::vulkan_buffer_util
)
