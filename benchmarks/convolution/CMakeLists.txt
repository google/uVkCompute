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

#-------------------------------------------------------------------------------
# Convolution (Adreno)
#-------------------------------------------------------------------------------

uvkc_glsl_shader_permutation(
  NAME
    conv2d_f32_tiled_shader_adreno
  SRC
    "conv2d_tiled.glsl"
  PERMUTATION
    "{WG_X,WG_Y,WG_Z}=[{64,1,1}|{32,2,1}|{16,4,1}|{16,2,2}|{8,4,2}|{4,4,4}]"
    "IVC_OH=[1|2|4]"
    "IVC_OW=[1|2|4]"
    "IVC_OC=1" # Number of vec4
    "VEC4TYPE=vec4"
)

uvkc_glsl_shader_permutation(
  NAME
    conv2d_f16_packed_shader_adreno
  SRC
    "conv2d_packed.glsl"
  PERMUTATION
    "{WG_X,WG_Y,WG_Z}=[{64,1,1}|{32,2,1}|{16,4,1}|{16,2,2}|{8,4,2}|{4,4,4}]"
    "IVC_OH=[1|2|4]"
    "IVC_OW=[1|2|4]"
    "IVC_OC=1" # Number of 2xf16vec4
)

uvkc_cc_binary(
  NAME
    conv2d_adreno
  SRCS
    "conv2d_main.cc"
  DEPS
    ::conv2d_f16_packed_shader_adreno
    ::conv2d_f32_tiled_shader_adreno
    benchmark::benchmark
    uvkc::benchmark::fp16_util
    uvkc::benchmark::main
    uvkc::benchmark::vulkan_buffer_util
  COPTS
    -DUVKC_ADRENO
)

#-------------------------------------------------------------------------------
# Convolution (Mali Valhall)
#-------------------------------------------------------------------------------

uvkc_glsl_shader_permutation(
  NAME
    conv2d_f32_tiled_shader_valhall
  SRC
    "conv2d_tiled.glsl"
  PERMUTATION
    "{WG_X,WG_Y,WG_Z}=[{16,1,1}|{8,2,1}|{4,4,1}|{4,2,2}]"
    "IVC_OH=[1|2|4]"
    "IVC_OW=[1|2|4]"
    "IVC_OC=1" # Number of vec4
    "VEC4TYPE=vec4"
)

uvkc_glsl_shader_permutation(
  NAME
    conv2d_f16_packed_shader_valhall
  SRC
    "conv2d_packed.glsl"
  PERMUTATION
    "{WG_X,WG_Y,WG_Z}=[{16,1,1}|{8,2,1}|{4,4,1}|{4,2,2}]"
    "IVC_OH=[1|2|4]"
    "IVC_OW=[1|2|4]"
    "IVC_OC=1" # Number of 2xf16vec4
)

uvkc_cc_binary(
  NAME
    conv2d_mali_valhall
  SRCS
    "conv2d_main.cc"
  DEPS
    ::conv2d_f16_packed_shader_valhall
    ::conv2d_f32_tiled_shader_valhall
    benchmark::benchmark
    uvkc::benchmark::fp16_util
    uvkc::benchmark::main
    uvkc::benchmark::vulkan_buffer_util
  COPTS
    -DUVKC_MALI_VALHALL
)

#-------------------------------------------------------------------------------
# Depthwise Convolution (Adreno)
#-------------------------------------------------------------------------------

uvkc_glsl_shader_permutation(
  NAME
    depthwise_conv2d_tiled_shader_adreno
  SRC
    "depthwise_conv2d_tiled.glsl"
  PERMUTATION
    "{WG_X,WG_Y,WG_Z}=[{64,1,1}|{32,2,1}|{16,4,1}|{16,2,2}|{8,4,2}|{4,4,4}]"
    "IVC_OH=[1|2|4]"
    "IVC_OW=[1|2|4]"
    "IVC_OC=1" # Number of vec4
)

uvkc_cc_binary(
  NAME
    depthwise_conv2d_adreno
  SRCS
    "depthwise_conv2d_main.cc"
  DEPS
    ::depthwise_conv2d_tiled_shader_adreno
    benchmark::benchmark
    uvkc::benchmark::main
    uvkc::benchmark::vulkan_buffer_util
  COPTS
    -DUVKC_ADRENO
)

#-------------------------------------------------------------------------------
# Depthwise Convolution (Mali Valhall)
#-------------------------------------------------------------------------------

uvkc_glsl_shader_permutation(
  NAME
    depthwise_conv2d_tiled_shader_valhall
  SRC
    "depthwise_conv2d_tiled.glsl"
  PERMUTATION
    "{WG_X,WG_Y,WG_Z}=[{16,1,1}|{8,2,1}|{4,4,1}|{4,2,2}]"
    "IVC_OH=[1|2|4]"
    "IVC_OW=[1|2|4]"
    "IVC_OC=1" # Number of vec4
)

uvkc_cc_binary(
  NAME
    depthwise_conv2d_mali_valhall
  SRCS
    "depthwise_conv2d_main.cc"
  DEPS
    ::depthwise_conv2d_tiled_shader_valhall
    benchmark::benchmark
    uvkc::benchmark::main
    uvkc::benchmark::vulkan_buffer_util
  COPTS
    -DUVKC_MALI_VALHALL
)
