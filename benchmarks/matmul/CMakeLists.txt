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

#-------------------------------------------------------------------------------
# Matmul (Adreno)
#-------------------------------------------------------------------------------

uvkc_glsl_shader_permutation(
  NAME
    matmul_tiled_shader_f16_adreno
  SRC
    "matmul_tiled_fp16.glsl"
  PERMUTATION
    "TILE_M=[2|4|8|16|32]"
    "TILE_N=[128|256]"
    "TILE_K=[4|8]"
    "TEXTURE=[1|0]"
    "WG_X=32"
    "WG_Y=2"
)

uvkc_glsl_shader_permutation(
  NAME
    matmul_tiled_shader_f32_adreno
  SRC
    "matmul_tiled_fp32.glsl"
  PERMUTATION
    "TILE_M=[2|4|8|16|32]"
    "TILE_N=[128|256]"
    "TILE_K=[4|8]"
    "WG_X=32"
    "WG_Y=2"
)

uvkc_glsl_shader_permutation(
  NAME
    matmul_tiled_shader_i32_adreno
  SRC
    "matmul_tiled_i32.glsl"
  PERMUTATION
    "TILE_M=[2|4|8|16|32]"
    "TILE_N=[128|256]"
    "TILE_K=[4|8]"
    "WG_X=32"
    "WG_Y=2"
)

uvkc_glsl_shader_permutation(
  NAME
    matmul_tiled_shader_i8_adreno
  SRC
    "matmul_tiled_i8.glsl"
  PERMUTATION
    "TILE_M=[2|4|8|16|32]"
    "TILE_N=[128|256]"
    "TILE_K=[4|8]"
    "WG_X=32"
    "WG_Y=2"
)

uvkc_glsl_shader_permutation(
  NAME
    matmul_tiled_shader_i8_innerproduct_adreno
  SRC
    "matmul_tiled_i8_innerproduct.glsl"
  PERMUTATION
    "TILE_M=[2|4|8|16|32]"
    "TILE_N=[128|256]"
    "TILE_K=[4|8]"
    "WG_X=32"
    "WG_Y=2"
)

uvkc_cc_binary(
  NAME
    matmul_tiled_adreno
  SRCS
    "matmul_tiled_main.cc"
  DEPS
    ::matmul_tiled_shader_f16_adreno
    ::matmul_tiled_shader_f32_adreno
    ::matmul_tiled_shader_i32_adreno
    ::matmul_tiled_shader_i8_adreno
    ::matmul_tiled_shader_i8_innerproduct_adreno
    benchmark::benchmark
    uvkc::benchmark::core
    uvkc::benchmark::main
  COPTS
    -DUVKC_ADRENO
)

#-------------------------------------------------------------------------------
# Matmul (Mali Valhall)
#-------------------------------------------------------------------------------

uvkc_glsl_shader_permutation(
  NAME
    matmul_tiled_shader_f16_valhall
  SRC
    "matmul_tiled_fp16.glsl"
  PERMUTATION
    "TILE_M=[2|4|8|16|32]"
    "TILE_N=[64|128]"
    "TILE_K=[4|8]"
    "TEXTURE=[1|0]"
    "WG_X=8"
    "WG_Y=2"
)

uvkc_glsl_shader_permutation(
  NAME
    matmul_tiled_shader_f32_valhall
  SRC
    "matmul_tiled_fp32.glsl"
  PERMUTATION
    "TILE_M=[2|4|8|16|32]"
    "TILE_N=[64|128]"
    "TILE_K=[4|8]"
    "WG_X=16"
    "WG_Y=1"
)

uvkc_glsl_shader_permutation(
  NAME
    matmul_tiled_shader_i32_valhall
  SRC
    "matmul_tiled_i32.glsl"
  PERMUTATION
    "TILE_M=[2|4|8|16|32]"
    "TILE_N=[64|128]"
    "TILE_K=[4|8]"
    "WG_X=16"
    "WG_Y=1"
)

uvkc_glsl_shader_permutation(
  NAME
    matmul_tiled_shader_i8_valhall
  SRC
    "matmul_tiled_i8.glsl"
  PERMUTATION
    "TILE_M=[2|4|8|16|32]"
    "TILE_N=[64|128]"
    "TILE_K=[4|8]"
    "WG_X=16"
    "WG_Y=1"
)

uvkc_glsl_shader_permutation(
  NAME
    matmul_tiled_shader_i8_innerproduct_valhall
  SRC
    "matmul_tiled_i8_innerproduct.glsl"
  PERMUTATION
    "TILE_M=[2|4|8|16|32]"
    "TILE_N=[64|128]"
    "TILE_K=[4|8]"
    "WG_X=16"
    "WG_Y=1"
)

uvkc_cc_binary(
  NAME
    matmul_tiled_mali_valhall
  SRCS
    "matmul_tiled_main.cc"
  DEPS
    ::matmul_tiled_shader_f16_valhall
    ::matmul_tiled_shader_f32_valhall
    ::matmul_tiled_shader_i32_valhall
    ::matmul_tiled_shader_i8_valhall
    ::matmul_tiled_shader_i8_innerproduct_valhall
    benchmark::benchmark
    uvkc::benchmark::core
    uvkc::benchmark::main
  COPTS
    -DUVKC_MALI_VALHALL
)
