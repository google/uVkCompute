# Copyright 2023 Google LLC
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
# mmt (Adreno)
#-------------------------------------------------------------------------------

uvkc_glsl_shader_permutation(
  NAME
    mmt_i8_shader_adreno
  SRC
    "mmt_i8.glsl"
  PERMUTATION
    "M0=[4|8|16|32]"
    "N0=[128|256]"
    "K0=[4|8|16]"
    "WG_X=32"
    "WG_Y=2"
  GLSLC_ARGS
    "--target-env=vulkan1.1"
)

uvkc_cc_binary(
  NAME
    mmt_adreno
  SRCS
    "mmt_main.cc"
  DEPS
    ::mmt_i8_shader_adreno
    benchmark::benchmark
    uvkc::benchmark::core
    uvkc::benchmark::main
  COPTS
    -DUVKC_ADRENO
)

#-------------------------------------------------------------------------------
# mmt (Mali Valhall)
#-------------------------------------------------------------------------------

uvkc_glsl_shader_permutation(
  NAME
    mmt_i8_shader_valhall
  SRC
    "mmt_i8.glsl"
  PERMUTATION
    "M0=[4|8|16|32]"
    "N0=[16|32|64]"
    "K0=[4|8|16]"
    "WG_X=16"
    "WG_Y=1"
  GLSLC_ARGS
    "--target-env=vulkan1.1"
)

uvkc_cc_binary(
  NAME
    mmt_mali_valhall
  SRCS
    "mmt_main.cc"
  DEPS
    ::mmt_i8_shader_valhall
    benchmark::benchmark
    uvkc::benchmark::core
    uvkc::benchmark::main
  COPTS
    -DUVKC_MALI_VALHALL
)
