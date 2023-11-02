# Copyright 2023 Nod Inc.
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
# vmt (RDNA3)
#-------------------------------------------------------------------------------

uvkc_glsl_shader_permutation(
  NAME
    vmt_i8_shader_rdna3
  SRC
    "vmt_i8.glsl"
  PERMUTATION
    "N0=[1|2|4]"
    "K0=[16|32|64|128]"
    "WG_X=64"
    "WG_Y=[1|2|4]"
  GLSLC_ARGS
    "--target-env=vulkan1.1"
)

uvkc_cc_binary(
  NAME
    vmt_rdna3
  SRCS
    "vmt_main.cc"
  DEPS
    ::vmt_i8_shader_rdna3
    benchmark::benchmark
    uvkc::benchmark::core
    uvkc::benchmark::main
  COPTS
  -DUVKC_RDNA3
)
