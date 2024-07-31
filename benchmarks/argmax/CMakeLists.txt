# Copyright 2024 Advanced Micro Devices Inc.
# Copyright 2020-2024 Google LLC
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
    one_workgroup_argmax_loop_shader
  SRC
    "one_workgroup_argmax_loop.glsl"
  GLSLC_ARGS
    "--target-env=vulkan1.1"
)

uvkc_glsl_shader_instance(
  NAME
    one_workgroup_argmax_subgroup_shader
  SRC
    "one_workgroup_argmax_subgroup.glsl"
  GLSLC_ARGS
    "--target-env=vulkan1.1"
)

uvkc_cc_binary(
  NAME
    one_workgrop_argmax
  SRCS
    "one_workgroup_argmax_main.cc"
  DEPS
    ::one_workgroup_argmax_loop_shader
    ::one_workgroup_argmax_subgroup_shader
    benchmark::benchmark
    uvkc::benchmark::core
    uvkc::benchmark::main
)
