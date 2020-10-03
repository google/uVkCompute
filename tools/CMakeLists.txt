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

uvkc_cc_binary(
  NAME
    print_pipeline_layout
  SRCS
    "print_pipeline_layout_main.cc"
    "${UVKC_SOURCE_ROOT}/third_party/SPIRV-Reflect/common/output_stream.cpp"
    "${UVKC_SOURCE_ROOT}/third_party/SPIRV-Reflect/examples/common.cpp"
    "${UVKC_SOURCE_ROOT}/third_party/SPIRV-Reflect/spirv_reflect.c"
  INCLUDES
    "${UVKC_SOURCE_ROOT}/third_party/SPIRV-Reflect"
  DEPS
    absl::flags
    absl::flags_parse
    absl::flags_usage
    absl::strings
    uvkc::base::file
)
