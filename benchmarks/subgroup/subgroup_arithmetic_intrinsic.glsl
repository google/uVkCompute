// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#version 450 core

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout (local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const int kArraySize = 64;

layout(set = 0, binding = 0) buffer InputBuffer {
    float input_values[kArraySize];
};

// Use an output buffer of the same size to make sure we use each element
// in the input buffer.
layout(set = 0, binding = 1) buffer OutputBuffer {
    float output_values[kArraySize];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    uint count = gl_SubgroupSize;
    float value = 0.f;

#ifdef ARITHMETIC_ADD
    float total = subgroupAdd(input_values[index]);
#endif
#ifdef ARITHMETIC_MUL
    float total = subgroupMul(input_values[index]);
#endif
    if (subgroupElect()) {
      value = total;
    } else {
      value = input_values[index];
    }

    output_values[index] = value;
}

