// Copyright 2020-2023 Google LLC
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

#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_basic : enable

layout (local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const int kElementsPerThread = 1;

layout(set = 0, binding = 0) buffer InputBuffer {
    vec4 input_values[];
};

layout(set = 0, binding = 1) buffer OutputBuffer {
    vec4 output_values[];
};

const uint WG_X = 32;

void main() {
    // Must guarantee index is in range during dispatch.
    uint index = gl_WorkGroupID.x * WG_X * kElementsPerThread +
                 gl_SubgroupID * gl_SubgroupSize * kElementsPerThread +
                 gl_SubgroupInvocationID;
    uint stride = gl_SubgroupSize;
    // We want to space out memory accesses by `stride`, so that adjacent threads
    // access adjacent memory.
    [[unroll]] for (uint i = 0; i < kElementsPerThread; ++i, index += stride) {
        output_values[index] = input_values[index];
    }
}

