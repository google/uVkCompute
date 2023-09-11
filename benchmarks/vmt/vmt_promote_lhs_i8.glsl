// Copyright 2023 Google LLC
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
#extension GL_KHR_shader_subgroup_arithmetic : enable
#pragma use_vulkan_memory_model

#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require

#extension GL_KHR_shader_subgroup_basic : enable

layout(binding = 0) buffer InputA { i8vec4 x[]; } inputA;
layout(binding = 1) buffer InputB { i8vec4 x[]; } inputB;
layout(binding = 2) buffer Output { int32_t x[]; } outputO;

layout(local_size_x = WG_X, local_size_y = WG_Y, local_size_z = 1) in;

layout(constant_id = 0) const uint N = 1;
layout(constant_id = 1) const uint K = 1;

const uint VECTORIZE_K = 4;
const uint K_VEC = K / VECTORIZE_K;
const uint K0_VEC = K0 / VECTORIZE_K;

const uint strideB = K_VEC; // Stride of the `inputB` matrix.

// Each workgroup processes a total of N0 rows, therefore
// each subgroup processes N0/WG_Y rows.
const uint C_ROWS = N0 / WG_Y;

/// Returns the index of `X[i, j]`, where `X` is a 2D matrix of stride |stride|.
uint coordToOffset(uint i, uint j, uint stride) { return stride * i + j; }

int32_t sdot(i8vec4 lhs, i8vec4 rhs) {
  i16vec4 mul = i16vec4(lhs) * i16vec4(rhs);
  return int32_t(mul.x) + int32_t(mul.y) + int32_t(mul.z) + int32_t(mul.w);
}

shared i8vec4 LHS[K_VEC]; // Shared data for the LHS.

void main() {
  const uvec2 wgID = gl_WorkGroupID.xy;
  const uvec2 localID = gl_LocalInvocationID.xy;
  const uint threadID = gl_SubgroupInvocationID;

  const uint laneCount = gl_WorkGroupSize.x;
  const uint partialVec = laneCount * K0_VEC;

  const uint yCount = gl_WorkGroupSize.y;
  const uint flatPartialVec = yCount * laneCount;

  // Copy the LHS vector to LDS for reuse with the rows of the RHS matrix.
  [[unroll]] for (uint i = 0; i < K_VEC; i += flatPartialVec) {
    uint k_offset = i + localID.y * laneCount + localID.x;
    // Assume the number of threads used to do the copy divides the number of elements.
    // if (k_offset < K_VEC) {
      LHS[k_offset] = inputA.x[k_offset];
    // }
  }

  barrier();

  // The start offsets of the tile processed by this thread in this workgroup.
  const uint startRow = wgID.x * N0 + localID.y;

  for (uint r = startRow; r < startRow + C_ROWS * WG_Y; r += WG_Y) {
    int32_t laneResult = 0;

    for (uint k = 0; k < K_VEC; k += partialVec) {
      [[unroll]] for (uint kk = 0; kk < K0_VEC; ++kk) {
        uint gk = k + kk + threadID * K0_VEC;
        i8vec4 lhs = LHS[gk];
        i8vec4 rhs = inputB.x[coordToOffset(r, gk, strideB)];
        laneResult += sdot(lhs, rhs);
      }
    }

    // Final reduction with one subgroup
    int32_t wgResult = subgroupAdd(laneResult);
    if (subgroupElect()) {
      outputO.x[r] = wgResult;
    }
  }
}
