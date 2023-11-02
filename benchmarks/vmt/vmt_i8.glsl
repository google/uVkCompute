// Copyright 2023 Nod Inc.
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

// Multiplies vector `inputA` of length `K` by matrix `inputB`
// of size `K x N`.
// We use `K0` and `N0` to denote the tile sizes for N and K,
// respectively.
//
// We assign `N0` rows for each subgroup to process.
// Subgroups load a batch of values from the vector and the row,
// calculate the inner product of the two, and accumulate the results at the
// subgroup-level.
// Each workgroup produces `N0` * `WG_Y` results. We assume that WG_X is the
// same as the subgroup size to simplify the implementation. This is a shortcut
// that should be fixed.

layout(binding = 0) buffer InputA { i32vec4 x[]; } inputA;  // Input vector.
layout(binding = 1) buffer InputB { i32vec4 x[]; } inputB;  // Input matrix.
layout(binding = 2) buffer Output { int32_t x[]; } outputO;  // Output vector.

// These are constants defined at compile time.
layout(local_size_x = WG_X, local_size_y = WG_Y, local_size_z = 1) in;

layout(constant_id = 0) const uint N = 1;
layout(constant_id = 1) const uint K = 1;

// We process 16-element i8 vectors along the K dimension, and treat i32vec4
// (packed format) as the effective element type.
const uint VECTORIZE_K = 16;
const uint K_VEC = K / VECTORIZE_K;
const uint K0_VEC = K0 / VECTORIZE_K;

const uint strideB = K_VEC; // Stride of the `inputB` matrix.

// Each subgroup processes a total of N0 rows, therefore
// each workgroup processes N0 * WG_Y rows.
const uint WG_ROWS = N0 * WG_Y;

// Offset between elements accessed by a thread in a workgroup.
const uint WG_K_STRIDE = WG_X * K0_VEC;

/// Returns the index of `X[i, j]`, where `X` is a 2D matrix of stride `stride`.
uint coordToOffset(uint i, uint j, uint stride) { return stride * i + j; }

int32_t sdot(int32_t lhsPacked, int32_t rhsPacked) {
  i8vec4 lhs = unpack8(lhsPacked);
  i8vec4 rhs = unpack8(rhsPacked);
  i16vec4 mul = i16vec4(lhs) * i16vec4(rhs);
  return int32_t(mul.x) + int32_t(mul.y) + int32_t(mul.z) + int32_t(mul.w);
}

int32_t sdot4(i32vec4 lhs, i32vec4 rhs) {
  return sdot(lhs.x, rhs.x) + sdot(lhs.y, rhs.y) + sdot(lhs.z, rhs.z) + sdot(lhs.w, rhs.w);
}

void main() {
  const uvec2 wgID = gl_WorkGroupID.xy;
  const uvec2 localID = gl_LocalInvocationID.xy;
  const uint threadID = gl_SubgroupInvocationID;

  // The start offsets of the row tile processed by this thread in this workgroup.
  const uint startRow = wgID.x * WG_ROWS;

  for (uint y = 0; y < N0; ++y) {
    uint r = startRow + y * WG_Y + localID.y;
    int32_t laneResult = 0;
    i32vec4 tileA[K0_VEC];
    i32vec4 tileB[K0_VEC];

    for (uint k = 0; k < K_VEC; k += WG_K_STRIDE) {
      // Prefetch LHS and RHS to reduce the latency.
      [[unroll]] for (uint kk = 0; kk < K0_VEC; ++kk) {
        uint gk = k + kk * WG_X + threadID;
        tileB[kk] = inputB.x[coordToOffset(r, gk, strideB)];
        tileA[kk] = inputA.x[gk];
      }

      [[unroll]] for (uint kk = 0; kk < K0_VEC; ++kk) {
        i32vec4 lhs = tileA[kk];
        i32vec4 rhs = tileB[kk];
        laneResult += sdot4(lhs, rhs);
      }
    }

    // Final reduction with one subgroup.
    int32_t sgResult = subgroupAdd(laneResult);
    if (subgroupElect()) {
      outputO.x[r] = sgResult;
    }
  }

  // Assert that the subgroup and workgroup sizes match.
  // This simplifies the code but doesn't have to be true on all targets.
  if (threadID != localID.x)
    outputO.x[0] = -1;
}
