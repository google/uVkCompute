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
#pragma use_vulkan_memory_model

#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require

layout(binding = 0) buffer InputA { i8vec4 x[]; } inputA;
layout(binding = 1) buffer InputB { int8_t x[]; } inputB;
layout(binding = 2) buffer Output { int32_t x[]; } outputO;

layout(local_size_x = WG_X, local_size_y = WG_Y, local_size_z = 1) in;

layout(constant_id = 0) const uint M = 1;
layout(constant_id = 1) const uint N = 1;
layout(constant_id = 2) const uint K = 1;

const uint strideA = K;
const uint strideB = N;
const uint strideC = N;

const uint C_ROWS = TILE_M / WG_Y;
const uint C_COLS = TILE_N / WG_X;

uint coordToOffset(uint i, uint j, uint stride) { return (stride * i + j); }

void main() {
  uvec2 gID = gl_WorkGroupID.xy;
  uvec2 laneId = gl_LocalInvocationID.xy;
  int32_t C[C_ROWS][C_COLS]; // Local data for the output.
  i8vec4 B[TILE_K / 4][C_COLS]; // Prefetched data for RHS.

  // Initialize result to zero.
  [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
    [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
      C[i][j] = 0;
    }
  }

  // TODO(kuhar@): Further optimize this.
  for (uint k_pos = 0; k_pos < K; k_pos += TILE_K) {
    // Prefetch RHS.
    [[unroll]] for (uint k = 0; k < TILE_K; k += 4) {
      uint gk = k + k_pos;
      [[unroll]] for (uint j = 0; j < C_COLS; j += 4) {
        uint x = gID.x * TILE_N + j * WG_X + laneId.x;
        [[unroll]] for (uint jj = 0; jj < 4; ++jj) {
          B[k / 4][j + jj].x = inputB.x[coordToOffset(gk + 0, x + jj * WG_X, strideB)];
          B[k / 4][j + jj].y = inputB.x[coordToOffset(gk + 1, x + jj * WG_X, strideB)];
          B[k / 4][j + jj].z = inputB.x[coordToOffset(gk + 2, x + jj * WG_X, strideB)];
          B[k / 4][j + jj].w = inputB.x[coordToOffset(gk + 3, x + jj * WG_X, strideB)];
        }
      }
    }

    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
      [[unroll]] for (uint k = 0; k < TILE_K; k += 4) {
        uint y = gID.y * TILE_M + i * WG_Y + laneId.y;
        uint gk = k + k_pos;
        i16vec4 lhs = inputA.x[coordToOffset(y, gk, strideA) / 4];
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
          uint x = gID.x * TILE_N + j * WG_X + laneId.x;

          int32_t acc = 0;
          i16vec4 rhs = B[k / 4][j];
          i16vec4 mul = lhs * rhs;
          acc += int32_t(mul.x) + int32_t(mul.y) + int32_t(mul.z) + int32_t(mul.w);
          C[i][j] += acc;
        }
      }
    }
  }

  [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
    uint gi = gID.y * TILE_M + laneId.y + i * WG_Y;
    [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
      uint gj = gID.x * TILE_N + laneId.x + j * WG_X;
      outputO.x[coordToOffset(gi, gj, strideC)] = C[i][j];
    }
  }
}
