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
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require

#extension GL_KHR_shader_subgroup_basic : enable

layout(binding = 0) buffer InputA { i8vec4 x[]; } inputA;
layout(binding = 1) buffer InputB { i8vec4 x[]; } inputB;
layout(binding = 2) buffer Output { int32_t x[]; } outputO;

layout(local_size_x = WG_X, local_size_y = WG_Y, local_size_z = 1) in;

layout(constant_id = 0) const uint M = 1;
layout(constant_id = 1) const uint N = 1;
layout(constant_id = 2) const uint K = 1;

const uint VECTORIZE_K = 4;
const uint K_VEC = K / VECTORIZE_K;
const uint K0_VEC = K0 / VECTORIZE_K;

const uint VECTORIZE_N = 4;
const uint N_VEC = N / VECTORIZE_N;
const uint N0_VEC = N0 / VECTORIZE_N;

const uint strideA = K_VEC; // Stride of the `inputA` matrix.
const uint strideB = K_VEC; // Stride of the `inputB` matrix.
const uint strideC = N; // Stride of the `outputO` matrix.

// Each workgroup processes an output tile of size [M0 x N0], therefore
// each thread processes a [M0/WG_Y x N0/WG_X] subview.
const uint C_ROWS = M0 / WG_Y;
const uint C_COLS = N0 / WG_X;

/// Returns the index of `X[i, j]`, where `X` is a 2D matrix of stride |stride|.
uint coordToOffset(uint i, uint j, uint stride) { return stride * i + j; }

int32_t sdot(i8vec4 lhs, i8vec4 rhs) {
  i16vec4 mul = i16vec4(lhs) * i16vec4(rhs);
  return int32_t(mul.x) + int32_t(mul.y) + int32_t(mul.z) + int32_t(mul.w);
}

void main() {
  const uvec2 wgID = gl_WorkGroupID.xy;
  const uvec2 localID = gl_LocalInvocationID.xy;
  const uint threadID = gl_SubgroupInvocationID;
  const uint subgroupID = gl_SubgroupID;
  const uint subgroupSz = gl_SubgroupSize;
  const uint numSubgroups = gl_NumSubgroups;

  // The start offsets of the tile processed by this thread in this workgroup.
  const uint x_offset = wgID.x * N0 + C_COLS * localID.x;
  const uint y_offset = wgID.y * M0 + C_ROWS * localID.y;

  i8vec4 A[C_ROWS][K0_VEC];
  i8vec4 B[K0_VEC][C_COLS];
  int32_t C[C_ROWS][C_COLS]; // Local data for the output.

  // Initialize result to zero.
  [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
    [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
      C[i][j] = 0;
    }
  }

  for (uint k = 0; k < K_VEC; k += K0_VEC) {
    // Prefetch LHS and RHS.
    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
      [[unroll]] for (uint kk = 0; kk < K0_VEC; ++kk) {
        uint gk = k + (kk + threadID) % K0_VEC;
        A[i][kk] = inputA.x[coordToOffset(y_offset + i, gk, strideA)];
      }
    }
    [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
      [[unroll]] for (uint kk = 0; kk < K0_VEC; ++kk) {
        uint gk = k + (kk + threadID) % K0_VEC;
        B[kk][j] = inputB.x[coordToOffset(x_offset + j, gk, strideB)];
      }
    }

    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
      [[unroll]] for (uint kk = 0; kk < K0_VEC; ++kk) {
        i8vec4 lhs = A[i][kk];
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
          // Calculate the inner product `C[i, j] := sum(A[i, ..] * B[j, ..])`.
          i8vec4 rhs = B[kk][j];
          C[i][j] += sdot(lhs, rhs);
        }
      }
    }
  }

  // Store the accumulated results in `outputO`.
  [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
    uint y = y_offset + i;
    [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
      uint x = x_offset + j;
      outputO.x[coordToOffset(y, x, strideC)] = C[i][j];
    }
  }
}
