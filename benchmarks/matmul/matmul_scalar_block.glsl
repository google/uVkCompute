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
#pragma use_vulkan_memory_model
#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_scalar_block_layout : enable

layout(binding=0) buffer InputA { float x[]; } inputA;
layout(binding=1) buffer InputB { float x[]; } inputB;
layout(binding=2) buffer InputC { float x[]; } inputC;
layout(binding=3) buffer Output { float x[]; } outputO;
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// M/N/K values filled out at pipeline creation time
layout(constant_id = 6) const uint K = 1;
layout(constant_id = 7) const uint strideA = 1;
layout(constant_id = 8) const uint strideB = 1;
layout(constant_id = 9) const uint strideC = 1;
layout(constant_id = 10)const uint strideD = 1;



const uint C_ROWS = TILE_M / 8;
const uint C_COLS = TILE_N / 8;

uint coordToOffset(uint i, uint j, uint stride)
{
    return (stride * i + j);
}


void main()
{
   float C[C_ROWS][C_COLS];
   uvec2 tileID = uvec2(gl_WorkGroupID.xy);

    // Initialize result to zero
    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            C[i][j] = 0.f;
        }
    }

    uvec2 inv = gl_LocalInvocationID.xy;
    // On each iteration, load a row of cooperative matrices from matrix A,
    // load a column of cooperative matrices from matrix B, and multiply all
    // pairs of those matrices.
    for (uint chunkK = 0; chunkK < K; chunkK++) {
        float A[C_ROWS];
        [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
            uint gi = TILE_M * tileID.y + i + inv.y * C_ROWS;
            uint gk = chunkK;
            A[i] = inputA.x[coordToOffset(gi, gk, strideA)];
        }
        float B;
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint gj = TILE_N * tileID.x + j + inv.x * C_COLS;
            uint gk = chunkK;
            B = inputB.x[coordToOffset(gk, gj, strideB)];
            [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
                C[i][j] += A[i] * B;
            }
        }
    }

    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint gi = TILE_M * tileID.y + i + inv.y * C_ROWS;
            uint gj = TILE_N * tileID.x + j + inv.x * C_COLS;
            outputO.x[gi * strideD + gj] = C[i][j];
        }
    }
}
