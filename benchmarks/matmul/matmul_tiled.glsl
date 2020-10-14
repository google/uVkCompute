#version 450 core
#pragma use_vulkan_memory_model
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_control_flow_attributes : enable

layout(binding=0) buffer InputA { vec4 x[]; } inputA;
layout(binding=1) buffer InputB { vec4 x[]; } inputB;
layout(binding=2) buffer Output { vec4 x[]; } outputO;
layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint M = 1;
layout(constant_id = 1) const uint N = 1;
layout(constant_id = 2) const uint K = 1;

const uint strideA = K;
const uint strideB = N;
const uint strideC = N;

const uint C_ROWS = TILE_M / 1;
const uint C_COLS = TILE_N / 64;

uint coordToOffset(uint i, uint j, uint stride)
{
    return (stride * i + j);
}

void main()
{
    uint gID = gl_WorkGroupID.x;
    uint laneId = gl_LocalInvocationID.x;
    uvec2 tileID = uvec2(gl_GlobalInvocationID.xy);
    vec4 C[C_ROWS][C_COLS];
    vec4 A[C_ROWS];

    // Initialize result to zero
    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            C[i][j] = vec4(0.f, 0.f, 0.f, 0.f);
        }
    }

    for (uint k = 0; k < K; k+=4) {
        [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
            uint gi = tileID.y*C_ROWS+i;
            uint gk = k/4;
            A[i] = inputA.x[coordToOffset(gi, gk, strideA/4)];
        }

        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
          uint gj = gID * (TILE_N / 4) + laneId +j*16;
          uint gk = k;
          vec4 B = inputB.x[coordToOffset(gk, gj, strideB/4)];
          [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
            C[i][j] += vec4(A[i].x, A[i].x, A[i].x, A[i].x)*B;
          }
        }

        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
          uint gj = gID * (TILE_N / 4) + laneId +j*16;
          uint gk = k+1;
          vec4 B = inputB.x[coordToOffset(gk, gj, strideB/4)];
          [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
            C[i][j] += vec4(A[i].y, A[i].y, A[i].y, A[i].y)*B;
          }
        }

        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
          uint gj = gID * (TILE_N / 4) + laneId +j*16;
          uint gk = k+2;
          vec4 B = inputB.x[coordToOffset(gk, gj, strideB/4)];
          [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
            C[i][j] += vec4(A[i].z, A[i].z, A[i].z, A[i].z)*B;
          }
        }

        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
          uint gj = gID * (TILE_N / 4) + laneId +j*16;
          uint gk = k+3;
          vec4 B = inputB.x[coordToOffset(gk, gj, strideB/4)];
          [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
            C[i][j] += vec4(A[i].w, A[i].w, A[i].w, A[i].w)*B;
          }
        }
    }

    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint gi = tileID.y*C_ROWS+i;
            uint gj = gID * (TILE_N / 4) + laneId +j*16;
            outputO.x[gi * strideC/4 + gj] = C[i][j];
        }
    }
}