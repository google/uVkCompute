#version 450 core
#pragma use_vulkan_memory_model
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_control_flow_attributes : enable

layout(binding=0) buffer InputA { vec4 x[]; } inputA;
layout(binding=1) buffer InputB { vec4 x[]; } inputB;
layout(binding=2) buffer Output { vec4 x[]; } outputO;

layout(local_size_x = WG_X, local_size_y = WG_Y, local_size_z = 1) in;

layout(constant_id = 0) const uint M = 1;
layout(constant_id = 1) const uint N = 1;
layout(constant_id = 2) const uint K = 1;

const uint strideA = K;
const uint strideB = N;
const uint strideC = N;

const uint C_ROWS = TILE_M / WG_Y;
const uint C_COLS = TILE_N / (4*WG_X);

uint coordToOffset(uint i, uint j, uint stride)
{
    return (stride * i + j);
}

void main()
{
    uvec2 gID = gl_WorkGroupID.xy;
    uvec2 laneId = gl_LocalInvocationID.xy;
    vec4 C[C_ROWS][C_COLS];
    vec4 B[TILE_K][C_COLS];

    // Initialize result to zero
    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            C[i][j] = vec4(0.f, 0.f, 0.f, 0.f);
        }
    }

    for (uint k = 0; k < K; k+=TILE_K) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
          [[unroll]] for (uint i = 0; i < TILE_K; ++i) {
            uint gj = gID.x * (TILE_N / 4) + laneId.x +j*WG_X;
            uint gk = k+i;
            B[i][j] = inputB.x[coordToOffset(gk, gj, strideB/4)];
          }
        }

        [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
          uint gi = gID.y * TILE_M + laneId.y + i*WG_Y;
          uint gk = k/4;
          [[unroll]] for (uint kk = 0; kk < TILE_K/4; kk++) {
            vec4 A = inputA.x[coordToOffset(gi, gk+kk, strideA/4)];
            [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
              C[i][j] += vec4(A.x, A.x, A.x, A.x)*B[0+4*kk][j];
              C[i][j] += vec4(A.y, A.y, A.y, A.y)*B[1+4*kk][j];
              C[i][j] += vec4(A.z, A.z, A.z, A.z)*B[2+4*kk][j];
              C[i][j] += vec4(A.w, A.w, A.w, A.w)*B[3+4*kk][j];
            }
          }
        }
    }

    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint gi = gID.y * TILE_M + laneId.y + i*WG_Y;
            uint gj = gID.x * (TILE_N / 4) + laneId.x +j*WG_X;
            outputO.x[gi * strideC/4 + gj] = C[i][j];
        }
    }
}
