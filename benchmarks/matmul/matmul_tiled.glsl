#version 450 core
#pragma use_vulkan_memory_model
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_control_flow_attributes : enable
#extension GL_AMD_gpu_shader_half_float: enable

layout(binding=0) buffer InputA { TYPE x[]; } inputA;
layout(binding=1) buffer InputB { TYPE x[]; } inputB;
layout(binding=2) buffer Output { TYPE x[]; } outputO;
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
    TYPE C[C_ROWS][C_COLS];
    TYPE B[TILE_K][C_COLS];

    // Initialize result to zero
    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            C[i][j] = TYPE(0.f, 0.f, 0.f, 0.f);
        }
    }

    for (uint k = 0; k < K; k+=TILE_K) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
          [[unroll]] for (uint i = 0; i < TILE_K; ++i) {
            uint gj = gID * (TILE_N / 4) + laneId +j*16;
            uint gk = k+i;
            B[i][j] = inputB.x[coordToOffset(gk, gj, strideB/4)];
          }
        }

        [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
          uint gi = tileID.y*C_ROWS+i;
          uint gk = k/4;
          [[unroll]] for (uint kk = 0; kk < TILE_K/4; kk++) {
            vec4 A = inputA.x[coordToOffset(gi, gk+kk, strideA/4)];
            [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
              C[i][j] += TYPE(A.x, A.x, A.x, A.x)*B[0+4*kk][j];
              C[i][j] += TYPE(A.y, A.y, A.y, A.y)*B[1+4*kk][j];
              C[i][j] += TYPE(A.z, A.z, A.z, A.z)*B[2+4*kk][j];
              C[i][j] += TYPE(A.w, A.w, A.w, A.w)*B[3+4*kk][j];
            }
          }
        }
    }

    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint gi = tileID.y*TILE_M+i;
            uint gj = gID * (TILE_N / 4) + laneId +j*16;
            outputO.x[gi * strideC/4 + gj] = C[i][j];
        }
    }
}