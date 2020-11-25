#version 450 core
#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer InputBuffer { ivec4 data[]; } Input;
layout(set=0, binding=1) buffer OutputBuffer { int data; }  Output;

// Macro to be defined at compile time
// BATCH_SIZE: how many scalar elements to process for each workgroup

// Each workgroup contains just one subgroup.

void main() {
  uint wgID = gl_WorkGroupID.x;
  uint laneID = gl_LocalInvocationID.x;

  if (laneID != 0) return;

  uint wgBaseOffset = wgID * BATCH_SIZE / 4;
  ivec4 wgResult = Input.data[wgBaseOffset];

  [[unroll]] for (uint i = 1; i < BATCH_SIZE / 4; ++i) {
    wgResult += Input.data[wgBaseOffset + i];
  }

  int intResult = wgResult.x + wgResult.y + wgResult.z + wgResult.w;
  atomicAdd(Output.data, intResult);
}
