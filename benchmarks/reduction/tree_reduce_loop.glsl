#version 450 core
#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer DataBuffer { TYPE data[]; } IOBuffer;

layout(constant_id = 0) const uint stride = 1; // Stride between elements

// Macro to be defined at compile time
// BATCH_SIZE: how many elements to process for each workgroup

// Each workgroup contains just one subgroup.

void main() {
  uint wgID = gl_WorkGroupID.x;
  uint laneID = gl_LocalInvocationID.x;

  if (laneID != 0) return;

  TYPE wgResult = IOBuffer.data[wgID];

  [[unroll]] for (uint i = 1; i < BATCH_SIZE; ++i) {
    wgResult += IOBuffer.data[wgID + stride * i];
  }

  IOBuffer.data[wgID] = wgResult;
}
