#version 450 core
#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer InputBuffer { float data[]; } Input;
layout(set=0, binding=1) buffer OutputBuffer { int data; } Output;

layout(constant_id = 0) const uint totalCount = 1; // Total number of scalars

// Each workgroup contains just one subgroup.

void main() {
  uint laneID = gl_LocalInvocationID.x;

  if (laneID != 0) return;

  float wgMax = Input.data[0];
  uint wgResult = 0;

  for (uint i = 1; i < totalCount; ++i) {
    wgResult = Input.data[i] > wgMax ? i : wgResult;
    wgMax = max(wgMax, Input.data[i]);
  }

  Output.data = int(wgResult);
}

