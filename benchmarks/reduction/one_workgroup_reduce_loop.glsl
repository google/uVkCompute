#version 450 core
#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer InputBuffer { vec4 data[]; } Input;
layout(set=0, binding=1) buffer OutputBuffer { float data; } Output;

layout(constant_id = 0) const uint totalCount = 1; // Total number of scalars

// Each workgroup contains just one subgroup.

void main() {
  uint laneID = gl_LocalInvocationID.x;

  if (laneID != 0) return;

  vec4 wgResult = Input.data[0];

  for (uint i = 1; i < totalCount / 4; ++i) {
    wgResult += Input.data[i];
  }

  float floatResult = dot(wgResult, vec4(1.f, 1.f, 1.f, 1.f));

  Output.data = floatResult;
}

