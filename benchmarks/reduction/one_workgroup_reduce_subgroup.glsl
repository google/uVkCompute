#version 450 core
#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer InputBuffer { vec4 data[]; } Input;
layout(set=0, binding=1) buffer OutputBuffer { float data; } Output;

layout(constant_id = 0) const uint totalCount = 1; // Total number of scalars

// Each workgroup contains just one subgroup.

void main() {
  uint laneID = gl_LocalInvocationID.x;
  uint laneCount = gl_WorkGroupSize.x;

  vec4 laneResult = Input.data[laneID];

  uint numBatches = totalCount / (laneCount * 4);
  for (uint i = 1; i < numBatches; ++i) {
    laneResult += Input.data[laneCount * i + laneID];
  }

  // Final reduction with one subgroup
  vec4 wgResult = subgroupAdd(laneResult);
  float floatResult = dot(wgResult, vec4(1.f, 1.f, 1.f, 1.f));

  if (subgroupElect()) Output.data = floatResult;
}

