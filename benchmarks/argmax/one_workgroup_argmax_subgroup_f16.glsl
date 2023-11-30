#version 450 core
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_EXT_shader_subgroup_extended_types_float16 : enable
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer InputBuffer { float16_t data[]; } Input;
layout(set=0, binding=1) buffer OutputBuffer { uvec2 data; } Output;

layout(push_constant) uniform PushConstants { uint totalCount; }; // Total number of scalars

// Each workgroup contains just one subgroup.

void main() {
  uint laneID = gl_LocalInvocationID.x;
  uint laneCount = gl_WorkGroupSize.x;

  float16_t laneMax = Input.data[laneID];
  uint laneResult = 0;

  uint numBatches = totalCount / (laneCount);
  for (int i = 1; i < numBatches; ++i) {
    uint idx = laneCount * i + laneID;
    float16_t new_in = Input.data[idx];
    laneResult = new_in > laneMax ? idx : laneResult;
    laneMax = max(laneMax, new_in);
  }

  // Final reduction with one subgroup
  float16_t wgMax = subgroupMax(laneMax);

  bool eq = wgMax == laneMax;
  uvec4 ballot = subgroupBallot(eq);
  uint lsb = subgroupBallotFindLSB(ballot);

  uint upper32bits = 0;
  if (laneID == lsb) Output.data = uvec2(laneResult, upper32bits);
}

