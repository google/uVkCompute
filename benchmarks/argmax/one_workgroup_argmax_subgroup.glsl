# version 450 core
#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer InputBuffer { float data[]; } Input;
layout(set=0, binding=1) buffer OutputBuffer { uint data; } Output;

layout(constant_id = 0) const uint totalCount = 1; // Total number of scalars

// Each workgroup contains just one subgroup.

void main() {
  uint laneID = gl_LocalInvocationID.x;
  uint laneCount = gl_WorkGroupSize.x;

  uint laneResult = 0;
  float laneMax = Input.data[laneID];

  uint numBatches = totalCount / laneCount;

  for (uint i = 1; i < numBatches; ++i) {
    uint idx = laneCount * i + laneID;
    float elem = Input.data[idx];
    if (elem > laneMax) {
      laneResult = idx;
      laneMax = elem;
    }
  }

  // Find the max of workgroup (containing only one subgroup).
  float wgMax = subgroupMax(laneMax);

  // Find the smallest thread ID with the max element.
  bool bit = laneMax == wgMax;
  uvec4 mask = subgroupBallot(bit);
  uint smallestID = subgroupBallotFindLSB(mask);

  // The thread is responsible for outputing result.
  if (laneID == smallestID)
    Output.data = laneResult;
}
