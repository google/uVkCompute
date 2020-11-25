#version 450 core
#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer InputBuffer { vec4 data[]; } Input;
layout(set=0, binding=1) buffer OutputBuffer { uint data; }  Output;

// Macro to be defined at compile time
// BATCH_SIZE: how many scalar elements to process for each workgroup

// Each workgroup contains just one subgroup.

void main() {
  uint wgID = gl_WorkGroupID.x;
  uint laneID = gl_LocalInvocationID.x;

  uint wgBaseOffset = wgID * BATCH_SIZE / 4;
  vec4 laneResult = Input.data[wgBaseOffset + laneID];

  [[unroll]] for (uint i = 1; i < BATCH_SIZE / (16 * 4); ++i) {
    laneResult += Input.data[wgBaseOffset + 16 * i + laneID];
  }

  // Final reduction with one subgroup
  vec4 wgResult = subgroupAdd(laneResult);
  float floatResult = dot(wgResult, vec4(1.0f, 1.0f, 1.0f, 1.0f));

  if (subgroupElect()) {
    uint srcValue, originalValue;
    do {
      srcValue = Output.data;
      float srcFloatValue = uintBitsToFloat(srcValue);
      float dstFloatValue = srcFloatValue + floatResult;
      uint dstValue = floatBitsToUint(dstFloatValue);
      originalValue = atomicCompSwap(Output.data, srcValue, dstValue);
    } while (originalValue != srcValue);
  }
}
