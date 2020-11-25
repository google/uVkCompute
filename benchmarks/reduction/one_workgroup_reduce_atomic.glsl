#version 450 core
#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x_id = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer InputBuffer { vec4 data[]; } Input;
layout(set=0, binding=1) buffer OutputBuffer { float data; } Output;

layout(constant_id = 0) const uint totalCount = 1; // Total number of scalars

shared uint finalResult;

// Each workgroup contains (local_size_x_id / gl_SubgroupSize) subgroups.

void main() {
  uint threadID = gl_LocalInvocationID.x;
  uint threadCount = gl_WorkGroupSize.x;
  uint threadBatch = totalCount / (threadCount * 4);

  if (threadID == 0) finalResult = 0;

  barrier();

  vec4 threadResult = Input.data[threadID];
  for (uint i = 1; i < threadBatch; ++i) {
    threadResult += Input.data[threadCount * i + threadID];
  }

  vec4 subgroupResult = subgroupAdd(threadResult);
  float floatResult = dot(subgroupResult, vec4(1.f, 1.f, 1.f, 1.f));

  if (subgroupElect()) {
    uint srcValue, originalValue;
    do {
      srcValue = finalResult;
      float srcFloatValue = uintBitsToFloat(srcValue);
      float dstFloatValue = srcFloatValue + floatResult;
      uint dstValue = floatBitsToUint(dstFloatValue);
      originalValue = atomicCompSwap(finalResult, srcValue, dstValue);
    } while (originalValue != srcValue);
  }

  barrier();

  if (threadID == 0) Output.data = uintBitsToFloat(finalResult);
}
