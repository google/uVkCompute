#version 450 core
#pragma use_vulkan_memory_model
#extension GL_EXT_scalar_block_layout : enable

layout(binding=0) buffer InputA { vec4 x[]; } inputA;
layout(binding=1) buffer InputB { vec4 x[]; } inputB;
layout(binding=2) buffer Output { vec4 x[]; } outputO;
layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint kLoopSize = 1;

void main()
{
    vec4 a = inputA.x[gl_GlobalInvocationID.x];
    vec4 b = inputB.x[gl_GlobalInvocationID.x];
    vec4 c = vec4(1.f, 1.f, 1.f, 1.f);
    for(int i = 0; i < kLoopSize; i++) {
      c = a * c + b;
      c = a * c + b;
      c = a * c + b;
      c = a * c + b;
      c = a * c + b;
      c = a * c + b;
      c = a * c + b;
      c = a * c + b;
      c = a * c + b;
      c = a * c + b;
    }
    outputO.x[gl_GlobalInvocationID.x] = c;
}