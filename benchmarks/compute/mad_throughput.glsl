#version 450 core
#pragma use_vulkan_memory_model
#extension GL_EXT_scalar_block_layout : enable
#extension GL_AMD_gpu_shader_half_float: enable

layout(binding=0) buffer InputA { TYPE x[]; } inputA;
layout(binding=1) buffer InputB { TYPE x[]; } inputB;
layout(binding=2) buffer Output { TYPE x[]; } outputO;
layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint kLoopSize = 1;

void main()
{
    TYPE a = inputA.x[gl_GlobalInvocationID.x];
    TYPE b = inputB.x[gl_GlobalInvocationID.x];
    TYPE c = TYPE(1.f, 1.f, 1.f, 1.f);
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