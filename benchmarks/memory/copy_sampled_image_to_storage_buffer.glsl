#version 450

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D MySampledImage;

layout(set = 0, binding = 1) buffer OutputBuffer {
  float elements[];
} MyBuffer;

void main() {
  int dimx = int(gl_NumWorkGroups.x);
  int gx = int(gl_GlobalInvocationID.x);
  int gy = int(gl_GlobalInvocationID.y);
  MyBuffer.elements[gy * dimx * 16 + gx] = texelFetch(MySampledImage, ivec2(gx, gy), 0).x;
}
