#version 450

layout (local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const int kArraySize = 64 / 4;

layout(set = 0, binding = 0) buffer InputBuffer {
    vec4 input_values[kArraySize];
};

layout(set = 0, binding = 1) buffer OutputBuffer {
    vec4 output_values[kArraySize];
};

void main()
{
    // Must guarantee index is in range during dispatch.
    uint index = gl_GlobalInvocationID.x;
    output_values[index] = input_values[index];
}

