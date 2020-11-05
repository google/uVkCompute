#version 450 core
#extension GL_EXT_control_flow_attributes : enable

// A shader that computes 2-D convolution.
// - Output: NHoWoCo format
// - Input: NHiWiCi format
// - Filter: HfWfCiCo format
// - N is 1.
// - Wo must be a multiple of 2.
// - Ci must be a multiple of 4.
// - Co must be a multiple of 32.
// - No padding.
// - No dilation.

layout(local_size_x = 8, local_size_y = 2, local_size_z = 1) in;

layout(set=0, binding=0) buffer InputBuffer   { vec4 data[]; } Input;
layout(set=0, binding=1) buffer FilterBufffer { vec4 data[]; } Filter;
layout(set=0, binding=2) buffer OutputBuffer  { vec4 data[]; } Output;

layout(constant_id = 0) const uint OH = 1; // Output height
layout(constant_id = 1) const uint OW = 1; // Output width
layout(constant_id = 2) const uint OC = 1; // Output channel
layout(constant_id = 3) const uint IH = 1; // Input height
layout(constant_id = 4) const uint IW = 1; // Input width
layout(constant_id = 5) const uint IC = 1; // Input channel
layout(constant_id = 6) const uint FH = 1; // Filter height
layout(constant_id = 7) const uint FW = 1; // Filter width
layout(constant_id = 8) const uint SH = 1; // Height stride
layout(constant_id = 9) const uint SW = 1; // Width stride

// Macros to be defined at compile time
// WG_TILE_OW: tile size along the output width dimension for a workgroup
// WG_TILE_OC: tile size along the output channel dimension for a workgroup

// Workgroup count:
// - x: OC / WG_TILE_OC
// - y: OW / WG_TILE_OW
// - z: OH
// So each workgroup computes (WG_TILE_OW * WG_TILE_OC) output elements.

// Each workgroup just contains one subgroup.
// Each invocation in the workgroup processes:
// - x: WG_TILE_OC / (gl_WorkGroupSize.x * 4)
// - y: WG_TILE_OW / gl_WorkGroupSize.y
// Adjecent invocations process elements along x dimension cyclically;
// each handle 4 consecutive elements.

const uint IVC_OC = WG_TILE_OC / (/*gl_WorkGroupSize.x=*/8 * 4);
const uint IVC_OW = WG_TILE_OW / /*gl_WorkgroupSize.y=*/2;

uint inputCoordToOffset(uint h, uint w, uint c) {
  return (h  * IW * IC + w * IC + c) / 4;
}

uint filterCoordToOffset(uint h, uint w, uint ic, uint oc) {
  return (h * FW * IC * OC + w  * IC * OC + ic * OC + oc) / 4;
}

uint outputCoordToOffset(uint h, uint w, uint c) {
  return (h  * OW * OC + w * OC + c) / 4;
}

void main() {
  // Each invocation calculates (IVC_OW * IVC_OC * 4) output elements.
  vec4 O[IVC_OW][IVC_OC];

  // Use registers to keep the filter for this tile to increase data reuse.
  vec4 F[4][IVC_OC];

  uvec3 wgID = gl_WorkGroupID;
  uvec2 laneID = gl_LocalInvocationID.xy;
  uvec2 laneCount = gl_WorkGroupSize.xy;

  uint wgBaseOC = wgID.x * WG_TILE_OC; // Workgroup base output channel
  uint wgBaseOW = wgID.y * WG_TILE_OW; // Workgroup base output width
  uint wgOH = wgID.z;                  // Workgroup output height

  // Initialize the output for this batch to zero.
  [[unroll]] for (uint i = 0; i < IVC_OW; ++i) {
    [[unroll]] for (uint j = 0; j < IVC_OC; ++j) {
      O[i][j] = vec4(0.f, 0.f, 0.f, 0.f);
    }
  }

  for (uint fh = 0; fh < FH; ++fh) {
    for (uint fw = 0; fw < FW; ++fw) {
      // Tile input channel with each tile having 4 elements.
      for (uint ic = 0; ic < IC; ic += 4) {
        // Load the filter for this input channel tile.
        [[unroll]] for (uint i = 0; i < 4; ++i) {
          [[unroll]] for (uint j = 0; j < IVC_OC; ++j) {
            uint oc = (laneID.x + laneCount.x * j) * 4 + wgBaseOC;
            F[i][j] = Filter.data[filterCoordToOffset(fh, fw, ic + i, oc)];
          }
        }

        // Load this input channel tile and perform dot product with filters
        // for different output channels.
        [[unroll]] for (uint i = 0; i < IVC_OW; ++i) {
          [[unroll]] for (uint j = 0; j < IVC_OC; ++j) {
            uint ow = i + laneID.y * IVC_OW + wgBaseOW;
            vec4 feature = Input.data[inputCoordToOffset(wgOH * SH + fh, ow * SW + fw, ic)];
            O[i][j] += vec4(feature.x, feature.x, feature.x, feature.x) * F[0][j];
            O[i][j] += vec4(feature.y, feature.y, feature.y, feature.y) * F[1][j];
            O[i][j] += vec4(feature.z, feature.z, feature.z, feature.z) * F[2][j];
            O[i][j] += vec4(feature.w, feature.w, feature.w, feature.w) * F[3][j];
          }
        }
      }
    }
  }

  // Write out the computed output elements.
  [[unroll]] for (uint i = 0; i < IVC_OW; ++i) {
    [[unroll]] for (uint j = 0; j < IVC_OC; ++j) {
      uint ow = i + laneID.y * IVC_OW + wgBaseOW;
      uint oc = (laneID.x + laneCount.x * j) * 4 + wgBaseOC;
      Output.data[outputCoordToOffset(wgOH, ow, oc)] = O[i][j];
    }
  }
}
