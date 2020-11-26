# Convolution Benchmarks

This directory contains microbenchmarks for evaluating different strategy to
implement convolution.

### `conv2d`

Calculate 2-D convolution by tiling along both the output width dimension and
output channel dimension. Each tile is assigned to one workgroup, which is
further distributed among the invocations in the workgroup.

Invocations perform 4-element vector load of the filter in a cyclic way.
Each invocation calculates the full result of the output element it covers
so no synchronization. Invocations write out the output elements in a cyclic
way.

Benchmark 2-D convolution throughput.

### `depthwise_conv2d`

Calculate 2-D convolution by tiling along both the output width dimension and
output channel dimension. Each tile is assigned to one workgroup, which is
further distributed among the invocations in the workgroup.

Invocations perform 4-element vector load of the filter in a cyclic way.
Each invocation calculates the full result of the output element it covers
so no synchronization. Invocations write out the output elements in a cyclic
way.

Benchmark depthwise 2-D convolution throughput.
