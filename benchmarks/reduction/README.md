# Reduction Benchmarks

This directory contains microbenchmarks for evaluating different strategy to
implement reduction.

### `tree_reduce`

Performs tree reduction in-place over a data buffer. GPU invocations are
assigned to the data elements in a cyclic way so that their range don't overlap
with each other and we can write the partial result into the first data element.

A workgroup uses either a single thread to loop over all elements or subgroup
reduction operations involving all invocations.

### `atomic_reduce`

Divides the input buffer into batches and lets each workgroup handle one batch.
At the end each workgroup contributes the result back via atomic operations.
Emulates floating-point atomics via integer ones.

A workgroup uses either a single thread to loop over all elements or subgroup
reduction operations involving all invocations.

### `one_workgroup_reduce`

Performs reduction using just one workgroup. The workgroup just contains one
subgroup. This approach does not use any synchronization mechanisms.

A subgroup uses either a single thread to loop over all elements or subgroup
reduction operations involving all invocations.
