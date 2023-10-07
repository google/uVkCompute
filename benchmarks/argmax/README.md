# Reduction Benchmarks

This directory contains microbenchmarks for evaluating different strategy to
implement argmax.

### `one_workgroup_argmax`

Performs argmax using just one workgroup. The workgroup just contains one
subgroup. This approach does not use any synchronization mechanisms.

A subgroup uses either a single thread to loop over all elements or subgroup
reduction operations involving all invocations.
