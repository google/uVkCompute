# Memory Characteristics Benchmarks

This directory contains microbenchmarks for probing the target GPU's memory
characteristics.

### `copy_storage_buffer`

Copies the tightly packed data array in one storage buffer A to another
storage buffer B: each invocation copies either one scalar or a 4-element vector
in the array.

Benchmarks memory bandwidth.
