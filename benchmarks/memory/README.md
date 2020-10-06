# Memory Characteristics Benchmarks

This directory contains microbenchmarks for probing the target GPU's memory
characteristics.

### `copy_storage_buffer_scalar`

Copies the tightly packed data array in one storage buffer A to another
storage buffer B: each invocation just copies one scalar in the array.

Benchmarks memory throughout.
