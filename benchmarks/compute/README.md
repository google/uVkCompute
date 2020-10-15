# Compute benchmarks

This directory contains microbenchmarks for measuring peak compute throughput
of a GPU target.

### `mad_throughput`

Measure the throughput of MAD operations by computing a large chain of MAD
operation in a loop. Use vector 4 operation so that the hardware can hide the
latency of operation by interleaving 4 scalar MAD.

Benchmarks compute throughput.
