# Matrix multiplication benchmark

This directory contains microbenchmarks for evaluating different strategy to
implement matrix multiplication.

### `matmul_tiled`

Calculate matrix multiplication using simple tiled strategy. Run different tile
sizes for M and N dimension to find the optimal tile size. Tile with a
dimension of 4 along K to allow using load4 when accessing A matrix.

Benchmark matrix multiply throughput.