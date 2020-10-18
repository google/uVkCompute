# Memory Characteristics Benchmarks

This directory contains microbenchmarks for probing the target GPU's memory
characteristics.

### `copy_sampled_image_to_storage_buffer`

Copies all the data from 2-D sampled image to a storage buffer. Each invocation
copies one pixel (of float type) in the image.

Benchmarks memory bandwidth w.r.t. texture read.

### `copy_storage_buffer`

Copies the tightly packed data array in one storage buffer A to another
storage buffer B: each invocation copies either one scalar or a 4-element vector
in the array.

Benchmarks memory bandwidth w.r.t. storage buffer.
