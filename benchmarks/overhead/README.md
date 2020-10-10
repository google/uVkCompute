# Void Dispatch Benchmarks

This directory contains microbenchmarks for evaluating GPU overhead.
queue submit and wait.

### `dispatch_void_shader`

Submits and waits a command buffer that contains a one-workgroup dispatch
of a kernel.  The kernel does nothing.

Benchmarks queue submit and wait overhead.

