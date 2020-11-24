# Microbenchmarks

This directory contains various microbenchmarks. Please see each directory
for the details of the benchmarks.

## How to add a benchmark

Existing benchmarks can be good reference of how to add a new benchmark.
In general, to add a benchmark `foo`, the procedure should be:

* Add a `foo.glsl` file for the GLSL source code.
* Add a `foo_main.cc` file for the Vulkan C++ code.
* Rgister these two files in `CMakeLists.txt`:
  * Use `uvkc_glsl_shader_instance` (for generating a single SPIR-V shader
    module) or `uvkc_glsl_shader_permutation` (for generating a corpus of
    SPIR-V shader modules) for `foo.glsl`.
  * Use `uvkc_cc_binary` for `foo_main.cc`. Please also make sure to add
    `uvkc::benchmark::main` as a dependency for the `main()` function.
* In `foo_main.cc`:
  * `#include` the generated SPIR-V code: `foo_spirv_instance.inc` or
    `foo_spirv_permutation.inc`.
  * Define `uvkc::benchmark::CreateVulkanContext()` for creating the Vulkan
    context the persists among benchmark invocations.
  * Define `uvkc::benchmark::RegisterVulkanOverheadBenchmark()` for registering
    a benchmark to evaluate base latency overhead that should be subtracted
    from normal benchmark latency measurements.
  * Define `uvkc::benchmark::RegisterVulkanBenchmarks()` for programmatically
    registering benchmarks. Please refer to
    [Google Benchmark](https://github.com/google/benchmark) for APIs.

## How to run a benchmark

Benchmarks are stand-alone executables that pack all necessary resources inside,
including the SPIR-V shader code. So one can copy it anywhere and execute.

There are some common command-line options supported by all benchmark
executables:

### `--latency_measure_mode`

This option controls how shader execution time is measured. Right now three
modes are supported:

* `system_submit`: time spent from queue submit to returning from queue wait.
* `system_dispatch`: `system_submit` subtracted by time for a void dispatch.
  This tries to remove the overhead of queue submit and wait so that we can
  evaluate only the kernel "dispatch" time.
* `gpu_timestamp`: timestamp difference between top and bottom of the pipeline
  measured on GPU. This requires the GPU supports timestamp query.

### `--benchmark_*`

Various Google Benchmark control options. See `--help` for details.
