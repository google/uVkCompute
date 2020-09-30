# µVkCompute

µVkCompute is a micro Vulkan compute pipeline and a collection of compute
shaders for benchmarking/profiling purposes.

## Rationale

[Vulkan][vulkan] provides a ubiquitous way to access GPUs by many hardware
vendors across different form factors on various platforms. The great
reachability not only benefits graphics rendering; it can also be leveraged
for general compute, given that Vulkan is both a graphics and compute API.

However, being able to target various GPUs does not mean one size fits all.
Developers still needs to understand the characteristics of the target hardware
to gain the best utilization. A simple pipeline and a collection of shaders
to probe various characteristics of the target hardware often come as handy
for the purpose. Thus this repository.

## Goals

µVkCompute meant to provide a straightforward compute pipeline to facilitate
writing compute shader microbenchmarks. It tries to

* Hide Vulkan boilerplate that are required for every Vulkan application, e.g.,
  Vulkan instance and device creation.
* Simplify shader resource managemnet, e.g., using reflection over SPIR-V to
  construct pipeline layouts and compute pipelines.
* Provide thin wrapper over command buffer construction and shader dispatch.

µVkCompute does not try to demostrate Vulkan programming best practices. For
example, it just uses the system allocator and allocates separate memory for
each buffer. Simplicity is favored instead of building a production-level
Vulkan application.

## Dependencies

This repository requires a common C++ project development environment:

* [CMake][cmake] with version >= 3.7
* (Optional) the [Ninja][ninja] build system
* A C/C++ compiler that supports C11/C++14

It additionally requires the [Vulkan SDK][vulkan-sdk], which will be used for
both the Vulkan shared library and shader compilers like `glslc` for (GLSL)
and `dxc` (for HLSL).

## Building

```shell
git clone https://github.com/google/uVkCompute.git
cd uVkCompute
git submodule update --init
mkdir build && cd build
cmake ..
ninja
```

[cmake]: https://cmake.org/
[ninja]: https://ninja-build.org/
[vulkan]: https://www.khronos.org/vulkan/
[vulkan-sdk]: https://www.lunarg.com/vulkan-sdk/
