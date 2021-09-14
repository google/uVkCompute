# µVkCompute

![Android/Linux Build Status](https://github.com/google/uVkCompute/actions/workflows/build-android-linux.yml/badge.svg)
[![Windows Build status](https://ci.appveyor.com/api/projects/status/qnk10ht2ighagrdg/branch/main?svg=true)](https://ci.appveyor.com/project/antiagainst/uvkcompute/branch/main)

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

µVkCompute focuses more on single compute shader dispatch. µVkCompute does not
try to demostrate Vulkan programming best practices. For example, it just uses
the system allocator and allocates separate memory for each buffer. Simplicity
is favored instead of building a production-level Vulkan application.

## Dependencies

This repository requires a common C++ project development environment:

* [CMake][cmake] with version >= 3.13
* (Optional) the [Ninja][ninja] build system
* A C/C++ compiler that supports C11/C++14
* Python3

It additionally requires the [Vulkan SDK][vulkan-sdk], which will be used for
both the Vulkan shared library and shader compilers like `glslc` for (GLSL)
and `dxc` (for HLSL). Please make sure you have set the `VULKAN_SDK` environment
variable.

## Building and Running

### Android

```shell
git clone https://github.com/google/uVkCompute.git
cd uVkCompute
git submodule update --init

cmake -G Ninja -S ./ -B build-android/  \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK?}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-29
cmake --build build-android/
```

Where `ANDROID_NDK` is the path to the [Android NDK
installation][android-ndk-install]. See Android's CMake guide for explanation
over [`ANDROID_ABI`][android-abi] and [`ANROID_PLATFORM`][android-platform].

Afterwards, you can use `adb push` and `adb shell` to run the benchmark binaries
generated into the `build-android/` directory on Android devices. For example,
for a benchmark binary `bench` at `build-android/benchmarks/foo/bar/bench`:

```shell
# Push the benchmark to the Android device
adb push build-android/benchmarks/foo/bar/bench /data/local/tmp
adb shell "cd /data/local/tmp && ./bench"
```

Note that for Android 10, if you see the "Failed to match any benchmarks against
regex: ." error message, it means that no [Vulkan ICDs][vulkan-icd] (a.k.a.,
Vulkan vendor drivers) are discovered. This is a known issue that is fixed in
Android 11. A workaround is to copy the Vulkan ICD (normally as
`/vendor/lib[64]/hw/vulkan.*.so`) to `/data/local/tmp` and run the benchmark
binary with `LD_LIBRARY_PATH=/data/local/tmp`.

### Linux/macOS

```shell
git clone https://github.com/google/uVkCompute.git
cd uVkCompute
git submodule update --init

cmake -G Ninja -S ./ -B build/
cmake --build build/
```

Afterwards you can run the benchmark binaries generated into the `build/`
directory on the host machine.

### Windows

```shell
git clone https://github.com/google/uVkCompute.git
cd uVkCompute
git submodule update --init

cmake -G "Visual Studio 16 2019" -A x64 -S ./ -B build/
cmake --build build/
```

Afterwards you can run the benchmark binaries generated into the `build/`
directory on the host machine.

[android-abi]: https://developer.android.com/ndk/guides/cmake#android_abi
[android-ndk-install]: https://developer.android.com/ndk/downloads
[android-platform]: https://developer.android.com/ndk/guides/cmake#android_platform
[cmake]: https://cmake.org/
[ninja]: https://ninja-build.org/
[vulkan]: https://www.khronos.org/vulkan/
[vulkan-icd]: https://github.com/KhronosGroup/Vulkan-Loader/blob/master/loader/LoaderAndLayerInterface.md#installable-client-drivers
[vulkan-sdk]: https://www.lunarg.com/vulkan-sdk/
