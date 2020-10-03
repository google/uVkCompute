// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef UVKC_BASE_TARGET_PLATFORM_H_
#define UVKC_BASE_TARGET_PLATFORM_H_

//===------------------------------------------------------------------------===
// UVKC_ARCH_*
//===------------------------------------------------------------------------===

#if defined(__arm__) || defined(__arm64) || defined(__aarch64__) || \
    defined(__thumb__) || defined(__TARGET_ARCH_ARM) ||             \
    defined(__TARGET_ARCH_THUMB) || defined(_M_ARM)
#if defined(__arm64) || defined(__aarch64__)
#define UVKC_ARCH_ARM_64 1
#else
#define UVKC_ARCH_ARM_32 1
#endif  // __arm64
#endif  // ARM

#if defined(__wasm32__)
#define UVKC_ARCH_WASM_32 1
#elif defined(__wasm64__)
#define UVKC_ARCH_WASM_64 1
#elif defined(__asmjs__)
#define UVKC_ARCH_ASMJS 1
#endif  // wasm/asmjs

#if defined(__i386__) || defined(__i486__) || defined(__i586__) || \
    defined(__i686__) || defined(__i386) || defined(_M_IX86) || defined(_X86_)
#define UVKC_ARCH_X86_32 1
#elif defined(__x86_64) || defined(__x86_64__) || defined(__amd64__) || \
    defined(__amd64) || defined(_M_X64)
#define UVKC_ARCH_X86_64 1
#endif  // X86

#if !defined(UVKC_ARCH_ARM_32) && !defined(UVKC_ARCH_ARM_64) &&  \
    !defined(UVKC_ARCH_ASMJS) && !defined(UVKC_ARCH_WASM_32) &&  \
    !defined(UVKC_ARCH_WASM_64) && !defined(UVKC_ARCH_X86_32) && \
    !defined(UVKC_ARCH_X86_64)
#error Unknown architecture.
#endif  // all archs

//===------------------------------------------------------------------------===
// UVKC_COMPILER_*
//===------------------------------------------------------------------------===

#if defined(__clang__)
#define UVKC_COMPILER_CLANG 1
#define UVKC_COMPILER_GCC_COMPAT 1
#elif defined(__GNUC__)
#define UVKC_COMPILER_GCC 1
#define UVKC_COMPILER_GCC_COMPAT 1
#elif defined(_MSC_VER)
#define UVKC_COMPILER_MSVC 1
#else
#error Unrecognized compiler.
#endif  // compiler versions

//===------------------------------------------------------------------------===
// UVKC_PLATFORM_ANDROID
//===------------------------------------------------------------------------===

#if defined(__ANDROID__)
#define UVKC_PLATFORM_ANDROID 1
#endif  // __ANDROID__

//===------------------------------------------------------------------------===
// UVKC_PLATFORM_EMSCRIPTEN
//===------------------------------------------------------------------------===

#if defined(__EMSCRIPTEN__)
#define UVKC_PLATFORM_EMSCRIPTEN 1
#endif  // __ANDROID__

//===------------------------------------------------------------------------===
// UVKC_PLATFORM_IOS | UVKC_PLATFORM_MACOS
//===------------------------------------------------------------------------===

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define UVKC_PLATFORM_IOS 1
#else
#define UVKC_PLATFORM_MACOS 1
#endif  // TARGET_OS_IPHONE
#if TARGET_IPHONE_SIMULATOR
#define UVKC_PLATFORM_IOS_SIMULATOR 1
#endif  // TARGET_IPHONE_SIMULATOR
#endif  // __APPLE__

#if defined(UVKC_PLATFORM_IOS) || defined(UVKC_PLATFORM_MACOS)
#define UVKC_PLATFORM_APPLE 1
#endif  // UVKC_PLATFORM_IOS || UVKC_PLATFORM_MACOS

//===------------------------------------------------------------------------===
// UVKC_PLATFORM_LINUX
//===------------------------------------------------------------------------===

#if defined(__linux__) || defined(linux) || defined(__linux)
#define UVKC_PLATFORM_LINUX 1
#endif  // __linux__

//===------------------------------------------------------------------------===
// UVKC_PLATFORM_WINDOWS
//===------------------------------------------------------------------------===

#if defined(_WIN32) || defined(_WIN64)
#define UVKC_PLATFORM_WINDOWS 1
#endif  // _WIN32 || _WIN64

//===------------------------------------------------------------------------===
// Fallthrough for unsupported platforms
//===------------------------------------------------------------------------===

#if !defined(UVKC_PLATFORM_ANDROID) && !defined(UVKC_PLATFORM_EMSCRIPTEN) && \
    !defined(UVKC_PLATFORM_IOS) && !defined(UVKC_PLATFORM_LINUX) &&          \
    !defined(UVKC_PLATFORM_MACOS) && !defined(UVKC_PLATFORM_WINDOWS)
#error Unknown platform.
#endif  // all archs

#endif  // UVKC_BASE_TARGET_PLATFORM_H_
