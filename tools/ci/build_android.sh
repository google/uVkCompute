#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# A script intended o be used with Docker for continuous integration

set -x
set -e
set -o pipefail

# Print information about the build environment
cmake --version
ninja --version
"${CC?}" --version
"${CXX?}" --version
python3 --version
echo "Android NDK path: ${ANDROID_NDK?}"

# Cross compile towards Android
SOURCE_ROOT=$(git rev-parse --show-toplevel)
cd "${SOURCE_ROOT?}"
git submodule update --init
cmake -G Ninja -B build-android/ -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK?}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_PLATFORM=android-29
cmake --build build-android/
