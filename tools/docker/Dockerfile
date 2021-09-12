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

FROM ubuntu:20.04
WORKDIR /usr/src/

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  apt-transport-https \
  ca-certificates \
  clang \
  cmake \
  git \
  gnupg2 \
  ninja-build \
  software-properties-common \
  unzip \
  wget \
  # zlib is needed to compile SwiftShader.
  zlib1g-dev

ENV CC=/usr/bin/clang
ENV CXX=/usr/bin/clang++

#===------------------------------------------------------------------------===#
# Android NDK
#===------------------------------------------------------------------------===#

ARG ANDROID_NDK_VERSION=r23

RUN wget "https://dl.google.com/android/repository/android-ndk-${ANDROID_NDK_VERSION?}-linux.zip" \
  && unzip "android-ndk-${ANDROID_NDK_VERSION?}-linux.zip" -d /android-ndk && \
  rm -rf /usr/src

ENV ANDROID_NDK="/android-ndk/android-ndk-${ANDROID_NDK_VERSION}"

#===------------------------------------------------------------------------===#
# Vulkan SDK
#===------------------------------------------------------------------------===#

ARG VULKAN_SDK_VERSION=1.2.189

RUN wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add - \
  && wget -qO \
    "/etc/apt/sources.list.d/lunarg-vulkan-${VULKAN_SDK_VERSION?}-focal.list" \
    "http://packages.lunarg.com/vulkan/${VULKAN_SDK_VERSION?}/lunarg-vulkan-${VULKAN_SDK_VERSION?}-focal.list" \
  && apt-get update \
  && apt-get install -y vulkan-sdk

#===------------------------------------------------------------------------===#
# SwiftShader Vulkan ICD
#===------------------------------------------------------------------------===#

ARG SWIFTSHADER_COMMIT=7f2c7d18de0c916a326d5bf4d3c2d301eff2f4ee

RUN git clone https://github.com/google/swiftshader \
  && cd swiftshader && git checkout "${SWIFTSHADER_COMMIT?}" && cd .. \
  && cmake -S swiftshader/ -B build-swiftshader/ \
           -GNinja \
           -DSWIFTSHADER_BUILD_VULKAN=ON \
           -DSWIFTSHADER_BUILD_EGL=OFF \
           -DSWIFTSHADER_BUILD_GLESv2=OFF \
           -DSWIFTSHADER_BUILD_PVR=OFF \
           -DSWIFTSHADER_BUILD_TESTS=OFF \
  && cmake --build build-swiftshader/ \
           --config Release \
           --target vk_swiftshader \
  && cp -rf build-swiftshader/Linux /swiftshader \
  && echo "${SWIFTSHADER_COMMIT?}" > /swiftshader/git-commit \
  && rm -rf /usr/src
