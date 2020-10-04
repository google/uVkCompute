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

FROM ubuntu:18.04
WORKDIR /usr/src/

RUN apt-get update && apt-get install -y \
  apt-transport-https \
  ca-certificates \
  clang \
  git \
  gnupg2 \
  ninja-build \
  software-properties-common \
  unzip \
  wget

ENV CC=/usr/bin/clang
ENV CXX=/usr/bin/clang++

#===------------------------------------------------------------------------===#
# CMake
#===------------------------------------------------------------------------===#

# Install a newer version of CMake because SwiftShader requires CMake 3.13+.
RUN wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc \
      | gpg --dearmor - \
      | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
  && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' \
  && apt-get update \
  && apt-get install -y cmake

#===------------------------------------------------------------------------===#
# Android NDK
#===------------------------------------------------------------------------===#

ARG ANDROID_NDK_VERSION=r21d

RUN wget "https://dl.google.com/android/repository/android-ndk-${ANDROID_NDK_VERSION?}-linux-x86_64.zip"
RUN unzip "android-ndk-${ANDROID_NDK_VERSION?}-linux-x86_64.zip" -d /android-ndk

ENV ANDROID_NDK="/android-ndk/android-ndk-${ANDROID_NDK_VERSION}"

#===------------------------------------------------------------------------===#
# Vulkan SDK
#===------------------------------------------------------------------------===#

ARG VULKAN_SDK_VERSION=1.2.148

RUN wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add - \
  && wget -qO \
    "/etc/apt/sources.list.d/lunarg-vulkan-${VULKAN_SDK_VERSION?}-bionic.list" \
    "http://packages.lunarg.com/vulkan/${VULKAN_SDK_VERSION?}/lunarg-vulkan-${VULKAN_SDK_VERSION?}-bionic.list" \
  && apt-get update \
  && apt-get install -y vulkan-sdk

#===------------------------------------------------------------------------===#
# SwiftShader Vulkan ICD
#===------------------------------------------------------------------------===#

ARG SWIFTSHADER_COMMIT=b3c1e71eb1cb55ba78a3f884ab4e68ef1bd396da

# zlib is needed for compiling SwiftShader.
RUN apt-get update && apt-get install -y zlib1g-dev
RUN git clone https://github.com/google/swiftshader
RUN cd swiftshader && git checkout "${SWIFTSHADER_COMMIT?}" && cd ..
# Only build SwiftShader Vulkan ICD.
RUN cmake -S swiftshader/ -B build-swiftshader/ \
           -GNinja \
           -DSWIFTSHADER_BUILD_VULKAN=ON \
           -DSWIFTSHADER_BUILD_EGL=OFF \
           -DSWIFTSHADER_BUILD_GLESv2=OFF \
           -DSWIFTSHADER_BUILD_GLES_CM=OFF \
           -DSWIFTSHADER_BUILD_PVR=OFF \
           -DSWIFTSHADER_BUILD_TESTS=OFF
RUN cmake --build build-swiftshader/ \
           --config Release \
           --target vk_swiftshader
# Copy the ICD JSON and .so to a known place.
RUN cp -rf build-swiftshader/Linux /swiftshader
# Keep track of the commit we are using.
RUN echo "${SWIFTSHADER_COMMIT?}" > /swiftshader/git-commit


RUN rm -rf /usr/src/
