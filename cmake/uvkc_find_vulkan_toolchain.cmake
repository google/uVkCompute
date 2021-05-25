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

# Find the Vulkan SDK and shader toolchain

include(FindVulkan)
if(NOT Vulkan_FOUND)
  message(FATAL_ERROR "Unable to find the Vulkan SDK")
endif()

# Find glslc for compiling GLSL source code
if(NOT Vulkan_GLSLC_EXECUTABLE)
  if(WIN32)
    find_program(Vulkan_GLSLC_EXECUTABLE
      NAMES glslc
      HINTS "$ENV{VULKAN_SDK}/Bin32"
    )
  else(WIN32)
    find_program(Vulkan_GLSLC_EXECUTABLE
      NAME glslc
      HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{ANDROID_NDK}/shader-tools/*")
  endif(WIN32)
endif()

if (Vulkan_GLSLC_EXECUTABLE AND NOT TARGET Vulkan::glslc)
  add_executable(Vulkan::glslc IMPORTED)
  set_property(TARGET Vulkan::glslc PROPERTY
    IMPORTED_LOCATION "${Vulkan_GLSLC_EXECUTABLE}")
endif()

# Find dxc for compiling GLSL source code
if(NOT Vulkan_DXC_EXECUTABLE)
  if(WIN32)
    find_program(Vulkan_DXC_EXECUTABLE
      NAMES dxc
      HINTS "$ENV{VULKAN_SDK}/Bin32"
    )
  else(WIN32)
    find_program(Vulkan_DXC_EXECUTABLE
      NAME dxc
      HINTS "$ENV{VULKAN_SDK}/bin")
  endif(WIN32)
endif()

if (Vulkan_DXC_EXECUTABLE AND NOT TARGET Vulkan::dxc)
  add_executable(Vulkan::dxc IMPORTED)
  set_property(TARGET Vulkan::dxc PROPERTY
    IMPORTED_LOCATION "${Vulkan_DXC_EXECUTABLE}")
endif()

# Find spirv-as for compiling SPIR-V assembly code
if(NOT Vulkan_SPIRVAS_EXECUTABLE)
  if(WIN32)
    find_program(Vulkan_SPIRVAS_EXECUTABLE
      NAMES spirv-as
      HINTS "$ENV{VULKAN_SDK}/Bin32"
    )
  else(WIN32)
    find_program(Vulkan_SPIRVAS_EXECUTABLE
      NAME spirv-as
      HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{ANDROID_NDK}/shader-tools/*")
  endif(WIN32)
endif()

if (Vulkan_SPIRVAS_EXECUTABLE AND NOT TARGET Vulkan::spirvas)
  add_executable(Vulkan::spirvas IMPORTED)
  set_property(TARGET Vulkan::spirvas PROPERTY
    IMPORTED_LOCATION "${Vulkan_SPIRVAS_EXECUTABLE}")
endif()
