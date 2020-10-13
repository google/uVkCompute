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

include(CMakeParseArguments)

# Generates a C array variable for a SPIR-V shader module
#
# Given a GLSL shader 'foo' defined in root/a/b, this function defines a CMake
# target 'root_a_b_foo' and its alias 'root::a::b::foo', which invokes glslc
# to compile the GLSL source code into SPIR-V as an array of uint32_t numbers,
# so that it can be included in C++ source code. The generated SPIR-V uint32_t
# array will be put in a file named as 'foo_spirv_instance.inc' under the build
# directory for root/a/b. A target depending on 'root::a::b::foo' will have
# the proper include directory set up so it can directly include the
# 'foo_spirv_instance.inc' file.
#
# Parameters:
#
# * NAME: the name of this shader instance
# * SRC: the GLSL source code for this shader instance
# * GLSLC_ARGS: the list of additional arguments to glslc
function(uvkc_glsl_shader_instance)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRC"
    "GLSLC_ARGS"
    ${ARGN}
  )

  uvkc_package_ns(_PACKAGE_NS)

  # Prefix the library with the package name
  uvkc_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  # The list of output artifacts
  set(_OUTS "")

  set(_SPIRV_CODE "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_spirv_instance.inc")
  set(_SPIRV_ASM "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_spirv_instance.spvasm")
  list(APPEND _OUTS "${_SPIRV_CODE}")

  # Add a custom command to invoke glslc to compile the GLSL source code
  add_custom_command(
    OUTPUT
      "${_SPIRV_CODE}"
      "${_SPIRV_ASM}"
    COMMAND
      "${Vulkan_GLSLC_EXECUTABLE}"
        -c -O -fshader-stage=compute
        -mfmt=num
        "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRC}"
        -o "${_SPIRV_CODE}"
        ${_RULE_GLSLC_ARGS}
    # Also generate the SPIR-V assembly to ease inspection
    COMMAND
      "${Vulkan_GLSLC_EXECUTABLE}"
        -S -O -fshader-stage=compute
        "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRC}"
        -o "${_SPIRV_ASM}"
        ${_RULE_GLSLC_ARGS}
    WORKING_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}"
    MAIN_DEPENDENCY
      "${_RULE_SRC}"
    DEPENDS
      Vulkan::glslc
      "${_RULE_SRC}"
    COMMAND_EXPAND_LISTS
  )

  # Define a target to drive the artifact generation
  set(_GEN_TARGET "${_NAME}_gen")
  add_custom_target(
    ${_GEN_TARGET}
    DEPENDS
      ${_OUTS}
  )

  # Define an interface library with the name of this shader instance so that
  # we can depend on it in other targets.
  add_library(${_NAME} INTERFACE)
  # Create an alis library with the namespaced name for dependency reference use
  add_library(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_NAME})

  add_dependencies(${_NAME} ${_GEN_TARGET})

  # Propagate the include directory so dependencies can include the artifacts
  target_include_directories(${_NAME}
    INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
  )
endfunction()
