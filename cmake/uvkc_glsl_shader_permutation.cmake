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

# Generates C array variables for multiple SPIR-V shader modules
#
# Given a GLSL shader 'foo' defined in root/a/b, this function defines a CMake
# target 'root_a_b_foo' and its alias 'root::a::b::foo', which invokes glslc
# to compile the GLSL source code into SPIR-V as multiple arrays of uint32_t
# numbers, by permuting the choices presented in the 'PERMUTATION' parameter.
# The generated SPIR-V uint32_t arrays will be put in a file named as
# 'foo_spirv_permutation.inc' under the build directory for root/a/b. A target
# depending on 'root::a::b::foo' will have the proper include directory set up
# so it can directly include the 'foo_spirv_permutation.inc' file.
#
# Parameters:
#
# * NAME: the name of this shader instance
# * SRC: the GLSL source code for this shader instance
# * PERMUTATION: an array of macros together with choices in the form of
#                'FOO=[foo1|foo2]'
# * GLSLC_ARGS: the list of additional arguments to glslc
#
# Permutation:
#
# For example, given the following:
#   uvkc_glsl_shader_permutation(
#     NAME
#       matmul_corpus
#     SRC
#       matmul.glsl # contains macro TILE_M and TILE_N
#     PERMUTATION
#       "TILE_M=[16|32]"
#       "TILE_N=[8|16]"
#       "{X,Y}=[{1,2}|{3,4}]"
#   )
# glslc will be invoked four times with the following definitions:
#   - glslc .. -DTILE_M=16 -DTILE_N=8  -DX=1 -DY=2
#   - glslc .. -DTILE_M=16 -DTILE_N=8  -DX=3 -DY=4
#   - glslc .. -DTILE_M=16 -DTILE_N=16 -DX=1 -DY=2
#   - glslc .. -DTILE_M=16 -DTILE_N=16 -DX=3 -DY=4
#   - glslc .. -DTILE_M=32 -DTILE_N=8  -DX=1 -DY=2
#   - glslc .. -DTILE_M=32 -DTILE_N=8  -DX=3 -DY=4
#   - glslc .. -DTILE_M=32 -DTILE_N=16 -DX=1 -DY=2
#   - glslc .. -DTILE_M=32 -DTILE_N=16 -DX=3 -DY=4
# And each of the above will generate an uint32_t array with name:
#   - TILE_M_16_TILE_N_8_X_1_Y_2
#   - TILE_M_16_TILE_N_8_X_3_Y_4
#   - TILE_M_16_TILE_N_16_X_1_Y_2
#   - TILE_M_16_TILE_N_16_X_3_Y_4
#   - TILE_M_32_TILE_N_8_X_1_Y_2
#   - TILE_M_32_TILE_N_8_X_3_Y_4
#   - TILE_M_32_TILE_N_16_X_1_Y_2
#   - TILE_M_32_TILE_N_16_X_3_Y_4
# In the 'matmul_corpus_spirv_permutation.inc' file.
function(uvkc_glsl_shader_permutation)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRC"
    "GLSLC_ARGS;PERMUTATION"
    ${ARGN}
  )

  uvkc_package_ns(_PACKAGE_NS)

  # Prefix the library with the package name
  uvkc_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  # The list of output artifacts
  set(_OUTS "")

  set(_SPIRV_CODE "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_spirv_permutation.inc")
  list(APPEND _OUTS "${_SPIRV_CODE}")

  set(_PERMUTATION "")
  foreach(_MACRO IN LISTS _RULE_PERMUTATION)
    list(APPEND _PERMUTATION "--define")
    # Quote each define so that we passing to glslc they are treated as
    # proper strings.
    list(APPEND _PERMUTATION "\"${_MACRO}\"")
  endforeach()

  set(_GLSLC_ARGS "")
  foreach(_ARG IN LISTS _RULE_GLSLC_ARGS)
    list(APPEND _GLSLC_ARGS "--glslc-arg=${_ARG}")
  endforeach()

  # Add a custom command to invoke glslc to compile the GLSL source code
  add_custom_command(
    OUTPUT
      "${_SPIRV_CODE}"
    COMMAND
    "${Python3_EXECUTABLE}"
        "${UVKC_SOURCE_ROOT}/tools/generate_shader_permutations.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRC}"
        -o "${_SPIRV_CODE}"
        --glslc "${Vulkan_GLSLC_EXECUTABLE}"
        ${_PERMUTATION}
        ${_GLSLC_ARGS}
    WORKING_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}"
    MAIN_DEPENDENCY
      "${_RULE_SRC}"
    DEPENDS
      "${UVKC_SOURCE_ROOT}/tools/generate_shader_permutations.py"
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

  # Define an interface library with the name of this shader permtation so that
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

