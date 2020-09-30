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

# Defines a C++ binary
#
# Given a binary 'foo' defined in root/a/b, this function will define a CMake
# target 'root_a_b_foo', which generates a binary of name 'foo'.
#
# Parameters:
#
# * NAME: the name of this binary
# * SRCS: the list of source files for this binary
# * DEPS: the list of libraries that this binary depends on
# * COPTS: the list of private compile options
# * INCLUDES: the list of additional private include directories
function(uvkc_cc_binary)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;DEPS;COPTS;INCLUDES"
    ${ARGN}
  )

  uvkc_package_ns(_PACKAGE_NS)

  # Use fully qualified namespaces for dependencies
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  # Prefix the library with the package name
  uvkc_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  add_executable(${_NAME} "")
  # Create an alias executable for the original given name
  add_executable(${_RULE_NAME} ALIAS "${_NAME}")
  # Use the original given name for the output binary
  set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "${_RULE_NAME}")

  target_sources(${_NAME} PRIVATE ${_RULE_SRCS})

  target_include_directories(${_NAME}
    PRIVATE
      "$<BUILD_INTERFACE:${UVKC_SOURCE_ROOT}>"
      "$<BUILD_INTERFACE:${_RULE_INCLUDES}>"
  )

  target_compile_options(${_NAME} PRIVATE ${_RULE_COPTS})

  target_link_libraries(${_NAME} PUBLIC ${_RULE_DEPS})
endfunction()

