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

# Defines a C++ library
#
# Given a library 'foo' defined in root/a/b, this function will define a CMake
# target 'root_a_b_foo' and its alias 'root::a::b::foo'.
#
# Parameters:
#
# * NAME: the name of this library
# * HDRS: the list of public headers for this library
# * SRCS: the list of source files for this library
# * DEPS: the list of libraries that this library depends on
# * INCLUDES: the list of additional public include directories to this library
# * COPTS: the list of private compile options
# * LINKOPTS: the list of private link options
function(uvkc_cc_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "HDRS;SRCS;DEPS;INCLUDES;COPTS;LINKOPTS"
    ${ARGN}
  )

  uvkc_package_ns(_PACKAGE_NS)

  # Use fully qualified namespaces for dependencies
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  # Prefix the library with the package name
  uvkc_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  add_library(${_NAME} STATIC "")
  # Create an alis library with the namespaced name for dependency reference use
  add_library(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_NAME})

  target_sources(${_NAME} PRIVATE ${_RULE_SRCS} ${_RULE_HDRS})

  target_include_directories(${_NAME}
    PUBLIC
      "$<BUILD_INTERFACE:${_RULE_INCLUDES}>"
      # Include directory for uvkc libraries
      "$<BUILD_INTERFACE:${UVKC_SOURCE_ROOT}>"
      # Include directory for abseil libraries
      "$<BUILD_INTERFACE:${UVKC_SOURCE_ROOT}/third_party/abseil-cpp>"
  )

  target_compile_options(${_NAME} PRIVATE ${_RULE_COPTS})

  target_link_libraries(${_NAME}
    PUBLIC ${_RULE_DEPS}
    PRIVATE ${_RULE_LINKOPTS}
  )
endfunction()
