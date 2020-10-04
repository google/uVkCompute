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

uvkc_cc_library(
  NAME
    file
  HDRS
    "file.h"
  SRCS
    "file.cc"
  DEPS
    absl::status
    absl::statusor
)

uvkc_cc_library(
  NAME
    log
  HDRS
    "log.h"
  SRCS
    "log.cc"
)

uvkc_cc_library(
  NAME
    dynamic_library
  HDRS
    "dynamic_library.h"
  SRCS
    "dynamic_library_posix.cc"
    "dynamic_library_win32.cc"
  DEPS
    absl::span
    absl::statusor
    absl::strings
  LINKOPTS
    ${CMAKE_DL_LIBS} # for dlopen/dlclose on Linux
)
