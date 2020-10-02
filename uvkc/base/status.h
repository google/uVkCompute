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

#ifndef UVKC_STATUS_H_
#define UVKC_STATUS_H_

#include "absl/status/status.h"

//===----------------------------------------------------------------------===//
// Utility macros
//===----------------------------------------------------------------------===//

// Executes an expression `rexpr` that returns a `absl::Status`. On error,
// returns from the current function.
#define UVKC_RETURN_IF_ERROR(rexpr) \
  UVKC_RETURN_IF_ERROR_INNER_(      \
      UVKC_STATUS_IMPL_CONCAT_(_status_object, __LINE__), rexpr)

// Executes an expression `rexpr` that returns a `absl::StatusOr<T>`. On OK,
// moves its value into the variable defined by `lhs`, otherwise returns
// from the current function.
#define UVKC_ASSIGN_OR_RETURN(lhs, rexpr) \
  UVKC_ASSIGN_OR_RETURN_INNER_(           \
      UVKC_STATUS_IMPL_CONCAT_(_status_or_object, __LINE__), lhs, rexpr)

//===----------------------------------------------------------------------===//
// Macros internals
//===----------------------------------------------------------------------===//

#define UVKC_RETURN_IF_ERROR_INNER_(status, rexpr) \
  do {                                             \
    auto status = rexpr;                           \
    if (!status.ok()) return status;               \
  } while (0)

#define UVKC_ASSIGN_OR_RETURN_INNER_(statusor, lhs, rexpr) \
  auto statusor = rexpr;                                   \
  if (!statusor.ok()) {                                    \
    return std::move(statusor).status();                   \
  }                                                        \
  lhs = std::move(statusor).value()

#define UVKC_STATUS_IMPL_CONCAT_(x, y) UVKC_STATUS_IMPL_CONCAT_INNER_(x, y)

#define UVKC_STATUS_IMPL_CONCAT_INNER_(x, y) x##y

#endif  // UVKC_STATUS_H_
