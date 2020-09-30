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

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace uvkc {

// Reads the file at |path| and returns its contents as a string.
absl::StatusOr<std::string> ReadFile(const std::string& path);

// Writes the |content_size|-byte string starting at |content_data| into a file
// at the given |path|.
absl::Status WriteFile(const std::string& path, const char* content_data,
                       size_t content_size);

}  // namespace uvkc

#endif  // UVKC_STATUS_H_
