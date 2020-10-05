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

#include "uvkc/base/file.h"

#include <cstdio>
#include <memory>

#include "absl/status/status.h"

namespace uvkc {

absl::StatusOr<std::string> ReadFile(const std::string &path) {
  std::unique_ptr<FILE, void (*)(FILE *)> file = {std::fopen(path.c_str(), "r"),
                                                  +[](FILE *file) {
                                                    if (file) std::fclose(file);
                                                  }};
  if (file == nullptr) {
    return absl::InvalidArgumentError("cannot open file");
  }
  if (std::fseek(file.get(), 0, SEEK_END) == -1) {
    return absl::InvalidArgumentError("cannot seek forward in file");
  }
  size_t file_size = std::ftell(file.get());
  if (file_size == -1L) {
    return absl::InvalidArgumentError("cannot read file length");
  }
  if (std::fseek(file.get(), 0, SEEK_SET) == -1) {
    return absl::InvalidArgumentError("cannot seek back in file");
  }
  std::string contents;
  contents.resize(file_size);
  if (std::fread(const_cast<char *>(contents.data()), file_size, 1,
                 file.get()) != 1) {
    return absl::InvalidArgumentError("cannot read file content");
  }
  return contents;
}

absl::Status WriteFile(const std::string &path, const char *content_data,
                       size_t content_size) {
  std::unique_ptr<FILE, void (*)(FILE *)> file = {
      std::fopen(path.c_str(), "wb"), +[](FILE *file) {
        if (file) std::fclose(file);
      }};
  if (file == nullptr) {
    return absl::InvalidArgumentError("cannot open file");
  }
  if (std::fwrite(const_cast<char *>(content_data), content_size, 1,
                  file.get()) != 1) {
    return absl::InvalidArgumentError("cannot write file content");
  }
  return absl::OkStatus();
}

}  // namespace uvkc
