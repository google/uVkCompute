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

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "uvkc/base/dynamic_library.h"
#include "uvkc/base/target_platform.h"

#if defined(UVKC_PLATFORM_ANDROID) || defined(UVKC_PLATFORM_APPLE) || \
    defined(UVKC_PLATFORM_LINUX)

#include <dlfcn.h>

namespace uvkc {

class DynamicLibraryPosix : public DynamicLibrary {
 public:
  ~DynamicLibraryPosix() override { ::dlclose(library_); }

  static absl::StatusOr<std::unique_ptr<DynamicLibrary>> Load(
      absl::Span<const char *const> search_file_names) {
    for (int i = 0; i < search_file_names.size(); ++i) {
      void *library = ::dlopen(search_file_names[i], RTLD_LAZY | RTLD_LOCAL);
      if (library) {
        return absl::WrapUnique(
            new DynamicLibraryPosix(search_file_names[i], library));
      }
    }
    std::string error =
        absl::StrCat("Unable to open dynamic library: ", dlerror());
    return absl::UnavailableError(error.c_str());
  }

  void *GetSymbol(const char *symbol_name) const override {
    return ::dlsym(library_, symbol_name);
  }

 private:
  DynamicLibraryPosix(std::string file_name, void *library)
      : DynamicLibrary(file_name), library_(library) {}

  void *library_;
};

// static
absl::StatusOr<std::unique_ptr<DynamicLibrary>> DynamicLibrary::Load(
    absl::Span<const char *const> search_file_names) {
  return DynamicLibraryPosix::Load(search_file_names);
}

}  // namespace uvkc

#endif  // UVKC_PLATFORM_*
