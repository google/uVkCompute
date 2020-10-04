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

#include "uvkc/base/target_platform.h"

#if defined(UVKC_PLATFORM_WINDOWS)

#include <windows.h>

#include "absl/memory/memory.h"
#include "uvkc/base/dynamic_library.h"

namespace uvkc {

class DynamicLibraryWin : public DynamicLibrary {
 public:
  ~DynamicLibraryWin() override { ::FreeLibrary(library_); }

  static absl::StatusOr<std::unique_ptr<DynamicLibrary>> Load(
      absl::Span<const char *const> search_file_names) {
    for (int i = 0; i < search_file_names.size(); ++i) {
      HMODULE library = ::LoadLibraryA(search_file_names[i]);
      if (library) {
        return absl::WrapUnique(
            new DynamicLibraryWin(search_file_names[i], library));
      }
    }

    return absl::UnavailableError(
        "Unable to open dynamic library, not found on search paths");
  }

  void *GetSymbol(const char *symbol_name) const override {
    return reinterpret_cast<void *>(::GetProcAddress(library_, symbol_name));
  }

 private:
  DynamicLibraryWin(std::string file_name, HMODULE library)
      : DynamicLibrary(file_name), library_(library) {}

  HMODULE library_;
};

// static
absl::StatusOr<std::unique_ptr<DynamicLibrary>> DynamicLibrary::Load(
    absl::Span<const char *const> search_file_names) {
  return DynamicLibraryWin::Load(search_file_names);
}

}  // namespace uvkc

#endif  // UVKC_PLATFORM_*
