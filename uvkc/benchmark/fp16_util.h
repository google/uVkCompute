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
#include <memory>

namespace uvkc {
namespace benchmark {
enum class Precision {
  fp16,
  fp32,
};

size_t GetSize(Precision precision);

// Class to emulate half float on CPU.
class fp16 {
 public:
  fp16(uint16_t v) : value_(v) {}
  fp16(const float &x) { fromFloat(x); }

  fp16 &operator=(const float &x) {
    fromFloat(x);
    return *this;
  }
  fp16 &operator=(const int &x) {
    fromFloat((float)x);
    return *this;
  }
  fp16 &operator+=(const fp16 &x) {
    fromFloat(toFloat() + x.toFloat());
    return *this;
  }
  operator float() { return toFloat(); }

  void fromFloat(const float &x);
  float toFloat() const;
  uint16_t getValue() { return value_; }

 private:
  uint16_t value_;
};
}  // namespace benchmark
}  // namespace uvkc
