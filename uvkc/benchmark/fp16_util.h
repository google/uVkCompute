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

size_t size(Precision precision);

// Class to emulate half float on CPU.
class fp16 {
 public:
  fp16(uint16_t v) { value = v; }
  void fromFloat(const float &x) {
    uint32_t asInt = *(uint32_t *)&x;
    int sign = (asInt & 0x80000000) >> 31;
    int exp = ((asInt & 0x7f800000) >> 23) - 127 + 15;
    int mantissa = (asInt & 0x7FFFFF);
    if (exp > 31) exp = 31;
    if (exp < 0) exp = 0;
    sign = sign << 15;
    exp = exp << 10;
    mantissa = mantissa >> (23 - 10);
    asInt = sign | exp | mantissa;
    value = asInt;
  }
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
  float toFloat() const {
    uint32_t asInt = value;
    int sign = (asInt & 0x8000) >> 15;
    int exp = ((asInt & 0x7c00) >> 10);
    int mantissa = (asInt & 0x3FF);
    sign = sign << 31;
    if (exp > 0) {
      exp = (exp + 127 - 15) << 23;
      mantissa = mantissa << (23 - 10);
    } else {
      mantissa = 0;
    }
    asInt = sign | exp | mantissa;
    return *(float *)&asInt;
  }
  operator float() { return toFloat(); }
  uint16_t getValue() { return value; }

 private:
  uint16_t value;
};
}  // namespace benchmark
}  // namespace uvkc