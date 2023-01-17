// Copyright 2020-2023 Google LLC
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

#include "uvkc/benchmark/data_type_util.h"

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace uvkc::benchmark {

template <typename To, typename From>
static To bitcast(From x) {
  static_assert(std::is_trivially_copyable_v<From>);
  static_assert(std::is_trivially_copy_constructible_v<To>);
  static_assert(sizeof(From) == sizeof(To));
  To result{};
  memcpy(&result, &x, sizeof(result));
  return result;
}

void fp16::fromFloat(float x) {
  auto asInt = bitcast<uint32_t>(x);
  int sign = (asInt & 0x80000000) >> 31;
  int exp = ((asInt & 0x7f800000) >> 23) - 127 + 15;
  int mantissa = (asInt & 0x7FFFFF);
  if (exp > 31) exp = 31;
  if (exp < 0) exp = 0;
  sign = sign << 15;
  exp = exp << 10;
  mantissa = mantissa >> (23 - 10);
  asInt = sign | exp | mantissa;
  value_ = static_cast<uint16_t>(asInt);
}

float fp16::toFloat() const {
  auto asInt = static_cast<uint32_t>(value_);
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
  return bitcast<float>(asInt);
}

}  // namespace uvkc::benchmark
