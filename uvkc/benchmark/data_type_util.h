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

#include <cstddef>
#include <cstdint>

namespace uvkc::benchmark {
class fp16;

enum class DataType {
  fp16,
  fp32,
  i8,
};

template <DataType precision>
struct DataTypeTraits {
  // Specialize this trait to support new data types.
  using storage_type = void;
  using runtime_type = void;
};

template <>
struct DataTypeTraits<DataType::fp16> {
  using storage_type = uint16_t;
  using runtime_type = fp16;
};

template <>
struct DataTypeTraits<DataType::fp32> {
  using storage_type = float;
  using runtime_type = float;
};

template <>
struct DataTypeTraits<DataType::i8> {
  using storage_type = int8_t;
  using runtime_type = int8_t;
};

constexpr std::size_t GetSize(DataType data_type) {
  switch (data_type) {
    case DataType::fp16:
      return sizeof(DataTypeTraits<DataType::fp16>::storage_type);
    case DataType::fp32:
      return sizeof(DataTypeTraits<DataType::fp32>::storage_type);
    case DataType::i8:
      return sizeof(DataTypeTraits<DataType::i8>::storage_type);
  }
}

// Class to emulate half float on CPU.
class fp16 {
 public:
  fp16(uint16_t v) : value_(v) {}
  fp16(float x) { fromFloat(x); }

  fp16 &operator=(const float &x) {
    fromFloat(x);
    return *this;
  }
  fp16 &operator=(const int &x) {
    fromFloat(static_cast<float>(x));
    return *this;
  }
  fp16 &operator+=(const fp16 &x) {
    fromFloat(toFloat() + x.toFloat());
    return *this;
  }
  operator float() const { return toFloat(); }

  void fromFloat(float x);
  float toFloat() const;
  uint16_t getValue() const { return value_; }

 private:
  uint16_t value_;
};

}  // namespace uvkc::benchmark
