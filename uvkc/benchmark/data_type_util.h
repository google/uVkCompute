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
#include <ostream>
#include <utility>

namespace uvkc::benchmark {
class fp16;

enum class DataType {
  fp16,
  fp32,
  i8,
  i32,
};

template <DataType DT>
struct DataTypeTraits {
  // Specialize this trait to support new data types.
  using storage_type = void;
  using runtime_type = void;
  static constexpr char name[] = "";
};

template <>
struct DataTypeTraits<DataType::fp16> {
  using storage_type = uint16_t;
  using runtime_type = fp16;
  static constexpr char name[] = "fp16";
};

template <>
struct DataTypeTraits<DataType::fp32> {
  using storage_type = float;
  using runtime_type = float;
  static constexpr char name[] = "fp32";
};

template <>
struct DataTypeTraits<DataType::i8> {
  using storage_type = int8_t;
  using runtime_type = int8_t;
  static constexpr char name[] = "i8";
};

template <>
struct DataTypeTraits<DataType::i32> {
  using storage_type = int32_t;
  using runtime_type = int32_t;
  static constexpr char name[] = "i32";
};

/// Invokes the |fn| functor with a DataTypeTraits object matching |data_type|,
/// followed by the remaining arguments |args|. This is useful when converting
/// runtime data_type back to types available at the compilation time. Compared
/// to ad-hoc switch statements, this helper makes it easier to *statically*
/// make sure that all data types were handled.
template <typename Fn, typename... Args>
constexpr auto InvokeWithTraits(DataType data_type, Fn &&fn, Args &&...args) {
  switch (data_type) {
    case DataType::fp16:
      return fn(DataTypeTraits<DataType::fp16>{}, std::forward<Args>(args)...);
    case DataType::fp32:
      return fn(DataTypeTraits<DataType::fp32>{}, std::forward<Args>(args)...);
    case DataType::i8:
      return fn(DataTypeTraits<DataType::i8>{}, std::forward<Args>(args)...);
    case DataType::i32:
      return fn(DataTypeTraits<DataType::i32>{}, std::forward<Args>(args)...);
  }
}

constexpr std::size_t GetSize(DataType data_type) {
  return InvokeWithTraits(data_type, [](auto traits) {
    return sizeof(typename decltype(traits)::storage_type);
  });
}

constexpr const char *GetName(DataType data_type) {
  return InvokeWithTraits(data_type,
                          [](auto traits) { return decltype(traits)::name; });
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
  fp16 operator*(const fp16 &rhs) const {
    return fp16(toFloat() * rhs.toFloat());
  }
  bool operator==(const fp16 &rhs) const { return value_ == rhs.value_; }

  explicit operator float() const { return toFloat(); }
  explicit operator uint16_t() const { return getValue(); }

  void fromFloat(float x);
  float toFloat() const;
  uint16_t getValue() const { return value_; }

  friend std::ostream &operator<<(std::ostream &os, const fp16 &value) {
    return os << value.toFloat();
  }

 private:
  uint16_t value_;
};

}  // namespace uvkc::benchmark
