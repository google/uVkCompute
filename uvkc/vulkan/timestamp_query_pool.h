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

#ifndef UVKC_VULKAN_TIMESTAMP_QUERY_POOL_H_
#define UVKC_VULKAN_TIMESTAMP_QUERY_POOL_H_

#include <vulkan/vulkan.h>

#include <memory>

#include "absl/status/statusor.h"
#include "uvkc/vulkan/dynamic_symbols.h"

namespace uvkc {
namespace vulkan {

// A class representing a Vulkan query pool for timestamps.
class TimestampQueryPool {
 public:
  static absl::StatusOr<std::unique_ptr<TimestampQueryPool>> Create(
      VkDevice device, uint32_t valid_timestamp_bits,
      uint32_t nanoseconds_per_timestamp_value, uint32_t query_count,
      const DynamicSymbols &symbols);

  ~TimestampQueryPool();

  VkQueryPool query_pool() const { return query_pool_; }
  uint32_t query_count() const { return query_count_; }

  // Calculates the number of seconds elapsed between the query with index
  // |start| and |end|.
  absl::StatusOr<double> CalculateElapsedSecondsBetween(int start, int end);

 private:
  TimestampQueryPool(VkDevice device, VkQueryPool pool,
                     uint32_t nanoseconds_per_timestamp_value,
                     uint32_t query_count, const DynamicSymbols &symbols);

  VkQueryPool query_pool_;

  VkDevice device_;

  uint32_t nanoseconds_per_timestamp_value_;
  uint32_t query_count_;

  const DynamicSymbols &symbols_;
};

}  // namespace vulkan
}  // namespace uvkc

#endif  // UVKC_VULKAN_TIMESTAMP_QUERY_POOL_H_
