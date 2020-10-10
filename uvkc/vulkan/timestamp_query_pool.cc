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

#include "uvkc/vulkan/timestamp_query_pool.h"

#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "uvkc/vulkan/status_util.h"

namespace uvkc {
namespace vulkan {

// static
absl::StatusOr<std::unique_ptr<TimestampQueryPool>> TimestampQueryPool::Create(
    VkDevice device, uint32_t valid_timestamp_bits,
    uint32_t nanoseconds_per_timestamp_value, uint32_t query_count,
    const DynamicSymbols &symbols) {
  if (valid_timestamp_bits == 0) {
    return absl::UnavailableError("the device does not support timestamp");
  }

  // Create query pool.
  VkQueryPoolCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  create_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  create_info.queryCount = query_count;
  create_info.pipelineStatistics = 0;

  VkQueryPool query_pool = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(symbols.vkCreateQueryPool(device, &create_info,
                                               /*pAllocator=*/nullptr,
                                               &query_pool));
  return absl::WrapUnique(new TimestampQueryPool(
      device, query_pool, nanoseconds_per_timestamp_value, query_count,
      symbols));
}

TimestampQueryPool::~TimestampQueryPool() {
  symbols_.vkDestroyQueryPool(device_, query_pool_, /*pAllocator=*/nullptr);
}

absl::StatusOr<double> TimestampQueryPool::CalculateElapsedSecondsBetween(
    int start, int end) {
  if (end <= start) {
    return absl::InvalidArgumentError(
        "end index must be greater than start index");
  }

  uint32_t count = end - start + 1;
  std::vector<uint64_t> timestamps(count);
  VK_RETURN_IF_ERROR(symbols_.vkGetQueryPoolResults(
      device_, query_pool_, start, count, /*dataSize=*/count * sizeof(uint64_t),
      /*pData=*/reinterpret_cast<void *>(timestamps.data()),
      /*stride=*/sizeof(uint64_t),
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));

  double seconds = (timestamps.back() - timestamps.front()) *
                   nanoseconds_per_timestamp_value_ * 1e-9;
  return seconds;
}

TimestampQueryPool::TimestampQueryPool(VkDevice device, VkQueryPool pool,
                                       uint32_t nanoseconds_per_timestamp_value,
                                       uint32_t query_count,
                                       const DynamicSymbols &symbols)
    : device_(device),
      query_pool_(pool),
      nanoseconds_per_timestamp_value_(nanoseconds_per_timestamp_value),
      query_count_(query_count),
      symbols_(symbols) {}

}  // namespace vulkan
}  // namespace uvkc
