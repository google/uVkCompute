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

#include "uvkc/benchmark/main.h"

#include "benchmark/benchmark.h"
#include "uvkc/benchmark/status_util.h"

extern "C" int main(int argc, char **argv) {
  BM_CHECK_OK_AND_ASSIGN(auto context, uvkc::benchmark::CreateVulkanContext());
  uvkc::benchmark::RegisterVulkanBenchmarks(context.get());
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
