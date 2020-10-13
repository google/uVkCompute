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

#include "absl/flags/flag.h"
#include "absl/flags/internal/parse.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"
#include "uvkc/benchmark/dispatch_void_shader.h"
#include "uvkc/benchmark/status_util.h"
#include "uvkc/benchmark/vulkan_context.h"

namespace uvkc {
namespace benchmark {

bool AbslParseFlag(absl::string_view text, LatencyMeasureMode *mode,
                   std::string *error) {
  if (text == "system_submit") {
    *mode = LatencyMeasureMode::kSystemSubmit;
    return true;
  }

  if (text == "system_dispatch") {
    *mode = LatencyMeasureMode::kSystemDispatch;
    return true;
  }
  if (text == "gpu_timestamp") {
    *mode = LatencyMeasureMode::kGpuTimestamp;
    return true;
  }

  *error =
      "unknown value for latency measure mode; supported choices are "
      "'system_submit', 'system_dispatch', 'gpu_timestamp'";
  return false;
}

std::string AbslUnparseFlag(LatencyMeasureMode mode) {
  switch (mode) {
    case LatencyMeasureMode::kSystemSubmit:
      return "system_submit";
    case LatencyMeasureMode::kSystemDispatch:
      return "system_dispatch";
    case LatencyMeasureMode::kGpuTimestamp:
      return "gpu_timestamp";
  }
}

}  // namespace benchmark
}  // namespace uvkc

ABSL_FLAG(uvkc::benchmark::LatencyMeasureMode, latency_measure_mode,
          uvkc::benchmark::LatencyMeasureMode::kSystemSubmit,
          "Latency measure modes");

extern "C" int main(int argc, char **argv) {
  // We use two command-line libraries: Abseil's and Google Benchmark's. Both
  // supports `--help`. To ensure that each is able to parse its own flags, we
  // use an absl "internal" function (still with public visibility) to parse
  // while ignoring undefined flags. If it sees `--help` it will exit here, so
  // we include the benchmark library usage information in the manually-set help
  // output. Then we let Google Benchmark parse its flags. Finally we call the
  // absl::ParseCommandLine again to handle remaining unknown flags and print
  // errors.

  absl::SetProgramUsageMessage(R"(Run Vulkan compute benchmarks
    --latency_measure_mode=[system_submit|system_dispatch|gpu_timestamp]
      * system_submit: time spent from queue submit to returning from queue wait
      * system_dispatch: system_submit subtracted by time for void dispatch
      * gpu_timestamp: timestamp difference measured on GPU

  Optional flags from Google Benchmark library:
    [--benchmark_list_tests={true|false}]
    [--benchmark_filter=<regex>]
    [--benchmark_min_time=<min_time>]
    [--benchmark_repetitions=<num_repetitions>]
    [--benchmark_report_aggregates_only={true|false}]
    [--benchmark_display_aggregates_only={true|false}]
    [--benchmark_format=<console|json|csv>]
    [--benchmark_out=<filename>]
    [--benchmark_out_format=<json|console|csv>]
    [--benchmark_color={auto|true|false}]
    [--benchmark_counters_tabular={true|false}]
    [--v=<verbosity>])");

  absl::flags_internal::ParseCommandLineImpl(
      argc, argv, absl::flags_internal::ArgvListAction::kRemoveParsedArgs,
      absl::flags_internal::UsageFlagsAction::kHandleUsage,
      absl::flags_internal::OnUndefinedFlag::kIgnoreUndefined);
  ::benchmark::Initialize(&argc, argv);
  auto positional_args = absl::ParseCommandLine(argc, argv);
  BM_CHECK_EQ(positional_args.size(), 1)  // argv[0]
      << "cannot accept positional arguments";

  BM_CHECK_OK_AND_ASSIGN(auto context, uvkc::benchmark::CreateVulkanContext());
  auto mode = absl::GetFlag(FLAGS_latency_measure_mode);
  context->latency_measure.mode = mode;

  for (int i = 0; i < context->devices.size(); ++i) {
    const auto &physical_device = context->physical_devices[i];
    auto *device = context->devices[i].get();
    if (mode == uvkc::benchmark::LatencyMeasureMode::kSystemDispatch) {
      // Register the overhead benchmark first to update the overhead latency,
      // which will be used by following benchmarks. Note that we are only
      // **registering** the benchmark here. So relies on the implicit ordering
      // in benchmark execution to make sure the overhead is there when we run
      // following benchmarks. This is true if Google Benchmark, which runs
      // benchmarks at registration order.
      if (!uvkc::benchmark::RegisterVulkanOverheadBenchmark(
              physical_device, device,
              &context->latency_measure.overhead_seconds)) {
        uvkc::benchmark::RegisterDispatchVoidShaderBenchmark(
            physical_device.v10_properties.deviceName, device,
            &context->latency_measure.overhead_seconds);
      }
    }
    uvkc::benchmark::RegisterVulkanBenchmarks(physical_device, device,
                                              &context->latency_measure);
  }

  ::benchmark::RunSpecifiedBenchmarks();
}
