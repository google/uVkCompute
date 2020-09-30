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

#include <iostream>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "examples/common.h"  // from SPIRV-Reflect
#include "spirv_reflect.h"    // from SPIRV-Reflect
#include "uvkc/base/file.h"

ABSL_FLAG(std::string, input_file, "", "Input SPIR-V binary module");
ABSL_FLAG(std::string, output_file, "",
          "Output file for descriptor set layout");

static absl::Status GenerateDescriptorSetLayout(
    const std::string &input_file, const std::string &output_file) {
  auto status_or = uvkc::ReadFile(input_file);
  if (!status_or.ok()) return status_or.status();
  std::string shader_code = std::move(status_or.value());

  SpvReflectShaderModule module = {};
  SpvReflectResult result = spvReflectCreateShaderModule(
      shader_code.size(),
      reinterpret_cast<const uint32_t *>(shader_code.data()), &module);
  if (result != SPV_REFLECT_RESULT_SUCCESS) {
    return absl::InternalError("failed to reflect on SPIR-V binary module");
  }

  uint32_t count = 0;
  result = spvReflectEnumerateDescriptorSets(&module, &count, nullptr);
  if (result != SPV_REFLECT_RESULT_SUCCESS) {
    return absl::InternalError("failed to enumerate descriptor sets");
  }

  std::vector<SpvReflectDescriptorSet *> sets(count);
  result = spvReflectEnumerateDescriptorSets(&module, &count, sets.data());
  if (result != SPV_REFLECT_RESULT_SUCCESS) {
    return absl::InternalError("failed to enumerate descriptor sets");
  }

  const char *t = "  ";
  const char *tt = "    ";

  PrintModuleInfo(std::cout, module);
  std::cout << "\n\n";

  std::cout << "Descriptor sets:"
            << "\n";
  for (size_t index = 0; index < sets.size(); ++index) {
    auto p_set = sets[index];

    std::cout << t << index << ":\n";
    PrintDescriptorSet(std::cout, *p_set, tt);
    std::cout << "\n\n";
  }

  spvReflectDestroyShaderModule(&module);

  return absl::OkStatus();
}

int main(int argc, char **argv) {
  absl::SetProgramUsageMessage(absl::StrCat(
      "Read SPIR-V binary module and generate descriptor set layout. "
      "Example:\n  ",
      argv[0], "-input_file=/path/to/spirv-blob ",
      "-output_file=/path/to/descriptor/code"));

  absl::ParseCommandLine(argc, argv);

  auto status = GenerateDescriptorSetLayout(absl::GetFlag(FLAGS_input_file),
                                            absl::GetFlag(FLAGS_output_file));
  if (!status.ok()) std::cerr << status;

  return 0;
}
