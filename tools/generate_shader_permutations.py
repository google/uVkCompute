#!/usr/bin/env python3
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Interates #define choices in the same source shader to generate SPIR-V corpus

This scripts takes a list of #define macros, together with their choices, and
generates all productions of macro choices. Then shader compiler is invoked for
each of them to generate the corresponding SPIR-V code. Finally, all the SPIR-V
code are placed in the same output file.
"""

import argparse
import itertools
import os
import subprocess
import sys


def parse_arguments():
  """Parses command line arguments."""

  parser = argparse.ArgumentParser()

  parser.add_argument(
      "infile",
      metavar="<shader-source-file>",
      type=argparse.FileType("r"),
      help="Input source code file")
  parser.add_argument(
      "-o",
      "--outfile",
      metavar="<spirv-output-file>",
      type=argparse.FileType("w"),
      help="Output SPIR-V code file")
  parser.add_argument(
      "--define",
      metavar="<macro-choices>",
      type=str,
      action="append",
      help="A #define and its choices in the format of 'FOO=[BAR|BARZ]'")
  parser.add_argument(
      "--glslc",
      metavar="<glslc-executable>",
      type=str,
      help="Path to glslc executable")
  parser.add_argument(
      "--glslc-arg",
      metavar="<glslc-arg>",
      type=str,
      action="append",
      help="Additional arguments to pass to glslc")
  parser.add_argument(
      "--verbose",
      action="store_true",
      help="Print in verbose mode")

  args = parser.parse_args()

  if not os.path.isfile(args.glslc) or not os.access(args.glslc, os.X_OK):
    raise parser.error("Invalid glslc executable")

  return args


def parse_define(define):
  """Parses a 'FOO=[FOO1|FOO2]' string into a (FOO, [FOO1, FOO2]) tuple."""
  macro, choices = define.split("=")
  choices = choices.strip("[]").split("|")
  return (macro, choices)


def parse_multi_list(macro):
  """Parses a '{FOO,BAR}' string into a [FOO, BAR] list."""
  return macro.strip("{}").split(",")


def generate_productions(defines):
  """Generates all productions from defines.

  Arguments:
    - defines: an array of 'FOO=[FOO1|FOO2]' or
               '{FOO,BAR}=[{FOO1,BAR1}|{FOO2,BAR2}]' strings.
  """
  defines = [parse_define(d) for d in defines]
  all_macros = [d[0] for d in defines]
  all_choices = [d[1] for d in defines]
  for case in itertools.product(*all_choices):
    unexpanded_macro_choice = list(zip(all_macros, case))
    macro_choice = []
    # Expand ({FOO,BAR}, {FOO1,BAR1}) into ((FOO, FOO1), (BAR, BAR1)).
    for (macro, choice) in unexpanded_macro_choice:
      macros = parse_multi_list(macro)
      choices = parse_multi_list(choice)
      macro_choice.extend(list(zip(macros, choices)))
    var_name = "_".join(["{}_{}".format(m, c) for (m, c) in macro_choice])
    compiler_defines = ["-D{}={}".format(m, c) for (m, c) in macro_choice]
    yield (var_name, compiler_defines)


def main(args):
  # Base command for generating SPIR-V code
  base_code_command = [args.glslc, "-c", "-O", "-fshader-stage=compute",
                       "-mfmt=num", args.infile.name, "-o", "-"]
  if args.glslc_arg:
    base_code_command.extend(args.glslc_arg)
  # Base command for generating SPIR-V assembly
  base_asm_command = [args.glslc, "-S", "-O", "-fshader-stage=compute",
                      args.infile.name, "-o", "-"]
  if args.glslc_arg:
    base_asm_command.extend(args.glslc_arg)
  spirv_variables = []

  for case in generate_productions(args.define):
    var_name = case[0]

    # Generate SPIR-V code
    command = base_code_command
    command.extend(case[1])
    if args.verbose:
      print("glslc command: '{}'".format(" ".join(command)))
    spirv_code = subprocess.check_output(command).decode("ascii")

    # Generate SPIR-V assembly
    command = base_asm_command
    command.extend(case[1])
    if args.verbose:
      print("glslc command: '{}'".format(" ".join(command)))
    spirv_asm = subprocess.check_output(command).decode("ascii")

    spirv_variables.append(
        "static const uint32_t {}[] = {{\n/*\n{}*/\n{}}};\n".format(
            var_name, spirv_asm, spirv_code))

  all_variables = "\n".join(spirv_variables)
  args.outfile.write(all_variables)


if __name__ == "__main__":
  main(parse_arguments())
