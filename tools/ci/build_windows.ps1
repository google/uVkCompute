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

function Invoke-CMake {
    param([string]$cmake_path, [string[]]$cmake_params)

    $cmake_params = $cmake_params -join " "
    $process = Start-Process -FilePath $cmake_path -ArgumentList $cmake_params -NoNewWindow -PassThru
    # HACK: Start-Process is broken... wow.
    # https://stackoverflow.com/questions/10262231/obtaining-exitcode-using-start-process-and-waitforexit-instead-of-wait
    $handle = $process.Handle
    $exitcode = 1
    $timeout_millis = 10 * 60 * 1000 # 10 minutes
    if ($process.WaitForExit($timeout_millis) -eq $false) {
        Write-Error "cmake timed out after $timeout_millis millis"
    }
    else {
        $exitcode = $process.ExitCode
        Write-Debug "cmake returned in time with exit code $($exitcode)"
    }

    if ($process.ExitCode -ne 0) {
        Write-Debug "cmake exited with $exitcode"
        exit $exitcode
    }
}

if ($env:CMAKE_BIN -ne $null) {
    $cmake_path = $env:CMAKE_BIN
}
else {
    # The default install path for CMake
    $cmake_path = "C:/Program Files/CMake/bin/cmake.exe"
}

Write-Host -ForegroundColor Yellow "Using cmake executable: $cmake_path"
Invoke-CMake $cmake_path "-G", '"Visual Studio 16 2019"', "-A", "x64", "-S", "./", "-B", "build-windows"
Invoke-CMake $cmake_path "--build", "build-windows", "-j", "4"
