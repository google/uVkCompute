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

#ifndef UVKC_LOG_H_
#define UVKC_LOG_H_

#include <ostream>

namespace uvkc {

class Logger;

// Returns the logger that discards all messages.
Logger &GetNullLogger();

// Returns the logger that writes messages to std::clog.
Logger &GetErrorLogger();

// A simple logger that writes messages to an output stream if not null.
//
// This logger uses standard I/O so it should only be used in binaries.
class Logger {
 public:
  friend Logger &GetNullLogger();
  friend Logger &GetErrorLogger();

  template <class T>
  Logger &operator<<(const T &content);

 private:
  explicit Logger(std::ostream *stream) : stream_(stream) {}

  // Disable copy construction and assignment
  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  std::ostream *stream_;
};

template <class T>
Logger &Logger::operator<<(const T &content) {
  if (stream_) *stream_ << content;
  return *this;
}

}  // namespace uvkc

#endif  // UVKC_LOG_H_
