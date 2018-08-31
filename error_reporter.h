/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CONTRIB_LITE_ERROR_REPORTER_H_
#define TENSORFLOW_CONTRIB_LITE_ERROR_REPORTER_H_

#include <iostream>
#include <cstdarg>
#include "/home/tclxa/TfLite/context.h"
using namespace std;


namespace tflite {



class ErrorReporter {
 public:
	// virtual ~ErrorReporter();

	/**LiYu*/
	virtual int Report(const char* format, va_list args) = 0;
	int Report(const char* format, ...){
		va_list args;
		va_start(args, format);
		int code = Report(format, args);
		va_end(args);
		cout << "error_reporter Report function ...." << endl;
		return code;
	};

	int ReportError(void*, const char* format, ...);
};

// An error reporter that simplify writes the message to stderr.
struct StderrReporter : public ErrorReporter {
  int Report(const char* format, va_list args){
  	#ifdef __ANDROID__
    // On Android stderr is not captured for applications, only for code run from
    // the shell. Rather than assume all users will set up a custom error
    // reporter, let's output to logcat here
    va_list args_for_log;
    va_copy(args_for_log, args);
    __android_log_vprint(ANDROID_LOG_ERROR, "tflite", format, args_for_log);
    va_end(args_for_log);
  #endif
    const int result = vfprintf(stderr, format, args);
    fputc('\n', stderr);
  return result;
  }
};

// Return the default error reporter (output to stderr).
ErrorReporter* DefaultErrorReporter() {
	static StderrReporter* error_reporter = new StderrReporter;
	return error_reporter;
}

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_ERROR_REPORTER_H_
