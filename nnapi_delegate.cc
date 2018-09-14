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

#include "nnapi_delegate.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
// #include "tensorflow/contrib/lite/builtin_op_data.h"
#include "error_reporter.h"
#include "model.h"
// #include "tensorflow/contrib/lite/nnapi/NeuralNetworksShim.h"

#ifdef __ANDROID__
#include <android/log.h>
#include <sys/system_properties.h>
#endif


namespace tflite {

// void logError(const char* format, ...) {
//   // stderr is convenient for native tests, but is not captured for apps
//   va_list args_for_stderr;
//   va_start(args_for_stderr, format);
//   vfprintf(stderr, format, args_for_stderr);
//   va_end(args_for_stderr);
//   fprintf(stderr, "\n");
//   fflush(stderr);
// #ifdef __ANDROID__
//   // produce logcat output for general consumption
//   va_list args_for_log;
//   va_start(args_for_log, format);
//   __android_log_vprint(ANDROID_LOG_ERROR, "tflite", format, args_for_log);
//   va_end(args_for_log);
// #endif
// }

void FATAL(const char* format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fflush(stderr);
  exit(1);
}


#define CHECK_NN(x)                                                     \
  if (x != ANEURALNETWORKS_NO_ERROR) {                                  \
    FATAL("Aborting since NNAPI returned failure nnapi_delegate.cc:%d", \
          __LINE__);                                                    \
  }

NNAPIAllocation::NNAPIAllocation(const char* filename, ErrorReporter* error_reporter)
: MMAPAllocation(filename, error_reporter){
	if (mmapped_buffer_ != MAP_FAILED)
	CHECK_NN(ANeuralNetworksMemory_createFromFd(buffer_size_bytes_, PROT_READ,
	                                            mmap_fd_, 0, &handle_));
}


}  // namespace tflite

