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
#ifndef TENSORFLOW_CONTRIB_LITE_NNAPI_DELEGATE_H_
#define TENSORFLOW_CONTRIB_LITE_NNAPI_DELEGATE_H_

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "/home/tclxa/TfLite/allocation.h"
#include "/home/tclxa/TfLite/context.h"
#include "/home/tclxa/TfLite/error_reporter.h"
#include "/home/tclxa/TfLite/NeuralNetworksShim.h"
// #include "/home/tclxa/TfLite/interpreter.h"

class ANeuralNetworksModel;
class ANeuralNetworksMemory;
class ANeuralNetworksCompilation;

namespace tflite {

void logError(const char* format, ...) {
  // stderr is convenient for native tests, but is not captured for apps
  va_list args_for_stderr;
  va_start(args_for_stderr, format);
  vfprintf(stderr, format, args_for_stderr);
  va_end(args_for_stderr);
  fprintf(stderr, "\n");
  fflush(stderr);
#ifdef __ANDROID__
  // produce logcat output for general consumption
  va_list args_for_log;
  va_start(args_for_log, format);
  __android_log_vprint(ANDROID_LOG_ERROR, "tflite", format, args_for_log);
  va_end(args_for_log);
#endif
}

#define FATAL(...)       \
  logError(__VA_ARGS__); \
  exit(1);


#define CHECK_NN(x)                                                     \
  if (x != ANEURALNETWORKS_NO_ERROR) {                                  \
    FATAL("Aborting since NNAPI returned failure nnapi_delegate.cc:%d", \
          __LINE__);                                                    \
  }

class NNAPIAllocation : public tflite::MMAPAllocation {
 public:
  NNAPIAllocation(const char* filename, ErrorReporter* error_reporter)
  : MMAPAllocation(filename, error_reporter){
  if (mmapped_buffer_ != MAP_FAILED)
    CHECK_NN(ANeuralNetworksMemory_createFromFd(buffer_size_bytes_, PROT_READ,
                                                mmap_fd_, 0, &handle_));
}
//   ~NNAPIAllocation();

//   size_t offset(const void* ptr) const {
//     auto signed_offset = reinterpret_cast<const uint8_t*>(ptr) -
//                          reinterpret_cast<const uint8_t*>(mmapped_buffer_);

//     return static_cast<size_t>(signed_offset);
//   }

//   ANeuralNetworksMemory* memory() const { return handle_; }
//   bool valid() const override { return handle_ != nullptr; }

 private:
  mutable ANeuralNetworksMemory* handle_ = nullptr;
};

class NNAPIDelegate {
 public:
//   ~NNAPIDelegate();

//   // Convert a tflite graph to NNAPI
//   TfLiteStatus BuildGraph(Interpreter* interpreter);

//   // Run
//   TfLiteStatus Invoke(Interpreter* interpreter);

//   // Whether the current platform supports NNAPI delegation.
  static bool IsSupported(){ return NNAPIExists(); }

//  private:
//   // The NN API model handle
//   ANeuralNetworksModel* nn_model_ = nullptr;
//   // The NN API compilation handle
//   ANeuralNetworksCompilation* nn_compiled_model_ = nullptr;
//   // Model status
//   TfLiteStatus model_status_ = kTfLiteOk;

//   // List of state tensors for LSTM, RNN, SVDF.
//   // NN API does not allow ops to maintain states across multiple
//   // invocations. We need to manually create state input tensors from
//   // corresponding state output tensors of TFLite operations, and map them
//   // correctly.
//   std::vector<int> model_states_inputs_;   // holds NNAPI operand ids
//   std::vector<int> model_states_outputs_;  // holds TFLite tensor ids
};

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_NNAPI_DELEGATE_H_
