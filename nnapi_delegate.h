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
#ifndef NNAPI_DELEGATE_H_
#define NNAPI_DELEGATE_H_

#include "allocation.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "error_reporter.h"
#include "NeuralNetworksShim.h"
// #include "/home/tclxa/TfLite/interpreter.h"

class ANeuralNetworksModel;
class ANeuralNetworksMemory;
class ANeuralNetworksCompilation;


namespace tflite{

class NNAPIAllocation : public tflite::MMAPAllocation {
  public:
    NNAPIAllocation(const char* filename, ErrorReporter* error_reporter);
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
//   ~NNAPIAllocation();

//   size_t offset(const void* ptr) const {
//     auto signed_offset = reinterpret_cast<const uint8_t*>(ptr) -
//                          reinterpret_cast<const uint8_t*>(mmapped_buffer_);

//     return static_cast<size_t>(signed_offset);
//   }

//   ANeuralNetworksMemory* memory() const { return handle_; }
//   bool valid() const override { return handle_ != nullptr; }

 private:
  ANeuralNetworksModel* nn_model_ = nullptr;
  // The NN API compilation handle
  ANeuralNetworksCompilation* nn_compiled_model_ = nullptr;
};


} //namespace

#endif  // TENSORFLOW_CONTRIB_LITE_NNAPI_DELEGATE_H_
