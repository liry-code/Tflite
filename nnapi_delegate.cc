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

#include "/home/tclxa/TfLite/nnapi_delegate.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
// #include "tensorflow/contrib/lite/builtin_op_data.h"
#include "/home/tclxa/TfLite/error_reporter.h"
#include "/home/tclxa/TfLite/model.h"
// #include "tensorflow/contrib/lite/nnapi/NeuralNetworksShim.h"

#ifdef __ANDROID__
#include <android/log.h>
#include <sys/system_properties.h>
#endif

namespace tflite {


#define CHECK_NN(x)                                                     \
  if (x != ANEURALNETWORKS_NO_ERROR) {                                  \
    FATAL("Aborting since NNAPI returned failure nnapi_delegate.cc:%d", \
          __LINE__);                                                    \
  }

}  // namespace tflite
