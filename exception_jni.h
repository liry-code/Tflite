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

#ifndef EXCEPTION_JNI_H_
#define EXCEPTION_JNI_H_

#include <iostream>
#include <string>
#include "error_reporter.h"

// extern "C"
using namespace std;

// void throwException(JNIEnv* env, const char* clazz, const char* fmt, ...);
/**LiYu*/
// delete throwException function


class BufferErrorReporter: public tflite::ErrorReporter{
 public:
    // BufferErrorReporter(int limit);
    /** LiYu*/
    BufferErrorReporter(string exceptInfo){cout << exceptInfo << endl;}; 

    // virtual ~BufferErrorReporter();
    int Report(const char* format, va_list args) override {
        int size = 0;
        if (start_idx_ < end_idx_) {
            size = vsnprintf(buffer_ + start_idx_, end_idx_ - start_idx_, format, args);
          }
        start_idx_ += size;
        return size;
    }

     ~BufferErrorReporter(){}
private:    
    // const char* CachedErrorMessage();
    // void outInfo();
    string exceptInfo;
    char* buffer_;
    int start_idx_ = 0;
    int end_idx_ = 0;
};

#endif  // EXCEPTION_JNI_H_
