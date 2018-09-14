#ifndef NATIVEINTERPRETERWRAPPER_JNI_H_
#define NATIVEINTERPRETERWRAPPER_JNI_H_

#include <iostream>
#include <string>
#include <stdio.h>
#include <memory>
#include <time.h>
#include <vector>
#include <stdarg.h>
#include <stdlib.h>

#include <exception>
//#include "context.h"
#include "model.h"
#include "exception_jni.h"
#include "error_reporter.h"
#include "interpreter.h"
// #include "tensorflow/contrib/lite/java/src/main/native/tensor_jni.h"



namespace tflite{

} // namespace


/** LiYu*/
BufferErrorReporter* createErrorReporter();

/* LiYu */
std::unique_ptr<tflite::FlatBufferModel> createModel(BufferErrorReporter* error_reporter, string model_file) ;

/* LiYu */
void* run(
    long interpreter_handle, long error_handle,
    std::vector<std::vector<int>> sizes, std::vector<int> dataTypes, 
    std::vector<int> numsOfBytes,
    /*float values[],*/ bool memory_allocated);
#endif /* NATIVEINTERPRETERWRAPPER_JNI_H_ */
