#ifndef NATIVEINTERPRETERWRAPPER_JNI_H_
#define NATIVEINTERPRETERWRAPPER_JNI_H_

#include <iostream>
#include <string>
#include <stdio.h>
#include <time.h>
#include <vector>
#include <stdarg.h>
#include <stdlib.h>
#include <memory>
#include "/home/tclxa/TfLite/model.h"
#include "/home/tclxa/TfLite/exception_jni.h"
#include "/home/tclxa/TfLite/error_reporter.h"
#include "/home/tclxa/TfLite/interpreter.h"
// #include "tensorflow/contrib/lite/java/src/main/native/tensor_jni.h"

using namespace std;

namespace tflite{

} // namespace

/**LiYu*/
// Verifies whether the model is a flatbuffer file.
class JNIFlatBufferVerifier: public tflite::TfLiteVerifier {
 public:
  bool Verify(const char* data, int length,
    tflite::ErrorReporter* reporter)
    override {
   if (!VerifyModel(data, length)) {
     reporter->Report("The model is not a valid Flatbuffer file");
     return false;
   }
    cout << "nativeinterpreterwrapper_jni.h Verifier ..." << endl;
    return true;
  }

  // TODO(yichengfan): evaluate the benefit to use tflite verifier.
  bool VerifyModel(const void* buf, size_t len){
  flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buf), len);
  return tflite::VerifyModelBuffer(verifier);
}

  ~JNIFlatBufferVerifier(){}
};

/** LiYu*/
BufferErrorReporter* createErrorReporter() {
  string errorInfo = "BufferErrorReporter class exception ";
  // buffer->outInfo();
  BufferErrorReporter* error_reporter = new BufferErrorReporter(errorInfo);
  // BufferErrorReporter* error_reporter = buffer;
  cout << "createErrorReporter function" << endl;
  return error_reporter;
}


/*LiYu*/
std::unique_ptr<tflite::FlatBufferModel> createModel(BufferErrorReporter* error_reporter,
  string model_file){ 
  if (error_reporter == nullptr){
    cout << "error_reporter nullptr" << endl;
    return 0;
  }
  const char* path = const_cast<char*>(model_file.c_str());
  std::unique_ptr<tflite::TfLiteVerifier> verifier;
  verifier.reset(new JNIFlatBufferVerifier());
  tflite::FlatBufferModel* flatBuffer;
  auto model = flatBuffer->VerifyAndBuildFromFile(path,verifier.get(),error_reporter);

  if (!model) {
    cout << "create model faild" << endl;
    return 0;
  }
  cout << "createModel over ... " << endl;
  return model;
}

/*LiYu*/
std::unique_ptr<tflite::Interpreter> createInterpreter(string model_file, 
  int num_threads) {
  // tflite::FlatBufferModel *model = dynamic_cast<tflite::FlatBufferModel *>(bufferModel.get());
  // tflite::FlatBufferModel *model = dynamic_cast<tflite::FlatBufferModel *>(bufferModel.release());
 
  BufferErrorReporter* error_reporter = createErrorReporter();
  if (error_reporter == nullptr) return 0;
  std::unique_ptr<tflite::FlatBufferModel> model = createModel(error_reporter, model_file);
  if (model == nullptr) return 0;
  // auto resolver = ::tflite::CreateOpResolver();
  // std::unique_ptr<tflite::Interpreter> interpreter;
  // TfLiteStatus status = tflite::InterpreterBuilder(*model, *(resolver.get()))(
  //     &interpreter, static_cast<int>(num_threads));
  // if (status != kTfLiteOk) {
  //   throwException(env, kIllegalArgumentException,
  //                  "Internal error: Cannot create interpreter: %s",
  //                  error_reporter->CachedErrorMessage());
  //   return 0;
  // }
  // // allocates memory
  // status = interpreter->AllocateTensors();
  // if (status != kTfLiteOk) {
  //   throwException(
  //       env, kIllegalStateException,
  //       "Internal error: Unexpected failure when preparing tensor allocations:"
  //       " %s",
  //       error_reporter->CachedErrorMessage());
  //   return 0;
  // }
  // return interpreter;
  cout << "create interpreter ..." << endl;
  return NULL;
}


void* run(long interpreter_handle, long error_handle,
    /*jobjectArray sizes, jintArray data_types,*/ int nums_of_bytes[],
    /*jobjectArray values, jobject wrapper,*/ bool memory_allocated) {

  tflite::Interpreter* interpreter = nullptr;
  //     convertLongToInterpreter(env, interpreter_handle);
  // if (interpreter == nullptr) return nullptr;

  BufferErrorReporter* error_reporter = createErrorReporter();
  if (error_reporter == nullptr) return nullptr;

  // const int input_size = env->GetArrayLength(sizes);
  // // validates inputs
  // TfLiteStatus status = checkInputs(env, interpreter, input_size, data_types,
  //                                   nums_of_bytes, values, sizes);
  // if (status != kTfLiteOk) return nullptr;
  // if (!memory_allocated ||
  //     !areInputDimensionsTheSame(env, interpreter, input_size, sizes)) {
  //   // resizes inputs
  //   status = resizeInputs(env, interpreter, input_size, sizes);
  //   if (status != kTfLiteOk) {
  //     throwException(env, kNullPointerException, "Can not resize the input: %s",
  //                    error_reporter->CachedErrorMessage());
  //     return nullptr;
  //   }
  //   // allocates memory
  //   status = interpreter->AllocateTensors();
  //   if (status != kTfLiteOk) {
  //     throwException(env, kNullPointerException,
  //                    "Can not allocate memory for the given inputs: %s",
  //                    error_reporter->CachedErrorMessage());
  //     return nullptr;
  //   }
  // }
  // // sets inputs
  // status = setInputs(env, interpreter, input_size, data_types, nums_of_bytes,
  //                    values);
  // if (status != kTfLiteOk) return nullptr;
  // timespec beforeInference = ::tflite::getCurrentTime();
  // // runs inference
  // if (interpreter->Invoke() != kTfLiteOk) {
  //   throwException(env, kIllegalArgumentException,
  //                  "Failed to run on the given Interpreter: %s",
  //                  error_reporter->CachedErrorMessage());
  //   return nullptr;
  // }
  // timespec afterInference = ::tflite::getCurrentTime();
  // jclass wrapper_clazz = env->GetObjectClass(wrapper);
  // jfieldID fid =
  //     env->GetFieldID(wrapper_clazz, "inferenceDurationNanoseconds", "J");
  // if (env->ExceptionCheck()) {
  //   env->ExceptionClear();
  // } else if (fid != nullptr) {
  //   env->SetLongField(
  //       wrapper, fid,
  //       ::tflite::timespec_diff_nanoseconds(&beforeInference, &afterInference));
  // }
  // // returns outputs
  // const std::vector<int>& results = interpreter->outputs();
  // if (results.empty()) {
  //   throwException(env, kIllegalArgumentException,
  //                  "The Interpreter does not have any outputs.");
  //   return nullptr;
  // }
  // jlongArray outputs = env->NewLongArray(results.size());
  // size_t size = results.size();
  // for (int i = 0; i < size; ++i) {
  //   TfLiteTensor* source = interpreter->tensor(results[i]);
  //   jlong output = reinterpret_cast<jlong>(source);
  //   env->SetLongArrayRegion(outputs, i, 1, &output);
  // }
  // return outputs;
}


int* getInputDims(){
  //   JNIEnv* env, jclass clazz, jlong handle, jint input_idx, jint num_bytes) {
  // tflite::Interpreter* interpreter = convertLongToInterpreter(env, handle);
  // if (interpreter == nullptr) return nullptr;
  // const int idx = static_cast<int>(input_idx);
  // if (input_idx < 0 || input_idx >= interpreter->inputs().size()) {
  //   throwException(env, kIllegalArgumentException,
  //                  "Out of range: Failed to get %d-th input out of %d inputs",
  //                  input_idx, interpreter->inputs().size());
  //   return nullptr;
  // }
  // TfLiteTensor* target = interpreter->tensor(interpreter->inputs()[idx]);
  // int size = target->dims->size;
  // if (num_bytes >= 0) {  // verifies num of bytes matches if num_bytes if valid.
  //   int expected_num_bytes = elementByteSize(target->type);
  //   for (int i = 0; i < size; ++i) {
  //     expected_num_bytes *= target->dims->data[i];
  //   }
  //   if (num_bytes != expected_num_bytes) {
  //     throwException(env, kIllegalArgumentException,
  //                    "Failed to get input dimensions. %d-th input should have"
  //                    " %d bytes, but found %d bytes.",
  //                    idx, expected_num_bytes, num_bytes);
  //     return nullptr;
  //   }
  // }
  // jintArray outputs = env->NewIntArray(size);
  // env->SetIntArrayRegion(outputs, 0, size, &(target->dims->data[0]));
  // return outputs;
  return nullptr;
}

#endif /* NATIVEINTERPRETERWRAPPER_JNI_H_ */
