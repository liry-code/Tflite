#ifndef NATIVEINTERPRETERWRAPPER_JNI_H_
#define NATIVEINTERPRETERWRAPPER_JNI_H_

#include <iostream>
#include <string>
#include <stdio.h>
#include <time.h>
#include <vector>
#include <stdarg.h>
#include <stdlib.h>
#include "/home/tclxa/TfLite/model.h"
#include "/home/tclxa/TfLite/exception_jni.h"
#include "/home/tclxa/TfLite/context.h"
#include "/home/tclxa/TfLite/JNIFlatBufferVerifier.h"
// #include "tensorflow/contrib/lite/interpreter.h"
// #include "tensorflow/contrib/lite/java/src/main/native/tensor_jni.h"


using namespace std;


/** LiYu*/
BufferErrorReporter* createErrorReporter() {
  string errorInfo = "BufferErrorReporter class exception ";
  // buffer->outInfo();
  BufferErrorReporter* error_reporter = new BufferErrorReporter(errorInfo);
  // BufferErrorReporter* error_reporter = buffer;
  cout << "createErrorReporter function" << endl;
  return error_reporter;
}



// std::unique_ptr<FlatBufferModel> createModel(string model_file) {
//   BufferErrorReporter* error_reporter =
//       createErrorReporter();
//   if (error_reporter == nullptr) return 0;
//   const char* path = const_cast<char*>(model_file.c_str());

//   std::unique_ptr<tflite::TfLiteVerifier> verifier;
//   verifier.reset(new JNIFlatBufferVerifier());

//   auto model = tflite::FlatBufferModel::VerifyAndBuildFromFile(
//       path, verifier.get(), error_reporter);
//   if (!model) {
//     // throwException(env, kIllegalArgumentException,
//     //                "Contents of %s does not encode a valid "
//     //                "TensorFlowLite model: %s",
//     //                path, error_reporter->CachedErrorMessage());

//     cout << "create model faild" << endl;
//     //env->ReleaseStringUTFChars(model_file, path);
//     return 0;
//   }
//   //env->ReleaseStringUTFChars(model_file, path);
//   //return reinterpret_cast<jlong>(model.release());
//   return model;
// }


/**LiYu*/
std::unique_ptr<tflite::FlatBufferModel> createModel(string model_file) {

  BufferErrorReporter* error_reporter =
      createErrorReporter();
  if (error_reporter == nullptr) return 0;
  const char* path = const_cast<char*>(model_file.c_str());

// 报错
  // std::unique_ptr<tflite::TfLiteVerifier> verifier;
  std::unique_ptr<JNIFlatBufferVerifier> verifier;
  JNIFlatBufferVerifier* bufferVerifier = new JNIFlatBufferVerifier;
  verifier.reset(bufferVerifier);

  cout << "createModel over ... " << endl;

  // auto model = tflite::FlatBufferModel::VerifyAndBuildFromFile(
  //     path, verifier.get(), error_reporter);
  // if (!model) {
  //   // throwException(env, kIllegalArgumentException,
  //   //                "Contents of %s does not encode a valid "
  //   //                "TensorFlowLite model: %s",
  //   //                path, error_reporter->CachedErrorMessage());

  //   cout << "create model faild" << endl;
  //   //env->ReleaseStringUTFChars(model_file, path);
  //   return 0;
  // }
  //env->ReleaseStringUTFChars(model_file, path);
  //return reinterpret_cast<jlong>(model.release());
  return nullptr;
}


/**LiYu*/
// // Verifies whether the model is a flatbuffer file.
// class JNIFlatBufferVerifier : tflite::TfLiteVerifier {
//  public:
//   // bool Verify(const char* data, int length,
//   //             tflite::ErrorReporter* reporter) override {
//   //   if (!VerifyModel(data, length)) {
//   //     reporter->Report("The model is not a valid Flatbuffer file");
//   //     return false;
//   //   }
//   //   return true;
//   // }
// };


#endif /* NATIVEINTERPRETERWRAPPER_JNI_H_ */
