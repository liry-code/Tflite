#include <iostream>
#include "/home/tclxa/TfLite/schema_generated.h"
// #include "/home/tclxa/TfLite/exception_jni.h"
#include "/home/tclxa/TfLite/nativeinterpreterwrapper_jni.h"
#include "/home/tclxa/TfLite/model.h"



// #include "exception_jni.h"



JNIFlatBufferVerifier* error = new JNIFlatBufferVerifier();

/**LiYu*/
std::unique_ptr<tflite::FlatBufferModel> createModel(string model_file) ;


/** LiYu*/
BufferErrorReporter* createErrorReporter();