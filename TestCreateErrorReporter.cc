#include <iostream>
#include "/home/tclxa/TfLite/nativeinterpreterwrapper_jni.h"

using namespace std;

// test
int main(){
  cout << "start..." << endl;
  string model_file = "/home/tclxa/TCL/TV_ScenceDetection/tmp/model.tflite";
  // createErrorReporter();
  // BufferErrorReporter* errorReporter = createErrorReporter();
  
  // createModel(model_file)
  // std::unique_ptr<tflite::FlatBufferModel> model = createModel(errorReporter, model_file);
  int num_threads = 1;
  createInterpreter(model_file, num_threads);

  cout << "end...." << endl;
  return 0;
}
