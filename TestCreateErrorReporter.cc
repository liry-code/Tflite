#include <iostream>
#include "/home/tclxa/TfLite/nativeinterpreterwrapper_jni.h"

using namespace std;

// test
int main(){
  cout << "start..." << endl;
  // createErrorReporter();
  string model_file = "mobilenet_quant_v1_224.tflite";
  // createModel(model_file);
  createModel(model_file);
  cout << "end...." << endl;
  return 0;
}
