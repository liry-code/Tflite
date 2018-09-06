#include <iostream>
#include "/home/tclxa/TfLite/nativeinterpreterwrapper_jni.h"
#include <ctime>
#include <map>
#include "context.h"
#include <ext/hash_map>

#define DIX_WIDTH 112
#define DIX_DEPTH 112
#define DIX_CHANAL 3

// long inferenceDurationNanoseconds = -1;

// Tensor getInputTensor(int index) {
//   if (index < 0 || index >= inputTensors.length) {
//     throw new IllegalArgumentException("Invalid input Tensor index: " + index);
//   }
//   Tensor inputTensor = inputTensors[index];
//   if (inputTensor == null) {
//     inputTensor =
//         inputTensors[index] = Tensor.fromHandle(getInputTensor(interpreterHandle, index));
//   }
//   return inputTensor;
// }




void modelCompute(float inputs[1][DIX_WIDTH][DIX_DEPTH][DIX_CHANAL], map<int, float> outputs){
  int length = sizeof(inputs) / sizeof(inputs[0]);
  // 存放数据类型
  int* dataTypes = new int[length];
  // 将多维数组存为多个一维数组
  void* sizes = new int[length];
  int* numsOfBytes = new int[length];

  cout << "length : "<<length << endl;

  for (int i = 0; i < length; ++i) {
    int[] dims = shapeOf(inputs[i]);
    // sizes[i] = inputs;
    // numsOfBytes[i] = dataType.elemByteSize() * numElements(dims);
  } 
    // else {
    //   throw new IllegalArgumentException(
    //       String.format(
    //           "%d-th element of the %d inputs is not an array or a ByteBuffer.",
    //           i, inputs.length));
    // }
}


int* shapeOf(int ) {
  int size = 3;
  int* dimensions = new int[size];
  // fillShape(o, 0, dimensions);
  return dimensions;
}

// int[] shapeOf(float inputs[DIX_WIDTH][DIX_DEPTH][DIX_CHANAL]){

//   return NULL;
// }




// test
int main(){
  cout << "start..." << endl;
  time_t startTime = clock();

  string model_file = "/home/tclxa/TCL/NLDF_Bokeh/tmp/mobilenet_quant_v1_224.tflite";
  // createErrorReporter();
  BufferErrorReporter* errorReporter = createErrorReporter();
  
  // createModel(model_file)
  std::unique_ptr<tflite::FlatBufferModel> model = createModel(errorReporter, model_file);
  int num_threads = 1;
  // createInterpreter(model_file, num_threads);
  tflite::Interpreter* interpreterHandle = nullptr;

  // 将输入输出数据存放在Map
  // char imgData[1000] = "dscsd";
  float imgData[1][DIX_WIDTH][DIX_DEPTH][DIX_CHANAL] = {1,2,3,4,5,6,7,8,9,10};
 
  float output[56*56*2] = {20.2,33.5,33.4};
  
  map<int, float> outputs;
  outputs[0] = 2001.0;

  modelCompute(imgData, outputs);

  time_t endTime = clock();

  cout << endTime-startTime << " ms" << endl;
  cout << "end...." << endl;
  return 0;
}

