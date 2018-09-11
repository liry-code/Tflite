#include <iostream>
#include <fstream>
#include "nativeinterpreterwrapper_jni.h"
#include <ctime>
#include <map>
#include "context.h"
#include <ext/hash_map>
#include "DataType.h"

#define DIX_WIDTH 112
#define DIX_DEPTH 112
#define DIX_CHANAL 3


int numElements(std::vector<int> &vec){
  int sum = 1;
  vector<int>::iterator v = vec.begin();
   while( v != vec.end()) {
      sum *= *v;
      v++;
   }
  return sum;
}


void modelCompute(){
  float imgData[2][112][112][3] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
  float output[56*56*2] = {20.2,33.5,33.4};
  map<int, float> outputs;
  outputs[0] = 2001.0;
  
  std::vector<int> dataTypes;
  std::vector<int> numsOfBytes;
  std::vector<std::vector<int>> sizes;

  int length = sizeof(imgData) / sizeof(imgData[0]);
  for(int i = 0; i<length; i++){
    // 获取输入数据类型
    Data* data = new Data(typeid(imgData[i][0][0][0]).name());
    dataTypes.push_back(data->getNumber());
    std::vector<int> size;
    // 输入为数组形式
    size.push_back(sizeof(imgData) / sizeof(imgData[i]));
    size.push_back(sizeof(imgData[i]) / sizeof(imgData[i][0]));
    size.push_back(sizeof(imgData[i][0]) / sizeof(imgData[i][0][0]));
    size.push_back(sizeof(imgData[i][0][0]) / sizeof(imgData[i][0][0][0]));
    sizes.push_back(size);

    int numbers = numElements(sizes[i]);
    cout << numbers << endl;
    numsOfBytes.push_back(numElements(sizes[i]) * dataTypes[0]);
  }

  bool memory_allocated = true; 
  long interpreter_handle = 21345789;
  long error_handle = 345678909;
  run(interpreter_handle, error_handle, sizes, dataTypes, numsOfBytes, /*ingData,*/ memory_allocated);
}






// test
int main(){
  cout << "start..." << endl;
  time_t startTime = clock();

  string model_file = "/home/tclxa/TCL/NLDF_Bokeh/tmp/mobilenet_quant_v1_224.tflite";
  // createErrorReporter();
  //BufferErrorReporter* errorReporter = createErrorReporter();
  
  // createModel(model_file)
  //std::unique_ptr<tflite::FlatBufferModel> model = createModel(errorReporter, model_file);
  int num_threads = 1;
  // createInterpreter(model_file, num_threads);
  // tflite::Interpreter* interpreterHandle = nullptr;

  // 将输入输出数据存放在Map
  // char imgData[1000] = "dscsd";
  

  modelCompute();

  time_t endTime = clock();

  cout << endTime-startTime << " ms" << endl;
  cout << "end...." << endl;
  return 0;
}

