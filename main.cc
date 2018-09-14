#include "nativeinterpreterwrapper_jni.h"
#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <memory>
#include <ctime>
#include <map>
//#include <ext/hash_map>
//#include <typeinfo>
#include "DataType.h"

using namespace cv;
using namespace std;


#define DIM_BATCH_SIZE 1
#define DIM_IMG_SIZE_X 112
#define DIM_IMG_SIZE_Y 112
#define DIM_PIXEL_SIZE 3
#define numBytesPerChannel 4

vector<int> getInputDims(long interpreterHandle, int i, int numsOfBytes){
  vector<int> vec;
  vec.push_back(10);
  return vec;
}

void modelCompute(){
  float output[56*56*2] = {20.2,33.5,33.4};
  map<int, float> outputs;
  outputs[0] = 2001.0;
  
  string path = "/home/tclxa/11.jpg";
  Mat src = imread(path); //从路径名中读取图片

  Mat dst;
  imshow("src", src);//显示图片
  resize(src, dst, Size(DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y), (0, 0), (0, 0), INTER_LINEAR);//重新调整图像大小
  

  // 使用vector替换数组
  vector<char> bufferVector(DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE * numBytesPerChannel);
  int countVec = 0;
  for(int row = 0; row<DIM_IMG_SIZE_X; row++){
    for(int col =0 ; col<DIM_IMG_SIZE_Y; col++){
      Scalar intensity = dst.at<uchar>( row,  col);
      int value = intensity[0];
      // cout << "intensity value " << intensity[0] << endl;
      bufferVector.push_back((char)((value >> 16) & 0xFF));
      bufferVector.push_back((char)((value >> 8) & 0xFF));
      bufferVector.push_back((char)(value & 0xFF));
    }
  }

  vector<vector<char> > input;
  input.push_back(bufferVector);
  
  cout << "Vector input size " << input.size() << endl;
  
  
  vector<int> dataTypes(input.size());
  //int[] dataTypes = new int[input.size()];
  vector<vector<int> > sizes(input.size());
  //Object[] sizes = new Object[input.size()];
  vector<int> numsOfBytes(input.size());
  //int[] numsOfBytes = new int[input.size()];
  for(int i=0; i<input.size();i++){
    string dataTypeStr = typeid(input[i][0]).name();
    cout << "type name " << dataTypeStr << endl;
    DT dataType;
    dataType.setDataTypeName(dataTypeStr);
    cout << "bit " << dataType.getDataTypeBit() << endl;
    dataTypes.push_back(dataType.getDataTypeBit());
    cout << "it`s ok " << endl;
    if (strcmp("c",dataTypeStr.c_str())==0) {
      numsOfBytes[i] = input[i].size();
      //调用Native方法
      long interpreterHandle = 10000000000;
      sizes.push_back(getInputDims(interpreterHandle, i, numsOfBytes[i]));
      cout << "dataType class enter " << endl;
    }
  }

  bool memory_allocated = true; 
  long interpreter_handle = 21345789;
  long error_handle = 345678909;
  //run(interpreter_handle, error_handle, sizes, dataTypes, numsOfBytes, /*ingData,*/ memory_allocated);
}





// test
int main(){
  cout << "start..." << endl;
  time_t startTime = clock();

  string model_file = "/home/tclxa/TCL/tmp/mobilenet_quant_v1_224.tflite";
  // createErrorReporter();
  BufferErrorReporter* errorReporter = createErrorReporter();
  
  // createModel(model_file)
  std::unique_ptr<tflite::FlatBufferModel> model = createModel(errorReporter, model_file);
  int num_threads = 1;
  // createInterpreter(model_file, num_threads);
  tflite::Interpreter* interpreterHandle = nullptr;

  // 将输入输出数据存放在Map
  // char imgData[1000] = "dscsd";
  
  
  modelCompute();

  time_t endTime = clock();

  cout << endTime-startTime << " ms" << endl;
  cout << "end...." << endl;
  return 0;
}