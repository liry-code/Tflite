#include <iostream>
#include "schema_generated.h"
// #include "/home/tclxa/TfLite/exception_jni.h"
#include "nativeinterpreterwrapper_jni.h"
#include "model.h"

namespace {

const int kByteBufferValue = 999;
const int kBufferSize = 256;

// #include "exception_jni.h"



JNIFlatBufferVerifier* error = new JNIFlatBufferVerifier();
/**LiYu*/
std::unique_ptr<tflite::FlatBufferModel> createModel(string model_file) ;
/** LiYu*/
BufferErrorReporter* createErrorReporter();




int GetArrayLength(int arr[]){
  return sizeof(arr)/sizeof(arr[0]);
}

int GetArrayLength(float arr[]){
  return sizeof(arr)/sizeof(arr[0]);
}


void printDims(char* buffer, int max_size, int* dims, int num_dims) {
	if (max_size <= 0) return;
	buffer[0] = '?';
	int size = 1;
	for (int i = 1; i < num_dims; ++i) {
		if (max_size > size) {
			int written_size =
			  snprintf(buffer + size, max_size - size, ",%d", dims[i]);
			if (written_size < 0) return;
			size += written_size;
		}		
	}
}


TfLiteStatus checkInputs(tflite::Interpreter* interpreter,
                         const int input_size,  std::vector<int> &dataTypes, 
                         std::vector<int> &numsOfBytes,
                         std::vector<std::vector<int>> &sizes) {
  if (input_size != interpreter->inputs().size()) {
    // throwException(kIllegalArgumentException,
    //                "Expected num of inputs is %d but got %d",
    //                interpreter->inputs().size(), input_size);
    return kTfLiteError;
  }

  /*LiYu (TODO) Exception */
  if (input_size != dataTypes.size() ||
      input_size != numsOfBytes.size() 
      /* || input_size != GetArrayLength(values) */) {
  //   throwException(env, kIllegalArgumentException,
  //                  "Arrays in arguments should be of the same length, but got "
  //                  "%d sizes, %d data_types, %d nums_of_bytes, and %d values",
  //                  input_size, env->GetArrayLength(data_types),
  //                  env->GetArrayLength(nums_of_bytes),
  //                  env->GetArrayLength(values));
    return kTfLiteError;
  }
  for (int i = 0; i < input_size; ++i) {
    // cout << interpreter->inputs().size() << endl;
    int input_idx = interpreter->inputs()[i];
    TfLiteTensor* target = interpreter->tensor(input_idx);
    // jintArray dims =
        // static_cast<jintArray>(env->GetObjectArrayElement(sizes, i));
    std::vector<int> dims = sizes[i];
    // int num_dims = static_cast<int>(env->GetArrayLength(dims));
    int num_dims = dims.size();
    
    if (target->dims->size != num_dims) {
  //     throwException(env, kIllegalArgumentException,
  //                    "%d-th input should have %d dimensions, but found %d "
  //                    "dimensions",
  //                    i, target->dims->size, num_dims);
      return kTfLiteError;
    }
    // jint* ptr = env->GetIntArrayElements(dims, nullptr);
    for (int j = 1; j < num_dims; ++j) {
      if (target->dims->data[j] != dims[j]) {
        std::unique_ptr<char[]> expected_dims(new char[kBufferSize]);
        std::unique_ptr<char[]> obtained_dims(new char[kBufferSize]);
        printDims(expected_dims.get(), kBufferSize, target->dims->data,
                  num_dims);
        // 将vector转换为数组
        int *sizeArray = new int[sizes.size()];
        if (!sizes.empty()){
            memcpy(sizeArray, &sizes[0], sizes.size()*sizeof(int));
        }

        printDims(obtained_dims.get(), kBufferSize, sizeArray, num_dims);
  //       throwException(env, kIllegalArgumentException,
  //                      "%d-th input dimension should be [%s], but found [%s]",
  //                      i, expected_dims.get(), obtained_dims.get());
  //       env->ReleaseIntArrayElements(dims, ptr, JNI_ABORT);
        return kTfLiteError;
      }
    }
  //   env->ReleaseIntArrayElements(dims, ptr, JNI_ABORT);
  //   env->DeleteLocalRef(dims);
  //   if (env->ExceptionCheck()) return kTfLiteError;
  }
  cout << "nativewrapper checkInputs end..." << endl;
  return kTfLiteOk;
}

bool areDimsDifferent(TfLiteTensor* tensor, std::vector<int> &dims) {
	int num_dims = dims.size();
	// jint* ptr = env->GetIntArrayElements(dims, nullptr);
	// if (ptr == nullptr) {
	//   throwException(env, kIllegalArgumentException,
	//                  "Empty dimensions of input array.");
	//   return true;
	// }
	if (tensor->dims->size != num_dims) {
		return true;
	}
	for (int i = 0; i < num_dims; ++i) {
		// if (ptr[i] != tensor->dims->data[i]) {
		//   return true;
		// }
		if (dims[i] != tensor->dims->data[i]) {
		  return true;
		}
	}
	return false;
}

bool areInputDimensionsTheSame(tflite::Interpreter* interpreter,
                               int input_size, std::vector<std::vector<int>> &sizes) {
  if (interpreter->inputs().size() != input_size) {
    return false;
  }
  for (int i = 0; i < input_size; ++i) {
    int input_idx = interpreter->inputs()[i];
    // jintArray dims =
    //     static_cast<jintArray>(env->GetObjectArrayElement(sizes, i));
    std::vector<int> dims = sizes[i];
    TfLiteTensor* target = interpreter->tensor(input_idx);
    // if (areDimsDifferent(env, target, dims)) return false;
    if (areDimsDifferent(target, dims)) return false;
    // env->DeleteLocalRef(dims);
    // if (env->ExceptionCheck()) return false;
  }
  return true;
}


TfLiteStatus resizeInputs(tflite::Interpreter* interpreter,
                          int input_size, std::vector<std::vector<int>> &sizes) {
  for (int i = 0; i < input_size; ++i) {
    int input_idx = interpreter->inputs()[i];
    // jintArray dims =
    //     static_cast<jintArray>(env->GetObjectArrayElement(sizes, i));
    std::vector<int> dims = sizes[i];

    // TfLiteStatus status = interpreter->ResizeInputTensor(
        // input_idx, convertJIntArrayToVector(env, dims));
    TfLiteStatus status = interpreter->ResizeInputTensor(
        input_idx, dims);
    if (status != kTfLiteOk) {
      return status;
    }
    // env->DeleteLocalRef(dims);
    // if (env->ExceptionCheck()) return kTfLiteError;
  }
  return kTfLiteOk;
}


void* run(
    long interpreter_handle, long error_handle,
    std::vector<std::vector<int>> &sizes, std::vector<int> &dataTypes, 
    std::vector<int> &numsOfBytes,
    /*float values[],*/ bool memory_allocated) {

  tflite::Interpreter* interpreter = new tflite::Interpreter();
  if (interpreter == nullptr) return nullptr;

  BufferErrorReporter* error_reporter = nullptr;
  // if (error_reporter == nullptr) return nullptr;

  const int input_size = sizes.size();
  TfLiteStatus status = checkInputs(interpreter, input_size, dataTypes,
                                    numsOfBytes, /*values,*/ sizes);
  if (status != kTfLiteOk) return nullptr;
  if (!memory_allocated ||
      !areInputDimensionsTheSame(interpreter, input_size, sizes)) {
  //   // resizes inputs
    status = resizeInputs(interpreter, input_size, sizes);
    if (status != kTfLiteOk) {
      /*LiYu (TODO)Exceptin*/
  //     throwException(env, kNullPointerException, "Can not resize the input: %s",
  //                    error_reporter->CachedErrorMessage());
      return nullptr;
    }
    // allocates memory
    status = interpreter->AllocateTensors();
  //   if (status != kTfLiteOk) {
  //     throwException(env, kNullPointerException,
  //                    "Can not allocate memory for the given inputs: %s",
  //                    error_reporter->CachedErrorMessage());
  //     return nullptr;
  //   }
  }
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
  cout << "run over " << endl;
  return nullptr;
}

}  //namespace