/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Main abstraction controlling the tflite interpreter.
// See context.h for the API for defining operations (TfLiteRegistration).
#ifndef INTERPRETER_H_
#define INTERPRETER_H_

#include <iostream>
#include <complex>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <vector>
#include "context.h"

#include "arena_planner.h"
#include "allocation.h"
#include "error_reporter.h"
#include "tf_util.h"
#include "memory_planner.h"
#include "graph_info.h"
// #include "tensorflow/contrib/lite/profiling/profiler.h"

namespace tflite {

class Interpreter {
 public:
 	const std::vector<int>& inputs() const { return inputs_; }

 	TfLiteTensor* tensor(int tensor_index) {
    	if (tensor_index >= context_.tensors_size || tensor_index < 0)
      		return nullptr;
    	return &context_.tensors[tensor_index];
  	}

  	// void Interpreter::ReportErrorImpl(const char* format, va_list args) ;
  	void ReportErrorImpl(const char* format, va_list args);

  	// void ReportError(TfLiteContext* context, const char* format, ...);
  	static void ReportError(TfLiteContext* context, const char* format, ...);


	// TfLiteStatus Interpreter::BytesRequired(TfLiteType type, const int* dims,
                                        // int dims_size, size_t* bytes);
	TfLiteStatus BytesRequired(TfLiteType type, const int* dims,
                                        int dims_size, size_t* bytes);

	// TfLiteStatus ResizeTensorImpl(TfLiteTensor* tensor,
                                           // TfLiteIntArray* new_size);
	TfLiteStatus ResizeTensorImpl(TfLiteTensor* tensor,
                                           TfLiteIntArray* new_size);

	TfLiteStatus PrepareOpsStartingAt(int first_execution_plan_index,
                                    int* last_execution_plan_index_prepared);

  	// TfLiteStatus ResizeInputTensor(int tensor_index,
   //                               const std::vector<int>& dims);
  	TfLiteStatus ResizeInputTensor(int tensor_index,
                                            const std::vector<int>& dims) ;

	TfLiteStatus PrepareOpsAndTensors();

	// TfLiteStatus Interpreter::AllocateTensors();
	TfLiteStatus AllocateTensors();

	static TfLiteStatus ResizeTensor(TfLiteContext* context,
                                       TfLiteTensor* tensor,
                                       TfLiteIntArray* new_size);


	void OpFree(const TfLiteRegistration& op_reg, void* buffer) {
	    if (op_reg.free == nullptr) return;
	    if (buffer) {
	      op_reg.free(&context_, buffer);
	    }
	}

	int tensors_size() const { return context_.tensors_size; }
	const std::vector<int>& execution_plan() const { return execution_plan_; }
	const std::pair<TfLiteNode, TfLiteRegistration>* node_and_registration(
		int node_index) const {
		if (node_index >= nodes_and_registration_.size() || node_index < 0)
			return nullptr;
		return &nodes_and_registration_[node_index];
	}
	const std::vector<int>& outputs() const { return outputs_; }


 private:
 	// std::vector<int> inputs_;
 	/*LiYu*/
 	std::vector<int> inputs_ ;
 	std::vector<int> outputs_;
 	
 	TfLiteContext context_;
 	enum State {
	    // The interpreter isn't ready to be invoked.
	    // `AllocateTensor` need to be called to enter an invokable state.
	    kStateUninvokable = 0,
	    // The interpreter is ready to be invoked.
	    kStateInvokable,
	    // The interpreter is ready to be invoked, and graph can't be further
	    // modified. The interpreter will enter this state when calling
	    // `ModifyGraphWithDelegate` with `allow_dynamic_tensors=false`.
	    kStateInvokableAndImmutable,
	  };
 	State state_ = kStateUninvokable;

 	ErrorReporter* error_reporter_;
	int next_execution_plan_index_to_prepare_;
	std::unique_ptr<MemoryPlanner> memory_planner_;
	bool consistent_ = true;
	std::vector<std::pair<TfLiteNode, TfLiteRegistration>>
      nodes_and_registration_;
	std::vector<int> execution_plan_;
};







}  // namespace tflite
#endif  // INTERPRETER_H_
