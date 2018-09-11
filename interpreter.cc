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

#include "interpreter.h"

#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstring>

#include "arena_planner.h"
#include "context.h"
// #include "tensorflow/contrib/lite/context_util.h"
#include "error_reporter.h"
#include "graph_info.h"
#include "memory_planner.h"
#include "nnapi_delegate.h"
// #include "tensorflow/contrib/lite/profiling/profiler.h"
// #include "tensorflow/contrib/lite/schema/schema_generated.h"
#include "tf_util.h"

namespace tflite {
namespace {

}  // namespace



class InterpreterInfo : public GraphInfo {
 public:
	explicit InterpreterInfo(Interpreter* interpreter)
	  : interpreter_(interpreter) {}

	size_t num_tensors() const override { return interpreter_->tensors_size(); }
	TfLiteTensor* tensor(size_t index) override {
	return interpreter_->tensor(index);
	}
	size_t num_nodes() const override {
	return interpreter_->execution_plan().size();
	}
	const TfLiteNode& node(size_t index) const override {
	int node_index = interpreter_->execution_plan()[index];
	return interpreter_->node_and_registration(node_index)->first;
	}
	const std::vector<int>& inputs() const override {
	return interpreter_->inputs();
	}
	const std::vector<int>& outputs() const override {
	return interpreter_->outputs();
	}

	public:
	Interpreter* interpreter_;
};

TfLiteStatus Interpreter::ResizeInputTensor(int tensor_index,
                                            const std::vector<int>& dims) {
	if (state_ == kStateInvokableAndImmutable) {
		ReportError(&context_,
		            "ResizeInputTensor is disallowed when graph is immutable.");
		return kTfLiteError;
	}
	state_ = kStateUninvokable;

	// TODO(aselle): All bounds checks can be implemented as one-sided bounds
	// checks by casting to unsigned for efficiency. Profile before doing this.
	TF_LITE_ENSURE(&context_,
	             tensor_index < context_.tensors_size && tensor_index >= 0);
	TfLiteIntArray* dims_lite = ConvertVectorToTfLiteIntArray(dims);
	return ResizeTensorImpl(&context_.tensors[tensor_index], dims_lite);
}

TfLiteStatus Interpreter::BytesRequired(TfLiteType type, const int* dims,
                                        int dims_size, size_t* bytes) {
	// TODO(aselle): Check for overflow here using overflow.h in TensorFlow
	// MultiplyWithoutOverflow.
	TF_LITE_ENSURE(&context_, bytes != nullptr);
	size_t count = 1;
	for (int k = 0; k < dims_size; k++) count *= dims[k];
	switch (type) {
		case kTfLiteFloat32:
		  *bytes = sizeof(float) * count;
		  break;
		case kTfLiteInt32:
		  *bytes = sizeof(int32_t) * count;
		  break;
		case kTfLiteUInt8:
		  *bytes = sizeof(uint8_t) * count;
		  break;
		case kTfLiteInt64:
		  *bytes = sizeof(int64_t) * count;
		  break;
		default:
		  ReportError(&context_,
		              "Only float32, int32, int64, uint8 supported currently.");
		  return kTfLiteError;
	}
	return kTfLiteOk;
}

TfLiteStatus Interpreter::ResizeTensor(TfLiteContext* context,
                                       TfLiteTensor* tensor,
                                       TfLiteIntArray* new_size) {
	// Note here that context->impl_ is recovering the this pointer for an
	// instance of Interpreter to call into the member function ResizeTensorImpl
	// (this function is static).
	return static_cast<Interpreter*>(context->impl_)
	  ->ResizeTensorImpl(tensor, new_size);
}

TfLiteStatus Interpreter::ResizeTensorImpl(TfLiteTensor* tensor,
                                           TfLiteIntArray* new_size) {
	if (tensor->allocation_type == kTfLiteArenaRw ||
	  tensor->allocation_type == kTfLiteDynamic) {
		if (tensor->type != kTfLiteString) {
		  size_t bytesRequired;
		  TfLiteStatus status = BytesRequired(tensor->type, new_size->data,
		                                      new_size->size, &bytesRequired);
		  if (status != kTfLiteOk) {
		    TfLiteIntArrayFree(new_size);
		    return kTfLiteError;
		  }

		  // Realloc space for kTfLiteDynamic tensors.
		  TfLiteTensorRealloc(bytesRequired, tensor);
		  tensor->bytes = bytesRequired;
		}
		if (tensor->dims) TfLiteIntArrayFree(tensor->dims);
		tensor->dims = new_size;

		if (tensor->allocation_type != kTfLiteDynamic) {
		  tensor->data.raw = nullptr;
		}
	} else {
		// kTfLiteMmapRo tensors are stored in the flatbuffer and are therefore
		// of fixed size.
		TfLiteIntArrayFree(new_size);
		ReportError(&context_, "Attempting to resize a fixed-size tensor.");
		return kTfLiteError;
	}
	return kTfLiteOk;
}

TfLiteStatus Interpreter::AllocateTensors() {
	next_execution_plan_index_to_prepare_ = 0;
	if (memory_planner_) {
		TF_LITE_ENSURE_STATUS(memory_planner_->ResetAllocations());
	}

	if (!consistent_) {
		ReportError(&context_, "AllocateTensors() called on inconsistent model.");
		return kTfLiteError;
	}

	TF_LITE_ENSURE_STATUS(PrepareOpsAndTensors());
	if (state_ == kStateUninvokable) {
		state_ = kStateInvokable;
	}
	TF_LITE_ENSURE(&context_, state_ == kStateInvokable ||
	                            state_ == kStateInvokableAndImmutable);
	return kTfLiteOk;
}


void Interpreter::ReportErrorImpl(const char* format, va_list args) {
	error_reporter_->Report(format, args);
}

void Interpreter::ReportError(TfLiteContext* context, const char* format, ...) {
	va_list args;
	va_start(args, format);
	auto* f = static_cast<Interpreter*>(context->impl_);
	// Note here that context->impl_ is recovering the this pointer for an
	// instance of Interpreter to call into the member function ReportErrorImpl
	// (this function is static).
	f->ReportErrorImpl(format, args);
	va_end(args);
}

TfLiteStatus Interpreter::PrepareOpsStartingAt(
		int first_execution_plan_index, int* last_execution_plan_index_prepared) {
	// for (int execution_plan_index = first_execution_plan_index;
	//    execution_plan_index < execution_plan_.size(); execution_plan_index++) {
	// 	int node_index = execution_plan_[execution_plan_index];
	// 	TfLiteNode& node = nodes_and_registration_[node_index].first;
	// 	const TfLiteRegistration& registration =
	//     	nodes_and_registration_[node_index].second;
	// 	EnsureTensorsVectorCapacity();
	// 	if (OpPrepare(registration, &node) == kTfLiteError) {
	//   		return kTfLiteError;
	// 	}

	// 	*last_execution_plan_index_prepared = execution_plan_index;

	// 	// Discontinue if the node has dynamic outputs. Note that we don't
	// 	// stop for dynamic temporary tensors since they won't affect the
	// 	// sizes of other tensors in the graph.
	// 	if (HasDynamicTensor(context_, node.outputs)) {
	// 		break;
	// 	}
	// }
	cout << "PrepareOpsStartingAt function waiting coding ... " << endl;
	return kTfLiteOk;
}

TfLiteStatus Interpreter::PrepareOpsAndTensors() {
	if (!memory_planner_) {
		memory_planner_.reset(new ArenaPlanner(
		    &context_, std::unique_ptr<GraphInfo> (new InterpreterInfo(this))));
		memory_planner_->PlanAllocations();
	}

	int last_exec_plan_index_prepared = 0;

	TF_LITE_ENSURE_STATUS(PrepareOpsStartingAt(
	  next_execution_plan_index_to_prepare_, &last_exec_plan_index_prepared));
	TF_LITE_ENSURE_STATUS(memory_planner_->ExecuteAllocations(
	  next_execution_plan_index_to_prepare_, last_exec_plan_index_prepared));

	next_execution_plan_index_to_prepare_ = last_exec_plan_index_prepared + 1;
	return kTfLiteOk;
}


} // namespace tflite

