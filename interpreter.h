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

#include "/home/tclxa/TfLite/allocation.h"
#include "/home/tclxa/TfLite/context.h"
#include "/home/tclxa/TfLite/error_reporter.h"
// #include "tensorflow/contrib/lite/memory_planner.h"
// #include "tensorflow/contrib/lite/profiling/profiler.h"

namespace tflite {

// // Map statically from a c++ type to a TfLiteType (used below for safe casts).
// template <class T>
// constexpr TfLiteType typeToTfLiteType() {
//   return kTfLiteNoType;
// }
// template <>
// constexpr TfLiteType typeToTfLiteType<int>() {
//   return kTfLiteInt32;
// }
// template <>
// constexpr TfLiteType typeToTfLiteType<int16_t>() {
//   return kTfLiteInt16;
// }
// template <>
// constexpr TfLiteType typeToTfLiteType<int64_t>() {
//   return kTfLiteInt64;
// }
// template <>
// constexpr TfLiteType typeToTfLiteType<float>() {
//   return kTfLiteFloat32;
// }
// template <>
// constexpr TfLiteType typeToTfLiteType<unsigned char>() {
//   return kTfLiteUInt8;
// }
// template <>
// constexpr TfLiteType typeToTfLiteType<bool>() {
//   return kTfLiteBool;
// }
// template <>
// constexpr TfLiteType typeToTfLiteType<std::complex<float>>() {
//   return kTfLiteComplex64;
// }
// template <>
// constexpr TfLiteType typeToTfLiteType<string>() {
//   return kTfLiteString;
// }

// // Forward declare since NNAPIDelegate uses Interpreter.
// class NNAPIDelegate;

// // An interpreter for a graph of nodes that input and output from tensors.
// // Each node of the graph processes a set of input tensors and produces a
// // set of output Tensors. All inputs/output tensors are referenced by index.
// //
// // Usage:
// //
// // -- Create basic model
// // Interpreter foo(2, 1);
// // foo.SetTensorParametersReadWrite(0, ...);
// // foo.SetTensorParametersReadOnly(1, ...);
// // foo.SetNodeParameters(0, ...)
// //
// // -- Resize input array to 1 length.
// // foo.ResizeInputTensor(0, 1);
// // foo.AllocateTensors();
// // -- Install array data
// // foo.typed_tensor<float>(0)[0] = 3;
// // foo.Invoke();
// // foo.typed_tensor<float>(0)[0] = 4;
// // foo.Invoke();
// // -- Resize input array and set data.
// // foo.ResizeInputTensor(0, 2);
// // foo.AllocateTensors();
// // foo.typed_tensor<float>(0)[0] = 4;
// // foo.typed_tensor<float>(0)[1] = 8;
// // foo.Invoke();
// //

// struct TfLiteIntArrayDeleter {
//   void operator()(TfLiteIntArray* a) {
//     if (a) TfLiteIntArrayFree(a);
//   }
// };

class Interpreter {
 public:
//  explicit Interpreter(ErrorReporter* error_reporter = DefaultErrorReporter());

//   ~Interpreter();

  Interpreter(const Interpreter&) = delete;
  Interpreter& operator=(const Interpreter&) = delete;

//   TfLiteStatus SetInputs(std::vector<int> inputs);

//   TfLiteStatus SetOutputs(std::vector<int> outputs);


//   TfLiteStatus SetVariables(std::vector<int> variables);

//   void ReserveNodes(int count);

//   TfLiteStatus AddNodeWithParameters(const std::vector<int>& inputs,
//                                      const std::vector<int>& outputs,
//                                      const char* init_data,
//                                      size_t init_data_size, void* builtin_data,
//                                      const TfLiteRegistration* registration,
//                                      int* node_index = nullptr);


//   TfLiteStatus AddTensors(int tensors_to_add,
//                           int* first_new_tensor_index = nullptr);

//   inline TfLiteStatus SetTensorParametersReadOnly(
//       int tensor_index, TfLiteType type, const char* name,
//       const std::vector<int>& dims, TfLiteQuantizationParams quantization,
//       const char* buffer, size_t bytes,
//       const Allocation* allocation = nullptr) {
//     return SetTensorParametersReadOnly(tensor_index, type, name, dims.size(),
//                                        dims.data(), quantization, buffer, bytes,
//                                        allocation);
//   }

//   TfLiteStatus SetTensorParametersReadOnly(
//       int tensor_index, TfLiteType type, const char* name, const size_t rank,
//       const int* dims, TfLiteQuantizationParams quantization,
//       const char* buffer, size_t bytes, const Allocation* allocation = nullptr);


//   inline TfLiteStatus SetTensorParametersReadWrite(
//       int tensor_index, TfLiteType type, const char* name,
//       const std::vector<int>& dims, TfLiteQuantizationParams quantization,
//       bool is_variable = false) {
//     return SetTensorParametersReadWrite(tensor_index, type, name, dims.size(),
//                                         dims.data(), quantization, is_variable);
//   }
//   TfLiteStatus SetTensorParametersReadWrite(
//       int tensor_index, TfLiteType type, const char* name, const size_t rank,
//       const int* dims, TfLiteQuantizationParams quantization,
//       bool is_variable = false);


//   const std::vector<int>& inputs() const { return inputs_; }

//   const char* GetInputName(int index) const {
//     return context_.tensors[inputs_[index]].name;
//   }

//   const std::vector<int>& outputs() const { return outputs_; }

//   const std::vector<int>& variables() const { return variables_; }

//   const char* GetOutputName(int index) const {
//     return context_.tensors[outputs_[index]].name;
//   }

//   size_t tensors_size() const { return context_.tensors_size; }

//   size_t nodes_size() const { return nodes_and_registration_.size(); }

//   const std::vector<int>& execution_plan() const { return execution_plan_; }

//   TfLiteStatus SetExecutionPlan(const std::vector<int>& new_plan);

//   TfLiteTensor* tensor(int tensor_index) {
//     if (tensor_index >= context_.tensors_size || tensor_index < 0)
//       return nullptr;
//     return &context_.tensors[tensor_index];
//   }

//   const TfLiteTensor* tensor(int tensor_index) const {
//     if (tensor_index >= context_.tensors_size || tensor_index < 0)
//       return nullptr;
//     return &context_.tensors[tensor_index];
//   }

//   const std::pair<TfLiteNode, TfLiteRegistration>* node_and_registration(
//       int node_index) const {
//     if (node_index >= nodes_and_registration_.size() || node_index < 0)
//       return nullptr;
//     return &nodes_and_registration_[node_index];
//   }

//   template <class T>
//   T* typed_tensor(int tensor_index) {
//     if (TfLiteTensor* tensor_ptr = tensor(tensor_index)) {
//       if (tensor_ptr->type == typeToTfLiteType<T>()) {
//         return reinterpret_cast<T*>(tensor_ptr->data.raw);
//       }
//     }
//     return nullptr;
//   }

//   template <class T>
//   const T* typed_tensor(int tensor_index) const {
//     if (const TfLiteTensor* tensor_ptr = tensor(tensor_index)) {
//       if (tensor_ptr->type == typeToTfLiteType<T>()) {
//         return reinterpret_cast<const T*>(tensor_ptr->data.raw);
//       }
//     }
//     return nullptr;
//   }

//   template <class T>
//   T* typed_input_tensor(int index) {
//     return typed_tensor<T>(inputs_[index]);
//   }

//   template <class T>
//   const T* typed_input_tensor(int index) const {
//     return typed_tensor<T>(inputs_[index]);
//   }

//   template <class T>
//   T* typed_output_tensor(int index) {
//     return typed_tensor<T>(outputs_[index]);
//   }

//   template <class T>
//   const T* typed_output_tensor(int index) const {
//     return typed_tensor<T>(outputs_[index]);
//   }

//   TfLiteStatus ResizeInputTensor(int tensor_index,
//                                  const std::vector<int>& dims);

//   TfLiteStatus AllocateTensors();

//   TfLiteStatus Invoke();

//   void UseNNAPI(bool enable);

//   void SetNumThreads(int num_threads);

//   TfLiteStatus ModifyGraphWithDelegate(TfLiteDelegate* delegate,
//                                        bool allow_dynamic_tensors = false);

//   TfLiteStatus EnsureTensorDataIsReadable(int tensor_index) {
//     TF_LITE_ENSURE(&context_, tensor_index < tensors_size());
//     TfLiteTensor* tensor = &tensors_[tensor_index];
//     if (tensor->data_is_stale) {
//       TF_LITE_ENSURE(&context_, tensor->delegate != nullptr);
//       TF_LITE_ENSURE(&context_,
//                      tensor->buffer_handle != kTfLiteNullBufferHandle);
//       // This can be null if the delegate doesn't use its own buffer.
//       TF_LITE_ENSURE(&context_,
//                      tensor->delegate->CopyFromBufferHandle != nullptr);
//       tensor->delegate->CopyFromBufferHandle(&context_, tensor->delegate,
//                                              tensor->buffer_handle,
//                                              tensor->data.raw, tensor->bytes);
//       tensor->data_is_stale = false;
//     }
//     return kTfLiteOk;
//   }

//   TfLiteStatus SetBufferHandle(int tensor_index,
//                                TfLiteBufferHandle buffer_handle,
//                                TfLiteDelegate* delegate);

//   TfLiteStatus GetBufferHandle(int tensor_index,
//                                TfLiteBufferHandle* buffer_handle,
//                                TfLiteDelegate** delegate);

//   void SetProfiler(profiling::Profiler* profiler) { profiler_ = profiler; }

//   profiling::Profiler* GetProfiler() { return profiler_; }

//   static constexpr int kTensorsReservedCapacity = 128;
  
//   static constexpr int kTensorsCapacityHeadroom = 16;

//   void SetAllowBufferHandleOutput(bool allow_buffer_handle_output) {
//     allow_buffer_handle_output_ = allow_buffer_handle_output;
//   }

//   TfLiteStatus ResetVariableTensorsToZero();

//   const char* OpProfilingString(const TfLiteRegistration& op_reg,
//                                 const TfLiteNode* node) const {
//     if (op_reg.profiling_string == nullptr) return nullptr;
//     return op_reg.profiling_string(&context_, node);
//   }

//   void SetExternalContext(TfLiteExternalContextType type,
//                           TfLiteExternalContext* ctx);

 private:
//   friend class InterpreterBuilder;
//   friend class InterpreterTest;

//   void SwitchToKernelContext();

//   void SwitchToDelegateContext();

//   void* OpInit(const TfLiteRegistration& op_reg, const char* buffer,
//                size_t length) {
//     if (op_reg.init == nullptr) return nullptr;
//     return op_reg.init(&context_, buffer, length);
//   }

//   void OpFree(const TfLiteRegistration& op_reg, void* buffer) {
//     if (op_reg.free == nullptr) return;
//     if (buffer) {
//       op_reg.free(&context_, buffer);
//     }
//   }

//   TfLiteStatus OpPrepare(const TfLiteRegistration& op_reg, TfLiteNode* node) {
//     if (op_reg.prepare == nullptr) return kTfLiteOk;
//     return op_reg.prepare(&context_, node);
//   }

//   TfLiteStatus OpInvoke(const TfLiteRegistration& op_reg, TfLiteNode* node) {
//     if (op_reg.invoke == nullptr) return kTfLiteError;
//     return op_reg.invoke(&context_, node);
//   }

//   TfLiteStatus PrepareOpsAndTensors();

//   TfLiteStatus PrepareOpsStartingAt(int first_execution_plan_index,
//                                     int* last_execution_plan_index_prepared);

//   std::vector<TfLiteTensor> tensors_;

//   TfLiteStatus CheckTensorIndices(const char* label, const int* indices,
//                                   int length);

//   TfLiteStatus BytesRequired(TfLiteType type, const int* dims, size_t dims_size,
//                              size_t* bytes);

//   TfLiteStatus ResizeTensorImpl(TfLiteTensor* tensor, TfLiteIntArray* new_size);

//   void ReportErrorImpl(const char* format, va_list args);

//   static TfLiteStatus ResizeTensor(TfLiteContext* context, TfLiteTensor* tensor,
//                                    TfLiteIntArray* new_size);
//   static void ReportError(TfLiteContext* context, const char* format, ...);

//   static TfLiteStatus AddTensors(TfLiteContext* context, int tensors_to_add,
//                                  int* first_new_tensor_index);

//   static TfLiteStatus ReplaceSubgraphsWithDelegateKernels(
//       TfLiteContext* context, TfLiteRegistration registration,
//       const TfLiteIntArray* nodes_to_replace, TfLiteDelegate* delegate);

//   TfLiteStatus ReplaceSubgraphsWithDelegateKernels(
//       TfLiteRegistration registration, const TfLiteIntArray* nodes_to_replace,
//       TfLiteDelegate* delegate);

//   TfLiteStatus GetNodeAndRegistration(int node_index, TfLiteNode** node,
//                                       TfLiteRegistration** registration);

//   static TfLiteStatus GetNodeAndRegistration(struct TfLiteContext*,
//                                              int node_index, TfLiteNode** node,
//                                              TfLiteRegistration** registration);

//   TfLiteStatus GetExecutionPlan(TfLiteIntArray** execution_plan);

//   static TfLiteStatus GetExecutionPlan(struct TfLiteContext* context,
//                                        TfLiteIntArray** execution_plan);

//   TfLiteExternalContext* GetExternalContext(TfLiteExternalContextType type);
//   static TfLiteExternalContext* GetExternalContext(
//       struct TfLiteContext* context, TfLiteExternalContextType type);

//   static void SetExternalContext(struct TfLiteContext* context,
//                                  TfLiteExternalContextType type,
//                                  TfLiteExternalContext* ctx);

//   using TfLiteDelegatePtr =
//       std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>;

//   // Variant of the public ModifyGraphWithDelegate method that additionally
//   // Assumes ownership of the provided delegate.
//   // WARNING: This is an experimental API and subject to change.
//   template <typename Delegate>
//   TfLiteStatus ModifyGraphWithDelegate(std::unique_ptr<Delegate> typed_delegate,
//                                        bool allow_dynamic_tensors = false) {
//     TfLiteDelegatePtr delegate(typed_delegate.release(),
//                                [](TfLiteDelegate* delegate) {
//                                  delete static_cast<Delegate*>(delegate);
//                                });
//     // Note that we retain ownership of the delegate even if graph modification
//     // fails, as delegate use will be in an indeterminate state at that point.
//     owned_delegates_.push_back(std::move(delegate));
//     return ModifyGraphWithDelegate(owned_delegates_.back().get(),
//                                    allow_dynamic_tensors);
//   }

//   // Ensures that `tensors_` has at least `kTensorsCapacityHeadroom` extra
//   // capacity. Calling this function may invalidate existing pointers to
//   // tensors. After calling this function, adding `kTensorsCapacityHeadroom`
//   // more tensors won't invalidate the pointer to existing tensors.
//   void EnsureTensorsVectorCapacity() {
//     const size_t required_capacity = tensors_size() + kTensorsCapacityHeadroom;
//     if (required_capacity > tensors_.capacity()) {
//       tensors_.reserve(required_capacity);
//       context_.tensors = tensors_.data();
//     }
//   }

//   // The state of the Interpreter.
//   enum State {
//     // The interpreter isn't ready to be invoked.
//     // `AllocateTensor` need to be called to enter an invokable state.
//     kStateUninvokable = 0,
//     // The interpreter is ready to be invoked.
//     kStateInvokable,
//     // The interpreter is ready to be invoked, and graph can't be further
//     // modified. The interpreter will enter this state when calling
//     // `ModifyGraphWithDelegate` with `allow_dynamic_tensors=false`.
//     kStateInvokableAndImmutable,
//   };
//   State state_ = kStateUninvokable;

//   // A pure C data structure used to communicate with the pure C plugin
//   // interface. To avoid copying tensor metadata, this is also the definitive
//   // structure to store tensors.
//   TfLiteContext context_;

//   // Node inputs/outputs are stored in TfLiteNode and TfLiteRegistration stores
//   // function pointers to actual implementation.
//   std::vector<std::pair<TfLiteNode, TfLiteRegistration>>
//       nodes_and_registration_;

//   // Whether the model is consistent. That is to say if the inputs and outputs
//   // of every node and the global inputs and outputs are valid indexes into
//   // the tensor array.
//   bool consistent_ = true;

//   // Array of indices representing the tensors that are inputs to the
//   // interpreter.
//   std::vector<int> inputs_;

//   // Array of indices representing the tensors that are outputs to the
//   // interpreter.
//   std::vector<int> outputs_;

//   // Array of indices representing the tensors that are variable tensors.
//   std::vector<int> variables_;

//   // The error reporter delegate that tflite will forward queries errors to.
//   ErrorReporter* error_reporter_;

//   // Index of the next node to prepare.
//   // During Invoke(), Interpreter will allocate input tensors first, which are
//   // known to be fixed size. Then it will allocate outputs from nodes as many
//   // as possible. When there is a node that produces dynamic sized tensor.
//   // Interpreter will stop allocating tensors, set the value of next allocate
//   // node id, and execute the node to generate the output tensor before continue
//   // to allocate successors. This process repeats until all nodes are executed.
//   // NOTE: this relies on the order of nodes that is in topological order.
//   int next_execution_plan_index_to_prepare_;

//   // WARNING: This is an experimental interface that is subject to change.
//   // This is a list of node indices (to index into nodes_and_registration).
//   // This represents a valid topological sort (dependency ordered) execution
//   // plan. In particular, it is valid for this ordering to contain only a
//   // subset of the node indices.
//   std::vector<int> execution_plan_;

//   // In the future, we'd like a TfLiteIntArray compatible representation.
//   // TODO(aselle): replace execution_plan_ with this.
//   std::unique_ptr<TfLiteIntArray, TfLiteIntArrayDeleter> plan_cache_;

//   // Whether to delegate to NN API
//   std::unique_ptr<NNAPIDelegate> nnapi_delegate_;

//   // List of delegates that have been installed and are owned by this
//   // interpreter instance. Useful if client delegate ownership is burdensome.
//   // WARNING: This is an experimental API and subject to change.
//   std::vector<TfLiteDelegatePtr> owned_delegates_;

//   std::unique_ptr<MemoryPlanner> memory_planner_;

//   bool allow_buffer_handle_output_ = false;

//   // Tracking bit for whether a tensor was resized in the course of an op
//   // invocation. This is a useful hint to ensure that dynamic tensor outputs
//   // trigger downstream reallocation after op invocation.
//   bool tensor_resized_since_op_invoke_ = false;

//   // Profiler for this interpreter instance.
//   profiling::Profiler* profiler_ = nullptr;

//   // List of active external contexts.
//   TfLiteExternalContext* external_contexts_[kTfLiteMaxExternalContexts];
};

}  // namespace tflite
#endif  // INTERPRETER_H_
