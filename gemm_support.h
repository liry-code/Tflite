#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_GEMM_SUPPORT_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_GEMM_SUPPORT_H_

#include "/home/tclxa/TfLite/gemmlowp.h"
#include "/home/tclxa/TfLite/context.h"

namespace tflite {
namespace gemm_support {

// Returns the GemmContext stored in 'context', allowing multiple ops to
// share a single object, as long as they share a TfLiteContext. The caller
// must ensure that this is called between IncrementUsageCounter() and
// DecrementUsageCounter(). For example, in the implementation of an op:
//   void* Init(TfLiteContext* context, const char*, size_t) {
//     gemm_support::IncrementUsageCounter(context);
//     return nullptr;
//   }
//   void Free(TfLiteContext* context, void*) {
//     gemm_support::DecrementUsageCounter(context);
//   }
//   TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
//     auto* gemm_context = gemm_support::GetFromContext(context);
//   }
gemmlowp::GemmContext* GetFromContext(TfLiteContext* context);

// Let the framework know that the GemmContext stored in 'context' will be used
// by an op. If necessary a new GemmContext is created and placed in 'context'.
void IncrementUsageCounter(TfLiteContext* context);

// Let the framework know that the op stopped using the GemmContext stored in
// 'context'. If there are no more usages the GemmContext will be deleted.
void DecrementUsageCounter(TfLiteContext* context);

}  // namespace gemm_support
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_GEMM_SUPPORT_H_
