#ifndef REGISTER_H_
#define REGISTER_H_

#include <unordered_map>
#include "/home/tclxa/TfLite/context.h"
#include "/home/tclxa/TfLite/model.h"
#include "/home/tclxa/TfLite/op_resolver.h"
#include "/home/tclxa/TfLite/util.h"
#include "/home/tclxa/TfLite/activations.h"
#include "/home/tclxa/TfLite/op_resolver.h"
#include "/home/tclxa/TfLite/pooling.h"
#include "/home/tclxa/TfLite/conv.h"

namespace tflite {
namespace ops {
namespace builtin {

class BuiltinOpResolver : public MutableOpResolver {
 public:
  
  // const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
                                   // int version) const override;
  // const TfLiteRegistration* FindOp(const char* op, int version) const override;

  // BuiltinOpResolver();
  BuiltinOpResolver() {
  AddBuiltin(BuiltinOperator_RELU, Register_RELU());
  AddBuiltin(BuiltinOperator_RELU_N1_TO_1, Register_RELU_N1_TO_1());
  AddBuiltin(BuiltinOperator_RELU6, Register_RELU6());
  AddBuiltin(BuiltinOperator_TANH, Register_TANH());
  AddBuiltin(BuiltinOperator_LOGISTIC, Register_LOGISTIC());
  AddBuiltin(BuiltinOperator_AVERAGE_POOL_2D, Register_AVERAGE_POOL_2D());
  AddBuiltin(BuiltinOperator_MAX_POOL_2D, Register_MAX_POOL_2D());
  AddBuiltin(BuiltinOperator_L2_POOL_2D, Register_L2_POOL_2D());
  AddBuiltin(BuiltinOperator_CONV_2D, Register_CONV_2D());
  AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, Register_DEPTHWISE_CONV_2D());
  AddBuiltin(BuiltinOperator_SVDF, Register_SVDF());
  AddBuiltin(BuiltinOperator_RNN, Register_RNN());
  AddBuiltin(BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN,
             Register_BIDIRECTIONAL_SEQUENCE_RNN());
  AddBuiltin(BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN,
             Register_UNIDIRECTIONAL_SEQUENCE_RNN());
  AddBuiltin(BuiltinOperator_EMBEDDING_LOOKUP, Register_EMBEDDING_LOOKUP());
  AddBuiltin(BuiltinOperator_EMBEDDING_LOOKUP_SPARSE,
             Register_EMBEDDING_LOOKUP_SPARSE());
  AddBuiltin(BuiltinOperator_FULLY_CONNECTED, Register_FULLY_CONNECTED(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_LSH_PROJECTION, Register_LSH_PROJECTION());
  AddBuiltin(BuiltinOperator_HASHTABLE_LOOKUP, Register_HASHTABLE_LOOKUP());
  AddBuiltin(BuiltinOperator_SOFTMAX, Register_SOFTMAX());
  AddBuiltin(BuiltinOperator_CONCATENATION, Register_CONCATENATION());
  AddBuiltin(BuiltinOperator_ADD, Register_ADD());
  AddBuiltin(BuiltinOperator_SPACE_TO_BATCH_ND, Register_SPACE_TO_BATCH_ND());
  AddBuiltin(BuiltinOperator_BATCH_TO_SPACE_ND, Register_BATCH_TO_SPACE_ND());
  AddBuiltin(BuiltinOperator_MUL, Register_MUL());
  AddBuiltin(BuiltinOperator_L2_NORMALIZATION, Register_L2_NORMALIZATION());
  AddBuiltin(BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION,
             Register_LOCAL_RESPONSE_NORMALIZATION());
  AddBuiltin(BuiltinOperator_LSTM, Register_LSTM(), /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM,
             Register_BIDIRECTIONAL_SEQUENCE_LSTM());
  AddBuiltin(BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM,
             Register_UNIDIRECTIONAL_SEQUENCE_LSTM());
  AddBuiltin(BuiltinOperator_PAD, Register_PAD());
  AddBuiltin(BuiltinOperator_PADV2, Register_PADV2());
  AddBuiltin(BuiltinOperator_RESHAPE, Register_RESHAPE());
  AddBuiltin(BuiltinOperator_RESIZE_BILINEAR, Register_RESIZE_BILINEAR());
  AddBuiltin(BuiltinOperator_SKIP_GRAM, Register_SKIP_GRAM());
  AddBuiltin(BuiltinOperator_SPACE_TO_DEPTH, Register_SPACE_TO_DEPTH());
  AddBuiltin(BuiltinOperator_GATHER, Register_GATHER());
  AddBuiltin(BuiltinOperator_TRANSPOSE, Register_TRANSPOSE());
  AddBuiltin(BuiltinOperator_MEAN, Register_MEAN());
  AddBuiltin(BuiltinOperator_DIV, Register_DIV());
  AddBuiltin(BuiltinOperator_SUB, Register_SUB());
  AddBuiltin(BuiltinOperator_SPLIT, Register_SPLIT());
  AddBuiltin(BuiltinOperator_SQUEEZE, Register_SQUEEZE());
  AddBuiltin(BuiltinOperator_STRIDED_SLICE, Register_STRIDED_SLICE());
  AddBuiltin(BuiltinOperator_EXP, Register_EXP());
  AddBuiltin(BuiltinOperator_TOPK_V2, Register_TOPK_V2());
  AddBuiltin(BuiltinOperator_LOG, Register_LOG());
  AddBuiltin(BuiltinOperator_LOG_SOFTMAX, Register_LOG_SOFTMAX());
  AddBuiltin(BuiltinOperator_CAST, Register_CAST());
  AddBuiltin(BuiltinOperator_DEQUANTIZE, Register_DEQUANTIZE());
  AddBuiltin(BuiltinOperator_PRELU, Register_PRELU());
  AddBuiltin(BuiltinOperator_MAXIMUM, Register_MAXIMUM());
  AddBuiltin(BuiltinOperator_MINIMUM, Register_MINIMUM());
  AddBuiltin(BuiltinOperator_ARG_MAX, Register_ARG_MAX());
  AddBuiltin(BuiltinOperator_ARG_MIN, Register_ARG_MIN());
  AddBuiltin(BuiltinOperator_GREATER, Register_GREATER());
  AddBuiltin(BuiltinOperator_GREATER_EQUAL, Register_GREATER_EQUAL());
  AddBuiltin(BuiltinOperator_LESS, Register_LESS());
  AddBuiltin(BuiltinOperator_LESS_EQUAL, Register_LESS_EQUAL());
  AddBuiltin(BuiltinOperator_FLOOR, Register_FLOOR());
  AddBuiltin(BuiltinOperator_NEG, Register_NEG());
  AddBuiltin(BuiltinOperator_SELECT, Register_SELECT());
  AddBuiltin(BuiltinOperator_SLICE, Register_SLICE());
  AddBuiltin(BuiltinOperator_SIN, Register_SIN());
  AddBuiltin(BuiltinOperator_TRANSPOSE_CONV, Register_TRANSPOSE_CONV());
  AddBuiltin(BuiltinOperator_TILE, Register_TILE());
  AddBuiltin(BuiltinOperator_SUM, Register_SUM());
  AddBuiltin(BuiltinOperator_REDUCE_PROD, Register_REDUCE_PROD());
  AddBuiltin(BuiltinOperator_REDUCE_MAX, Register_REDUCE_MAX());
  AddBuiltin(BuiltinOperator_REDUCE_MIN, Register_REDUCE_MIN());
  AddBuiltin(BuiltinOperator_EXPAND_DIMS, Register_EXPAND_DIMS());
  AddBuiltin(BuiltinOperator_SPARSE_TO_DENSE, Register_SPARSE_TO_DENSE());
  AddBuiltin(BuiltinOperator_EQUAL, Register_EQUAL());
  AddBuiltin(BuiltinOperator_NOT_EQUAL, Register_NOT_EQUAL());
  AddBuiltin(BuiltinOperator_SQRT, Register_SQRT());
  AddBuiltin(BuiltinOperator_RSQRT, Register_RSQRT());
  AddBuiltin(BuiltinOperator_SHAPE, Register_SHAPE());
  AddBuiltin(BuiltinOperator_POW, Register_POW());
  AddBuiltin(BuiltinOperator_FAKE_QUANT, Register_FAKE_QUANT(), 1, 2);
  AddBuiltin(BuiltinOperator_PACK, Register_PACK());
  AddBuiltin(BuiltinOperator_ONE_HOT, Register_ONE_HOT());
  AddBuiltin(BuiltinOperator_LOGICAL_OR, Register_LOGICAL_OR());
  AddBuiltin(BuiltinOperator_LOGICAL_AND, Register_LOGICAL_AND());
  AddBuiltin(BuiltinOperator_LOGICAL_NOT, Register_LOGICAL_NOT());
  AddBuiltin(BuiltinOperator_UNPACK, Register_UNPACK());

  // TODO(andrewharp, ahentz): Move these somewhere more appropriate so that
  // custom ops aren't always included by default.
  AddCustom("Mfcc", tflite::ops::custom::Register_MFCC());
  AddCustom("AudioSpectrogram",
            tflite::ops::custom::Register_AUDIO_SPECTROGRAM());
  AddCustom("TFLite_Detection_PostProcess",
            tflite::ops::custom::Register_DETECTION_POSTPROCESS());
}
};

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_REGISTER_H_
