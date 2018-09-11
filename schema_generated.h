#ifndef SCHEMA_TFLITE_H_
#define SCHEMA_TFLITE_H_

#include <iostream>
#include <string>
#include "flatbuffers.h"

using namespace std;

namespace tflite {

struct Model FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
	/**
  typedef ModelT NativeTableType;
  enum {
    VT_VERSION = 4,
    VT_OPERATOR_CODES = 6,
    VT_SUBGRAPHS = 8,
    VT_DESCRIPTION = 10,
    VT_BUFFERS = 12,
    VT_METADATA_BUFFER = 14
  };
  uint32_t version() const {
    return GetField<uint32_t>(VT_VERSION, 0);
  }
  const flatbuffers::Vector<flatbuffers::Offset<OperatorCode>> *operator_codes() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<OperatorCode>> *>(VT_OPERATOR_CODES);
  }
  const flatbuffers::Vector<flatbuffers::Offset<SubGraph>> *subgraphs() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<SubGraph>> *>(VT_SUBGRAPHS);
  }
  const flatbuffers::String *description() const {
    return GetPointer<const flatbuffers::String *>(VT_DESCRIPTION);
  }
  const flatbuffers::Vector<flatbuffers::Offset<Buffer>> *buffers() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<Buffer>> *>(VT_BUFFERS);
  }
  const flatbuffers::Vector<int32_t> *metadata_buffer() const {
    return GetPointer<const flatbuffers::Vector<int32_t> *>(VT_METADATA_BUFFER);
  }
  ModelT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(ModelT *_o, const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<Model> Pack(flatbuffers::FlatBufferBuilder &_fbb, const ModelT* _o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);
  
  */


  bool Verify(flatbuffers::Verifier &verifier) const {
    // return VerifyTableStart(verifier) &&
    //        VerifyField<uint32_t>(verifier, VT_VERSION) &&
    //        VerifyOffset(verifier, VT_OPERATOR_CODES) &&
    //        verifier.Verify(operator_codes()) &&
    //        verifier.VerifyVectorOfTables(operator_codes()) &&
    //        VerifyOffset(verifier, VT_SUBGRAPHS) &&
    //        verifier.Verify(subgraphs()) &&
    //        verifier.VerifyVectorOfTables(subgraphs()) &&
    //        VerifyOffset(verifier, VT_DESCRIPTION) &&
    //        verifier.Verify(description()) &&
    //        VerifyOffset(verifier, VT_BUFFERS) &&
    //        verifier.Verify(buffers()) &&
    //        verifier.VerifyVectorOfTables(buffers()) &&
    //        VerifyOffset(verifier, VT_METADATA_BUFFER) &&
    //        verifier.Verify(metadata_buffer()) &&
    //        verifier.EndTable();
    cout << "schema_generated verify function ...." << endl;
    return true;
  }
};


inline const char *ModelIdentifier() {
  return "TFL3";
}

inline bool VerifyModelBuffer(
    flatbuffers::Verifier &verifier) {
	cout << "schema_generated verifymodelbuffer ... " << endl;
	return verifier.VerifyBuffer<tflite::Model>(ModelIdentifier());
}

inline const tflite::Model *GetModel(const void *buf) {
  return flatbuffers::GetRoot<tflite::Model>(buf);
}

inline bool ModelBufferHasIdentifier(const void *buf) {
  return flatbuffers::BufferHasIdentifier(
      buf, ModelIdentifier());
}


enum BuiltinOperator {
  BuiltinOperator_ADD = 0,
  BuiltinOperator_AVERAGE_POOL_2D = 1,
  BuiltinOperator_CONCATENATION = 2,
  BuiltinOperator_CONV_2D = 3,
  BuiltinOperator_DEPTHWISE_CONV_2D = 4,
  BuiltinOperator_DEQUANTIZE = 6,
  BuiltinOperator_EMBEDDING_LOOKUP = 7,
  BuiltinOperator_FLOOR = 8,
  BuiltinOperator_FULLY_CONNECTED = 9,
  BuiltinOperator_HASHTABLE_LOOKUP = 10,
  BuiltinOperator_L2_NORMALIZATION = 11,
  BuiltinOperator_L2_POOL_2D = 12,
  BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION = 13,
  BuiltinOperator_LOGISTIC = 14,
  BuiltinOperator_LSH_PROJECTION = 15,
  BuiltinOperator_LSTM = 16,
  BuiltinOperator_MAX_POOL_2D = 17,
  BuiltinOperator_MUL = 18,
  BuiltinOperator_RELU = 19,
  BuiltinOperator_RELU_N1_TO_1 = 20,
  BuiltinOperator_RELU6 = 21,
  BuiltinOperator_RESHAPE = 22,
  BuiltinOperator_RESIZE_BILINEAR = 23,
  BuiltinOperator_RNN = 24,
  BuiltinOperator_SOFTMAX = 25,
  BuiltinOperator_SPACE_TO_DEPTH = 26,
  BuiltinOperator_SVDF = 27,
  BuiltinOperator_TANH = 28,
  BuiltinOperator_CONCAT_EMBEDDINGS = 29,
  BuiltinOperator_SKIP_GRAM = 30,
  BuiltinOperator_CALL = 31,
  BuiltinOperator_CUSTOM = 32,
  BuiltinOperator_EMBEDDING_LOOKUP_SPARSE = 33,
  BuiltinOperator_PAD = 34,
  BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN = 35,
  BuiltinOperator_GATHER = 36,
  BuiltinOperator_BATCH_TO_SPACE_ND = 37,
  BuiltinOperator_SPACE_TO_BATCH_ND = 38,
  BuiltinOperator_TRANSPOSE = 39,
  BuiltinOperator_MEAN = 40,
  BuiltinOperator_SUB = 41,
  BuiltinOperator_DIV = 42,
  BuiltinOperator_SQUEEZE = 43,
  BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM = 44,
  BuiltinOperator_STRIDED_SLICE = 45,
  BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN = 46,
  BuiltinOperator_EXP = 47,
  BuiltinOperator_TOPK_V2 = 48,
  BuiltinOperator_SPLIT = 49,
  BuiltinOperator_LOG_SOFTMAX = 50,
  BuiltinOperator_DELEGATE = 51,
  BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM = 52,
  BuiltinOperator_CAST = 53,
  BuiltinOperator_PRELU = 54,
  BuiltinOperator_MAXIMUM = 55,
  BuiltinOperator_ARG_MAX = 56,
  BuiltinOperator_MINIMUM = 57,
  BuiltinOperator_LESS = 58,
  BuiltinOperator_NEG = 59,
  BuiltinOperator_PADV2 = 60,
  BuiltinOperator_GREATER = 61,
  BuiltinOperator_GREATER_EQUAL = 62,
  BuiltinOperator_LESS_EQUAL = 63,
  BuiltinOperator_SELECT = 64,
  BuiltinOperator_SLICE = 65,
  BuiltinOperator_SIN = 66,
  BuiltinOperator_TRANSPOSE_CONV = 67,
  BuiltinOperator_SPARSE_TO_DENSE = 68,
  BuiltinOperator_TILE = 69,
  BuiltinOperator_EXPAND_DIMS = 70,
  BuiltinOperator_EQUAL = 71,
  BuiltinOperator_NOT_EQUAL = 72,
  BuiltinOperator_LOG = 73,
  BuiltinOperator_SUM = 74,
  BuiltinOperator_SQRT = 75,
  BuiltinOperator_RSQRT = 76,
  BuiltinOperator_SHAPE = 77,
  BuiltinOperator_POW = 78,
  BuiltinOperator_ARG_MIN = 79,
  BuiltinOperator_FAKE_QUANT = 80,
  BuiltinOperator_REDUCE_PROD = 81,
  BuiltinOperator_REDUCE_MAX = 82,
  BuiltinOperator_PACK = 83,
  BuiltinOperator_LOGICAL_OR = 84,
  BuiltinOperator_ONE_HOT = 85,
  BuiltinOperator_LOGICAL_AND = 86,
  BuiltinOperator_LOGICAL_NOT = 87,
  BuiltinOperator_UNPACK = 88,
  BuiltinOperator_REDUCE_MIN = 89,
  BuiltinOperator_FLOOR_DIV = 90,
  BuiltinOperator_REDUCE_ANY = 91,
  BuiltinOperator_MIN = BuiltinOperator_ADD,
  BuiltinOperator_MAX = BuiltinOperator_REDUCE_ANY
};


}  //namespace

#endif