#ifndef SCHEMA_TFLITE_H_
#define SCHEMA_TFLITE_H_

#include <iostream>
#include <string>
#include "/home/tclxa/TfLite/flatbuffers.h"

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

}  //namespace

#endif