#ifndef MODEL_H_
#define MODEL_H_

#include <iostream>
#include <memory>
#include "flatbuffers.h"
#include "model.h"
#include "allocation.h"
#include "error_reporter.h"
#include "schema_generated.h"
#ifndef TFLITE_MCU
#include "nnapi_delegate.h"
#endif

using namespace std;

namespace tflite{

class TfLiteVerifier {
  public:
    virtual bool Verify(const char* data, int length,
                        tflite::ErrorReporter* reporter) = 0;
    virtual ~TfLiteVerifier() {}
};



class FlatBufferModel {
 public:
  /*LiYu*/
  static std::unique_ptr<FlatBufferModel> BuildFromFile(
      const char* filename,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  static std::unique_ptr<FlatBufferModel> VerifyAndBuildFromFile(const char* filename, 
    TfLiteVerifier* verifier ,
    ErrorReporter* error_reporter);

  FlatBufferModel(const FlatBufferModel&) = delete;
  // FlatBufferModel(Allocation* allocation,
                                 // ErrorReporter* error_reporter);
  // FlatBufferModel(Allocation* allocation,
  //                                ErrorReporter* error_reporter)
  //   : error_reporter_(ValidateErrorReporter(error_reporter));
  
  // ErrorReporter* error_reporter() const;
  ErrorReporter* error_reporter() const { return error_reporter_; }
  // bool CheckModelIdentifier() const;
  bool CheckModelIdentifier() const;

  bool initialized() const { return model_ != nullptr; }

  private:
    FlatBufferModel(Allocation* allocation,
                  ErrorReporter* error_reporter = DefaultErrorReporter());

    FlatBufferModel(const Model* model, ErrorReporter* error_reporter);

    ErrorReporter* error_reporter_;
    Allocation* allocation_ = nullptr;
    const tflite::Model* model_ = nullptr;
};
}  // namespace tflite




#endif  // TENSORFLOW_CONTRIB_LITE_MODEL_H_
