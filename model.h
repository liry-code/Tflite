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


namespace tflite {

ErrorReporter* ValidateErrorReporter(ErrorReporter* e) {
  return e ? e : DefaultErrorReporter();
}


std::unique_ptr<Allocation> GetAllocationFromFile(const char* filename,
                                                  bool mmap_file,
                                                  ErrorReporter* error_reporter,
                                                  bool use_nnapi) {
  std::unique_ptr<Allocation> allocation;
  if (mmap_file && tflite::MMAPAllocation::IsSupported()) {
    if (use_nnapi && tflite::NNAPIDelegate::IsSupported()){
      allocation.reset(new NNAPIAllocation(filename, error_reporter));
      cout << "issuporter if " << endl;
    }
    else{
      allocation.reset(new MMAPAllocation(filename, error_reporter));
      cout << "Issupported else " << endl;
    }
  } else {
    allocation.reset(new FileCopyAllocation(filename, error_reporter));
    cout << "filecopyallocation else " << endl;
  }
  allocation.reset(new FileCopyAllocation(filename, error_reporter));
  if(allocation ==  nullptr){
    cout << "create Allocation faild ..." << endl;
  }
  cout << "GetAllocation ... " << endl;
  return allocation;
}

class TfLiteVerifier {
 public:
  virtual bool Verify(const char* data, int length,
                      tflite::ErrorReporter* reporter) = 0;
  virtual ~TfLiteVerifier() {}
};



class FlatBufferModel {
 public:
  /*LiYu*/
  // static std::unique_ptr<FlatBufferModel> VerifyAndBuildFromFile(
      // const char* filename, TfLiteVerifier* verifier = nullptr,
      // ErrorReporter* error_reporter = DefaultErrorReporter());
  static std::unique_ptr<FlatBufferModel> VerifyAndBuildFromFile(const char* filename, 
    tflite::TfLiteVerifier* verifier = nullptr,
    tflite::ErrorReporter* error_reporter = tflite::DefaultErrorReporter()){
    cout << "model VerifyAndBuildFromFile ..." << endl;
    error_reporter = ValidateErrorReporter(error_reporter);
    std::unique_ptr<FlatBufferModel> model;

    auto allocation = GetAllocationFromFile(filename, /*mmap_file=*/true,
                                          error_reporter, /*use_nnapi=*/true);
    if (verifier &&
        !verifier->Verify(static_cast<const char*>(allocation->base()),
                          allocation->bytes(), error_reporter)) {
      return model;
    }
    model.reset(new FlatBufferModel(allocation.release(), error_reporter));
    if (!model->initialized()) model.reset();
    return model;
  }

  FlatBufferModel(const FlatBufferModel&) = delete;
  // FlatBufferModel(Allocation* allocation,
                                 // ErrorReporter* error_reporter);
  FlatBufferModel(Allocation* allocation,
                                 ErrorReporter* error_reporter)
    : error_reporter_(ValidateErrorReporter(error_reporter)) {
    allocation_ = allocation;
    if (!allocation_->valid() || !CheckModelIdentifier()) return;

    model_ = GetModel(allocation_->base());
  }
  
  // ErrorReporter* error_reporter() const;
  ErrorReporter* error_reporter() const { return error_reporter_; }
  // bool CheckModelIdentifier() const;
  bool CheckModelIdentifier() const {
    if (!tflite::ModelBufferHasIdentifier(allocation_->base())) {
      const char* ident = flatbuffers::GetBufferIdentifier(allocation_->base());
      error_reporter_->Report(
          "Model provided has model identifier '%c%c%c%c', should be '%s'\n",
          ident[0], ident[1], ident[2], ident[3], tflite::ModelIdentifier());
      return false;
    }
    return true;
  }

  bool initialized() const { return model_ != nullptr; }

  private:
    ErrorReporter* error_reporter_;
    Allocation* allocation_ = nullptr;
    const tflite::Model* model_ = nullptr;
};









}  // namespace tflite




#endif  // TENSORFLOW_CONTRIB_LITE_MODEL_H_
