#ifndef MODEL_H_
#define MODEL_H_

#include <iostream>
#include <memory>
#include "/home/tclxa/TfLite/flatbuffers.h"
#include "/home/tclxa/TfLite/model.h"
#include "/home/tclxa/TfLite/allocation.h"
#include "/home/tclxa/TfLite/error_reporter.h"
#include "/home/tclxa/TfLite/schema_generated.h"
#ifndef TFLITE_MCU
#include "/home/tclxa/TfLite/nnapi_delegate.h"
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
  //   allocation.reset(new FileCopyAllocation(filename, error_reporter));
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
  static std::unique_ptr<FlatBufferModel> VerifyAndBuildFromFile(const char* filename, 
    tflite::TfLiteVerifier* verifier = nullptr,
    tflite::ErrorReporter* error_reporter = tflite::DefaultErrorReporter()){

    error_reporter = ValidateErrorReporter(error_reporter);
    std::unique_ptr<FlatBufferModel> model;

    auto allocation = GetAllocationFromFile(filename, /*mmap_file=*/true,
                                          error_reporter, /*use_nnapi=*/true);
  // if (verifier &&
  //     !verifier->Verify(static_cast<const char*>(allocation->base()),
  //                       allocation->bytes(), error_reporter)) {
  //   return model;
  // }
  // model.reset(new FlatBufferModel(allocation.release(), error_reporter));
  // if (!model->initialized()) model.reset();
  // return model;

    cout << "model  VerifyAndBuildFromFile ..." << endl;
    return nullptr;
  }

  FlatBufferModel(const FlatBufferModel&) = delete;
  
  bool CheckModelIdentifier() const;

 private:
  
};









}  // namespace tflite




#endif  // TENSORFLOW_CONTRIB_LITE_MODEL_H_
