#ifndef MODEL_H_
#define MODEL_H_

#include <iostream>
#include <memory>
// #include "/home/tclxa/TfLite/model.h"
#include "/home/tclxa/TfLite/allocation.h"
#include "/home/tclxa/TfLite/error_reporter.h"
// #include "/home/tclxa/TfLite/schema_generated.h"
#ifndef TFLITE_MCU
// #include "/home/tclxa/TfLite/nnapi_delegate.h"
#endif

using namespace std;


namespace tflite {

class TfLiteVerifier {
 public:
  virtual bool Verify(const char* data, int length,
                      ErrorReporter* reporter) = 0;
  virtual ~TfLiteVerifier() {}
};



class FlatBufferModel {
 public:
  // static std::unique_ptr<FlatBufferModel> BuildFromFile(
  //     const char* filename,
  //     ErrorReporter* error_reporter = DefaultErrorReporter());

  static std::unique_ptr<FlatBufferModel> VerifyAndBuildFromFile(
      const char* filename, TfLiteVerifier* verifier = nullptr,
      ErrorReporter* error_reporter = DefaultErrorReporter()){
    cout << "VerifyAndBuildFromFile" << endl;
    return nullptr;
  }

  // static std::unique_ptr<FlatBufferModel> BuildFromBuffer(
  //     const char* buffer, size_t buffer_size,
  //     ErrorReporter* error_reporter = DefaultErrorReporter());

  // lptr in case of failure.
  // static std::unique_ptr<FlatBufferModel> BuildFromModel(
  //     const tflite::Model* model_spec,
  //     ErrorReporter* error_reporter = DefaultErrorReporter());

  // // Releases memory or unmaps mmaped memory.
  // ~FlatBufferModel();

  // Copying or assignment is disallowed to simplify ownership semantics.
  FlatBufferModel(const FlatBufferModel&) = delete;
  // FlatBufferModel& operator=(const FlatBufferModel&) = delete;

  // bool initialized() const { return model_ != nullptr; }
  // const tflite::Model* operator->() const { return model_; }
  // const tflite::Model* GetModel() const { return model_; }
  // ErrorReporter* error_reporter() const { return error_reporter_; }
  // const Allocation* allocation() const { return allocation_; }

  // Returns true if the model identifier is correct (otherwise false and
  // reports an error).
  bool CheckModelIdentifier() const;

 private:
  // eporter instance.
  // FlatBufferModel(Allocation* allocation,
  //                 ErrorReporter* error_reporter = DefaultErrorReporter());

  // FlatBufferModel(const Model* model, ErrorReporter* error_reporter);

  // const tflite::Model* model_ = nullptr;
  // ErrorReporter* error_reporter_;
  // Allocation* allocation_ = nullptr;
};
}  // namespace tflite


// std::unique_ptr<FlatBufferModel> FlatBufferModel::VerifyAndBuildFromFile(
//     const char* filename, TfLiteVerifier* verifier,
//     ErrorReporter* error_reporter) {
//   cout << "VerifyAndBuildFromFile" << endl;
//   error_reporter = ValidateErrorReporter(error_reporter);

//   std::unique_ptr<FlatBufferModel> model;
//   auto allocation = GetAllocationFromFile(filename, /*mmap_file=*/true,
//                                           error_reporter, /*use_nnapi=*/true);
//   if (verifier &&
//       !verifier->Verify(static_cast<const char*>(allocation->base()),
//                         allocation->bytes(), error_reporter)) {
//     return model;
//   }
//   model.reset(new FlatBufferModel(allocation.release(), error_reporter));
//   if (!model->initialized()) model.reset();
//   return nullptr;
// }


// // Loads a model from `filename`. If `mmap_file` is true then use mmap,
// // otherwise make a copy of the model in a buffer.
// std::unique_ptr<Allocation> GetAllocationFromFile(const char* filename,
//                                                   bool mmap_file,
//                                                   ErrorReporter* error_reporter,
//                                                   bool use_nnapi) {
//   std::unique_ptr<Allocation> allocation;
//   if (mmap_file && MMAPAllocation::IsSupported()) {
//     if (use_nnapi && NNAPIDelegate::IsSupported())
//       allocation.reset(new NNAPIAllocation(filename, error_reporter));
//     else
//       allocation.reset(new MMAPAllocation(filename, error_reporter));
//   } else {
//     allocation.reset(new FileCopyAllocation(filename, error_reporter));
//   }
//   return allocation;
// }





#endif  // TENSORFLOW_CONTRIB_LITE_MODEL_H_
