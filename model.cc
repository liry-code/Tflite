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
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "model.h"

namespace tflite {

// namespace {
// // Ensure that ErrorReporter is non-null.
// ErrorReporter* ValidateErrorReporter(ErrorReporter* e) {
//   return e ? e : DefaultErrorReporter();
// }
// }  // namespace


// bool JNIFlatBufferVerifier::Verify(const char* data, int length,
// 		tflite::ErrorReporter* reporter) ;
// {
// 	if (!VerifyModel(data, length)) {
// 		reporter->Report("The model is not a valid Flatbuffer file");
// 		return false;
// 	}
// 	return true;
// }


// bool JNIFlatBufferVerifier::VerifyModel(const void* buf, size_t len){}
// {
// 	flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buf), len);
// 	return VerifyModelBuffer(verifier);
// }


// std::unique_ptr<FlatBufferModel> FlatBufferModel::VerifyAndBuildFromFile(){}
    // const char* filename, 
    // JNIFlatBufferVerifier* verifier,
    // ErrorReporter* error_reporter
    // ) {
  // error_reporter = ValidateErrorReporter(error_reporter);

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
	// cout << "model  VerifyAndBuildFromFile  cc ..." << endl;
  // return nullptr;
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


}  // namespace tflite
