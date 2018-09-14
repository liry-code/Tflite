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
#include "model.h"
// #include "allocation.h"
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

namespace tflite {
namespace {

	ErrorReporter* ValidateErrorReporter(ErrorReporter* e) {
		return e ? e : DefaultErrorReporter();
	}
}

std::unique_ptr<tflite::Allocation> GetAllocationFromFile(const char* filename,
                                                  bool mmap_file,
                                                  tflite::ErrorReporter* error_reporter,
                                                  bool use_nnapi) {
	std::unique_ptr<tflite::Allocation> allocation;

	/*LiYu  (源码)*/
	if (mmap_file) {
		if (use_nnapi && NNAPIExists()){
			allocation.reset(new NNAPIAllocation(filename, error_reporter));
			cout << "issuporter if " << endl;
		}else{
			allocation.reset(new MMAPAllocation(filename, error_reporter));
			cout << "Issupported else " << endl;
		}
	} else {
		allocation.reset(new FileCopyAllocation(filename, error_reporter));
		cout << "filecopyallocation else " << endl;
	}

	cout << "GetAllocation ... " << endl;
	return allocation;


	// if (mmap_file) {
	// 	if (use_nnapi && tflite::NNAPIDelegate::IsSupported()){
	// 		allocation.reset(new NNAPIAllocation(filename, error_reporter));
	// 		cout << "issuporter if " << endl;
	// 	}
	// 	else{
	// 		allocation.reset(new MMAPAllocation(filename, error_reporter));
	// 		cout << "Issupported else " << endl;
	// 	}
	// } else {
	// 	allocation.reset(new FileCopyAllocation(filename, error_reporter));
	// 	cout << "filecopyallocation else " << endl;
	// }
	// allocation.reset(new FileCopyAllocation(filename, error_reporter));
	// if(allocation ==  nullptr){
	// 	cout << "create Allocation faild ..." << endl;
	// }
	
}

/*LiYu*/
std::unique_ptr<FlatBufferModel> FlatBufferModel::VerifyAndBuildFromFile(const char* filename, 
tflite::TfLiteVerifier* verifier = nullptr,
tflite::ErrorReporter* error_reporter = DefaultErrorReporter()){
	cout << "model VerifyAndBuildFromFile ..." << endl;
	error_reporter = ValidateErrorReporter(error_reporter);
	std::unique_ptr<tflite::FlatBufferModel> model;

	auto allocation = GetAllocationFromFile(filename, /*mmap_file=*/true,
	                                      error_reporter, /*use_nnapi=*/true);
	if (verifier &&
	    !verifier->Verify(static_cast<const char*>(allocation->base()),
	                      allocation->bytes(), error_reporter)) {
	  	return model;
	}

	if(model == nullptr){
		cout << "model.cc model is null" << endl;
	}
	model.reset(new FlatBufferModel(allocation.release(), error_reporter));
	if (!model->initialized()) model.reset();
	return model;
}


FlatBufferModel::FlatBufferModel(const Model* model,
                                 ErrorReporter* error_reporter)
    : error_reporter_(ValidateErrorReporter(error_reporter)) {
  model_ = model;
}

FlatBufferModel::FlatBufferModel(Allocation* allocation,
                                 ErrorReporter* error_reporter)
    : error_reporter_(ValidateErrorReporter(error_reporter)) {
  allocation_ = allocation;
  if (!allocation_->valid() || !CheckModelIdentifier()) return;

  model_ = ::tflite::GetModel(allocation_->base());
}



bool FlatBufferModel::CheckModelIdentifier() const {
	if (!tflite::ModelBufferHasIdentifier(allocation_->base())) {
		const char* ident = flatbuffers::GetBufferIdentifier(allocation_->base());
		error_reporter_->Report(
		  "Model provided has model identifier '%c%c%c%c', should be '%s'\n",
		  ident[0], ident[1], ident[2], ident[3], tflite::ModelIdentifier());
		return false;
	}
	return true;
}

} // namespace tflite

