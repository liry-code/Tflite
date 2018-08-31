// #include "/home/tclxa/TfLite/JNIFlatBufferVerifier.h"

// bool JNIFlatBufferVerifier::Verify(const char* data, int length,
// 		tflite::ErrorReporter* reporter){
// 	if (!VerifyModel(data, length)) {
// 		reporter->Report("The model is not a valid Flatbuffer file");
// 		return false;
// 	}
// 	return true;
// }


// bool JNIFlatBufferVerifier::VerifyModel(const void* buf, size_t len){
// 	flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buf), len);
// 	return tflite::VerifyModelBuffer(verifier);
// }

