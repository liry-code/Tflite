#ifndef JNIFLATBUFFERVERIFIER_H
#define JNIFLATBUFFERVERIFIER_H


#include "/home/tclxa/TfLite/flatbuffers.h"
#include "/home/tclxa/TfLite/schema_generated.h"

/**LiYu*/
// Verifies whether the model is a flatbuffer file.
class JNIFlatBufferVerifier :  tflite::TfLiteVerifier{
 public:
	bool Verify(const char* data, int length,
		tflite::ErrorReporter* reporter) override {
		if (!VerifyModel(data, length)) {
			reporter->Report("The model is not a valid Flatbuffer file");
			return false;
		}
		return true;
	}

	// TODO(yichengfan): evaluate the benefit to use tflite verifier.
	bool VerifyModel(const void* buf, size_t len) {
		flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buf), len);
		return tflite::VerifyModelBuffer(verifier);
	}
};



#endif