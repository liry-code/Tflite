#include "/home/tclxa/TfLite/register.h"

namespace tflite {

std::unique_ptr<OpResolver> CreateOpResolver() {  // NOLINT
  return std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver>(
      new tflite::ops::builtin::BuiltinOpResolver());
}

}  // namespace tflite
