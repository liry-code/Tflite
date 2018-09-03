#include "/home/tclxa/TfLite/op_resolver.h"
#include "/home/tclxa/TfLite/context.h"

namespace tflite {

// const TfLiteRegistration* MutableOpResolver::FindOp(tflite::BuiltinOperator op,
//                                                     int version) const {
//   auto it = builtins_.find(std::make_pair(op, version));
//   return it != builtins_.end() ? &it->second : nullptr;
// }

// const TfLiteRegistration* MutableOpResolver::FindOp(const char* op,
//                                                     int version) const {
//   auto it = custom_ops_.find(std::make_pair(op, version));
//   return it != custom_ops_.end() ? &it->second : nullptr;
// }

// void MutableOpResolver::AddBuiltin(tflite::BuiltinOperator op,
//                                    TfLiteRegistration* registration,
//                                    int min_version, int max_version) {
//   for (int version = min_version; version <= max_version; ++version) {
//     TfLiteRegistration new_registration = *registration;
//     new_registration.builtin_code = op;
//     new_registration.version = version;
//     auto op_key = std::make_pair(op, version);
//     builtins_[op_key] = new_registration;
//   }
// }

// void MutableOpResolver::AddCustom(const char* name,
//                                   TfLiteRegistration* registration,
//                                   int min_version, int max_version) {
//   for (int version = min_version; version <= max_version; ++version) {
//     TfLiteRegistration new_registration = *registration;
//     new_registration.builtin_code = BuiltinOperator_CUSTOM;
//     new_registration.version = version;
//     auto op_key = std::make_pair(name, version);
//     custom_ops_[op_key] = new_registration;
//   }
// }

}  // namespace tflite
