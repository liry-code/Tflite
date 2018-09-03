#ifndef OP_RESOLVER_H_
#define OP_RESOLVER_H_

#include <unordered_map>
#include "/home/tclxa/TfLite/context.h"
#include "/home/tclxa/TfLite/schema_generated.h"
#include "/home/tclxa/TfLite/util.h"

namespace tflite {

class OpResolver {
 public:
  // Finds the op registration for a builtin operator by enum code.
  // virtual const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
  //                                          int version) const = 0;
  // // Finds the op registration of a custom operator by op name.
  // virtual const TfLiteRegistration* FindOp(const char* op,
  //                                          int version) const = 0;
  // virtual ~OpResolver() {}
};

// Some versions of gcc doesn't support partial specialization in class scope,
// so these are defined in a namescope.
namespace op_resolver_hasher {
template <typename V>
struct ValueHasher {
  size_t operator()(const V& v) const { return std::hash<V>()(v); }
};

template <>
struct ValueHasher<tflite::BuiltinOperator> {
  size_t operator()(const tflite::BuiltinOperator& v) const {
    return std::hash<int>()(static_cast<int>(v));
  }
};

template <typename T>
struct OperatorKeyHasher {
  size_t operator()(const T& x) const {
    size_t a = ValueHasher<typename T::first_type>()(x.first);
    size_t b = ValueHasher<typename T::second_type>()(x.second);
    return CombineHashes({a, b});
  }
};

}  // namespace op_resolver_hasher


class MutableOpResolver : public OpResolver {
 public:
  // const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
  //                                  int version) const override;
  // const TfLiteRegistration* FindOp(const char* op, int version) const override;
  // void AddBuiltin(tflite::BuiltinOperator op, TfLiteRegistration* registration,
  //                 int min_version = 1, int max_version = 1);
  void AddBuiltin(tflite::BuiltinOperator op, TfLiteRegistration* registration,
                  int min_version = 1, int max_version = 1){
    for (int version = min_version; version <= max_version; ++version) {
      TfLiteRegistration new_registration = *registration;
      new_registration.builtin_code = op;
      new_registration.version = version;
      auto op_key = std::make_pair(op, version);
      builtins_[op_key] = new_registration;
    }
  }
  // void AddCustom(const char* name, TfLiteRegistration* registration,
  //                int min_version = 1, int max_version = 1);

 private:
  typedef std::pair<tflite::BuiltinOperator, int> BuiltinOperatorKey;
  // typedef std::pair<std::string, int> CustomOperatorKey;

  std::unordered_map<BuiltinOperatorKey, TfLiteRegistration,
                     op_resolver_hasher::OperatorKeyHasher<BuiltinOperatorKey> >
      builtins_;
  // std::unordered_map<CustomOperatorKey, TfLiteRegistration,
  //                    op_resolver_hasher::OperatorKeyHasher<CustomOperatorKey> >
  //     custom_ops_;
};

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_OP_RESOLVER_H_
