#ifndef ROUND_H_
#define ROUND_H_

#include <cmath>

namespace tflite {

// TODO(aselle): See if we can do this only on jdk. Also mikecase, check
// if you need this for java host build.
#if defined(__ANDROID__) && !defined(__NDK_MAJOR__)
template <class T>
inline float TfLiteRound(const float x) {
  return ::round(x);
}
inline double TfLiteRound(const double x) { return ::round(x); }
#else
template <class T>
inline T TfLiteRound(const T x) {
  return std::round(x);
}
#endif

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_ROUND_H_
