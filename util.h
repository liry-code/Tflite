#ifndef TENSORFLOW_CONTRIB_LITE_UTIL_H_
#define TENSORFLOW_CONTRIB_LITE_UTIL_H_

#include <vector>
#include "/home/tclxa/TfLite/context.h"

namespace tflite {

// The prefix of Eager op custom code.
// This will be matched agains the `custom_code` field in `OperatorCode`
// Flatbuffer Table.
// WARNING: This is an experimental API and subject to change.
constexpr char kEagerCustomCodePrefix[] = "Eager";

// Checks whether the prefix of the custom name indicates the operation is an
// Eager operation.
bool IsEagerOp(const char* custom_name);

// Converts a `std::vector` to a `TfLiteIntArray`. The caller takes ownership
// of the returned pointer.
TfLiteIntArray* ConvertVectorToTfLiteIntArray(const std::vector<int>& input);

// Converts an array (of the given size) to a `TfLiteIntArray`. The caller
// takes ownership of the returned pointer, and must make sure 'dims' has at
// least 'rank' elemnts.
TfLiteIntArray* ConvertArrayToTfLiteIntArray(const int rank, const int* dims);

// Checks whether a `TfLiteIntArray` and an int array have matching elements.
// The caller must guarantee that 'b' has at least 'b_size' elements.
bool EqualArrayAndTfLiteIntArray(const TfLiteIntArray* a, const int b_size,
                                 const int* b);

size_t CombineHashes(std::initializer_list<size_t> hashes);

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_UTIL_H_
