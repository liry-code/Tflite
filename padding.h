#ifndef PADDING_H_
#define PADDING_H_

#include "/home/tclxa/TfLite/builtin_op_data.h"

namespace tflite {

inline int ComputePadding(int stride, int dilation_rate, int in_size,
                          int filter_size, int out_size) {
  int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  int padding = ((out_size - 1) * stride + effective_filter_size - in_size) / 2;
  return padding > 0 ? padding : 0;
}

// Matching GetWindowedOutputSize in TensorFlow.
inline int ComputeOutSize(TfLitePadding padding, int image_size,
                          int filter_size, int stride) {
  switch (padding) {
    case kTfLitePaddingSame:
      return (image_size + stride - 1) / stride;
    case kTfLitePaddingValid:
      return (image_size + stride - filter_size) / stride;
    default:
      return 0;
  }
}

inline TfLitePaddingValues ComputePaddingHeightWidth(
    int stride_height, int stride_width, int dilation_rate, int in_height,
    int in_width, int filter_height, int filter_width, TfLitePadding padding) {
  int out_width = ComputeOutSize(padding, in_width, filter_width, stride_width);
  int out_height =
      ComputeOutSize(padding, in_height, filter_height, stride_height);

  TfLitePaddingValues padding_values;
  padding_values.height =
      ComputePadding(stride_height, 1, in_height, filter_height, out_height);
  padding_values.width =
      ComputePadding(stride_width, 1, in_width, filter_width, out_width);
  return padding_values;
}
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_PADDING_H_
