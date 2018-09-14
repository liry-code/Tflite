#ifndef CONTEXT_H_
#define CONTEXT_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum { kTfLiteOk = 0, kTfLiteError = 1 } TfLiteStatus;


typedef enum {
  kTfLiteEigenContext = 0,     // include eigen_support.h to use.
  kTfLiteGemmLowpContext = 1,  // include gemm_support.h to use.
  kTfLiteEdgeTpuContext = 2,   // Placeholder for Edge TPU support.
  kTfLiteMaxExternalContexts = 3
} TfLiteExternalContextType;


typedef struct {
  TfLiteExternalContextType type;
  TfLiteStatus (*Refresh)(struct TfLiteContext* context);
} TfLiteExternalContext;

// Forward declare so GetNode can use this is in Context.
typedef struct _TfLiteRegistration TfLiteRegistration;
typedef struct _TfLiteDelegate TfLiteDelegate;

#define kOptionalTensor (-1)

// Fixed size list of integers. Used for dimensions and inputs/outputs tensor
// indices
typedef struct {
  int size;
// gcc 6.1+ have a bug where flexible members aren't properly handled
// https://github.com/google/re2/commit/b94b7cd42e9f02673cd748c1ac1d16db4052514c
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && \
    __GNUC_MINOR__ >= 1
  int data[0];
#else
  int data[];
#endif
} TfLiteIntArray;

// Given the size (number of elements) in a TfLiteIntArray, calculate its size
// in bytes.
// int TfLiteIntArrayGetSizeInBytes(int size);
int TfLiteIntArrayGetSizeInBytes(int size);

// Create a array of a given `size` (uninitialized entries).
// This returns a pointer, that you must free using TfLiteIntArrayFree().
// TfLiteIntArray* TfLiteIntArrayCreate(int size);
TfLiteIntArray* TfLiteIntArrayCreate(int size);

// Check if two tensors are equal. Returns 1 if they are equal, 0 otherwise.
int TfLiteIntArrayEqual(TfLiteIntArray* a, TfLiteIntArray* b);

// Create a copy of an array passed as `src`.
// You are expected to free memory with TfLiteIntArrayFree
TfLiteIntArray* TfLiteIntArrayCopy(TfLiteIntArray* src);

// Free memory of array `v`.
// void TfLiteIntArrayFree(TfLiteIntArray* v);
void TfLiteIntArrayFree(TfLiteIntArray* v);


#define TF_LITE_ENSURE_MSG(context, value, msg)            \
  do {                                                     \
    if (!(value)) {                                        \
      (context)->ReportError((context), __FILE__ " " msg); \
      return kTfLiteError;                                 \
    }                                                      \
  } while (0)


#define TF_LITE_ENSURE(context, a)                                          \
  do {                                                                      \
    if (!(a)) {                                                             \
      (context)->ReportError((context), "%s:%d %s was not true.", __FILE__, \
                             __LINE__, #a);                                 \
      return kTfLiteError;                                                  \
    }                                                                       \
  } while (0)

#define TF_LITE_ENSURE_STATUS(a) \
  do {                           \
    if ((a) != kTfLiteOk) {      \
      return kTfLiteError;       \
    }                            \
  } while (0)


#define TF_LITE_ENSURE_EQ(context, a, b)                                       \
  do {                                                                         \
    if ((a) != (b)) {                                                          \
      (context)->ReportError((context), "%s:%d %s != %s (%d != %d)", __FILE__, \
                             __LINE__, #a, #b, (a), (b));                      \
      return kTfLiteError;                                                     \
    }                                                                          \
  } while (0)

#define TF_LITE_ENSURE_OK(context, status) \
  do {                                     \
    if ((status) != kTfLiteOk) {           \
      return status;                       \
    }                                      \
  } while (0)

// Single-precision complex data type compatible with the C99 definition.
typedef struct {
  float re, im;  // real and imaginary parts, respectively.
} TfLiteComplex64;

// Types supported by tensor
typedef enum {
  kTfLiteNoType = 0,
  kTfLiteFloat32 = 1,
  kTfLiteInt32 = 2,
  kTfLiteUInt8 = 3,
  kTfLiteInt64 = 4,
  kTfLiteString = 5,
  kTfLiteBool = 6,
  kTfLiteInt16 = 7,
  kTfLiteComplex64 = 8,
} TfLiteType;


typedef struct {
  float scale;
  int32_t zero_point;
} TfLiteQuantizationParams;

typedef union {
  int* i32;
  int64_t* i64;
  float* f;
  char* raw;
  const char* raw_const;
  uint8_t* uint8;
  bool* b;
  int16_t* i16;
  TfLiteComplex64* c64;
} TfLitePtrUnion;

typedef enum {
  kTfLiteMemNone = 0,
  kTfLiteMmapRo,
  kTfLiteArenaRw,
  kTfLiteArenaRwPersistent,
  kTfLiteDynamic,
} TfLiteAllocationType;

typedef int TfLiteBufferHandle;
const TfLiteBufferHandle kTfLiteNullBufferHandle = -1;

typedef struct {
  TfLiteType type;
  TfLitePtrUnion data;
  TfLiteIntArray* dims;
  TfLiteQuantizationParams params;
  TfLiteAllocationType allocation_type;
  size_t bytes;

  const void* allocation;

  const char* name;

  TfLiteDelegate* delegate;

  TfLiteBufferHandle buffer_handle;

  bool data_is_stale;

  bool is_variable;
} TfLiteTensor;



void TfLiteTensorDataFree(TfLiteTensor* t);

// Free memory of tensor `t`;
void TfLiteTensorFree(TfLiteTensor* t);

// Set all of a tensor's fields (and free any previously allocated data).
void TfLiteTensorReset(TfLiteType type, const char* name, TfLiteIntArray* dims,
                       TfLiteQuantizationParams quantization, char* buffer,
                       size_t size, TfLiteAllocationType allocation_type,
                       const void* allocation, bool is_variable,
                       TfLiteTensor* tensor);

// void TfLiteTensorRealloc(size_t num_bytes, TfLiteTensor* tensor);
void TfLiteTensorRealloc(size_t num_bytes, TfLiteTensor* tensor);

typedef struct {
  TfLiteIntArray* inputs;

  TfLiteIntArray* outputs;

  TfLiteIntArray* temporaries;

  void* user_data;

  void* builtin_data;

  const void* custom_initial_data;
  int custom_initial_data_size;

  TfLiteDelegate* delegate;
} TfLiteNode;

typedef struct TfLiteContext {
  // Number of tensors in the context.
  size_t tensors_size;

  TfLiteStatus (*GetExecutionPlan)(struct TfLiteContext* context,
                                   TfLiteIntArray** execution_plan);

  TfLiteTensor* tensors;

  void* impl_;

  TfLiteStatus (*ResizeTensor)(struct TfLiteContext*, TfLiteTensor* tensor,
                               TfLiteIntArray* new_size);
  void (*ReportError)(struct TfLiteContext*, const char* msg, ...);

  TfLiteStatus (*AddTensors)(struct TfLiteContext*, int tensors_to_add,
                             int* first_new_tensor_index);

  TfLiteStatus (*GetNodeAndRegistration)(struct TfLiteContext*, int node_index,
                                         TfLiteNode** node,
                                         TfLiteRegistration** registration);

  TfLiteStatus (*ReplaceSubgraphsWithDelegateKernels)(
      struct TfLiteContext*, TfLiteRegistration registration,
      const TfLiteIntArray* nodes_to_replace, TfLiteDelegate* delegate);

  int recommended_num_threads;

  TfLiteExternalContext* (*GetExternalContext)(struct TfLiteContext*,
                                               TfLiteExternalContextType);
  void (*SetExternalContext)(struct TfLiteContext*, TfLiteExternalContextType,
                             TfLiteExternalContext*);
} TfLiteContext;

typedef struct _TfLiteRegistration {
  void* (*init)(TfLiteContext* context, const char* buffer, size_t length);

  void (*free)(TfLiteContext* context, void* buffer);

  TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);

  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);

  const char* (*profiling_string)(const TfLiteContext* context,
                                  const TfLiteNode* node);

  int32_t builtin_code;

  const char* custom_name;

  int version;
} TfLiteRegistration;

typedef struct _TfLiteDelegate {
  void* data_;

  TfLiteStatus (*Prepare)(TfLiteContext* context, TfLiteDelegate* delegate);

  TfLiteStatus (*CopyFromBufferHandle)(TfLiteContext* context,
                                       TfLiteDelegate* delegate,
                                       TfLiteBufferHandle buffer_handle,
                                       void* data, size_t size);

  TfLiteStatus (*CopyToBufferHandle)(TfLiteContext* context,
                                     TfLiteDelegate* delegate,
                                     TfLiteBufferHandle buffer_handle,
                                     void* data, size_t size);

  void (*FreeBufferHandle)(TfLiteContext* context, TfLiteDelegate* delegate,
                           TfLiteBufferHandle* handle);
} TfLiteDelegate;

typedef struct {
  TfLiteDelegate* delegate;
  TfLiteIntArray* nodes_to_replace;
  TfLiteIntArray* input_tensors;
  TfLiteIntArray* output_tensors;
} TfLiteDelegateParams;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // TENSORFLOW_CONTRIB_LITE_CONTEXT_H_
