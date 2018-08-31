#ifndef NEURALNETWORKSSHIM_H_
#define NEURALNETWORKSSHIM_H_

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define NNAPI_LOG(format, ...) fprintf(stderr, format "\n", __VA_ARGS__);
#define LOAD_FUNCTION(name) \
  static name##_fn fn = reinterpret_cast<name##_fn>(loadFunction(#name));

#define EXECUTE_FUNCTION_RETURN(...) return fn != nullptr ? fn(__VA_ARGS__) : 0;

typedef struct ANeuralNetworksMemory ANeuralNetworksMemory;

inline void* loadLibrary(const char* name) {
  // TODO: change RTLD_LOCAL? Assumes there can be multiple instances of nn
  // api RT
  void* handle = nullptr;
#ifdef __ANDROID__
  handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr) {
    NNAPI_LOG("nnapi error: unable to open library %s", name);
  }
#endif
  return handle;
}


inline void* getLibraryHandle() {
  static void* handle = loadLibrary("libneuralnetworks.so");
  return handle;
}

inline void* loadFunction(const char* name) {
  void* fn = nullptr;
  if (getLibraryHandle() != nullptr) {
    fn = dlsym(getLibraryHandle(), name);
  }
  if (fn == nullptr) {
    NNAPI_LOG("nnapi error: unable to open function %s", name);
  }
  return fn;
}




inline bool NNAPIExists() {
  static bool nnapi_is_available = getLibraryHandle();
  return nnapi_is_available;
}

inline int ANeuralNetworksMemory_createFromFd(size_t size, int protect, int fd,
                                              size_t offset,
                                              ANeuralNetworksMemory** memory) {
  LOAD_FUNCTION(ANeuralNetworksMemory_createFromFd);
  EXECUTE_FUNCTION_RETURN(size, protect, fd, offset, memory);
}

#endif  // NEURALNETWORKSSHIM_H_
