//#include <stdarg.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include <string>
//#include <iostream>
#include "exception_jni.h"

using namespace std;

/** LiYu*/
BufferErrorReporter::BufferErrorReporter(string exceptInfo){cout << exceptInfo << endl;}; 

// virtual ~BufferErrorReporter();
int BufferErrorReporter::Report(const char* format, va_list args){
    int size = 0;
    if (start_idx_ < end_idx_) {
        size = vsnprintf(buffer_ + start_idx_, end_idx_ - start_idx_, format, args);
      }
    start_idx_ += size;
    return size;
}

