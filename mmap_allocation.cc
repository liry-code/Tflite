#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "allocation.h"
#include "error_reporter.h"

namespace tflite {

/**
MMAPAllocation::MMAPAllocation(const char* filename,
                               ErrorReporter* error_reporter)
    : Allocation(error_reporter), mmapped_buffer_(MAP_FAILED) {
  mmap_fd_ = open(filename, O_RDONLY);
  if (mmap_fd_ == -1) {
    error_reporter_->Report("Could not open '%s'.", filename);
    return;
  }
  struct stat sb;
  fstat(mmap_fd_, &sb);
  buffer_size_bytes_ = sb.st_size;
  mmapped_buffer_ =
      mmap(nullptr, buffer_size_bytes_, PROT_READ, MAP_SHARED, mmap_fd_, 0);
  if (mmapped_buffer_ == MAP_FAILED) {
    error_reporter_->Report("Mmap of '%s' failed.", filename);
    return;
  }
}

MMAPAllocation::~MMAPAllocation() {
  if (valid()) {
    munmap(const_cast<void*>(mmapped_buffer_), buffer_size_bytes_);
  }
  if (mmap_fd_ != -1) close(mmap_fd_);
}

const void* MMAPAllocation::base() const { return mmapped_buffer_; }

size_t MMAPAllocation::bytes() const { return buffer_size_bytes_; }

bool MMAPAllocation::valid() const { return mmapped_buffer_ != MAP_FAILED; }

bool MMAPAllocation::IsSupported() { return true; }


*/
}  // namespace tflite
