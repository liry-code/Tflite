#include "allocation.h"
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <utility>
#include "context.h"
#include "error_reporter.h"
using namespace std;

namespace tflite {

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

const void* MMAPAllocation::base() const { return mmapped_buffer_; }

size_t MMAPAllocation::bytes() const { return buffer_size_bytes_; }

bool MMAPAllocation::valid() const { return mmapped_buffer_ != MAP_FAILED; }



FileCopyAllocation::FileCopyAllocation(const char* filename, ErrorReporter* error_reporter): Allocation(error_reporter) {
  // Obtain the file size, using an alternative method that is does not
  // require fstat for more compatibility.
  std::unique_ptr<FILE, decltype(&fclose)> file(fopen(filename, "rb"), fclose);
  if (!file) {
    error_reporter_->Report("Could not open '%s'.", filename);
    return;
  }
  // TODO(ahentz): Why did you think using fseek here was better for finding
  // the size?
  struct stat sb;
  if (fstat(fileno(file.get()), &sb) != 0) {
    error_reporter_->Report("Failed to get file size of '%s'.", filename);
    return;
  }
  buffer_size_bytes_ = sb.st_size;
  std::unique_ptr<char[]> buffer(new char[buffer_size_bytes_]);
  if (!buffer) {
    error_reporter_->Report("Malloc of buffer to hold copy of '%s' failed.",
                            filename);
    return;
  }
  size_t bytes_read =
      fread(buffer.get(), sizeof(char), buffer_size_bytes_, file.get());
  if (bytes_read != buffer_size_bytes_) {
    error_reporter_->Report("Read of '%s' failed (too few bytes read).",
                            filename);
    return;
  }
  // Versions of GCC before 6.2.0 don't support std::move from non-const
  // char[] to const char[] unique_ptrs.
  copied_buffer_.reset(const_cast<char const*>(buffer.release()));
}

const void* FileCopyAllocation::base() const { return copied_buffer_.get(); }

size_t FileCopyAllocation::bytes() const { return buffer_size_bytes_; }

bool FileCopyAllocation::valid() const { return copied_buffer_ != nullptr; }

}  // namespace tflite
