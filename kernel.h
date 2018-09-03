#ifndef GEMMLOWP_INTERNAL_KERNEL_H_
#define GEMMLOWP_INTERNAL_KERNEL_H_

#include "/home/tclxa/TfLite/bit_depth.h"
#include "/home/tclxa/TfLite/common.h"

namespace gemmlowp {

enum class CellOrder { DepthMajor, WidthMajor, Diagonal };

// CellFormat describes how data is laid
// out in a cell. That is, a CellOrder together with actual dimensions.
template <int tWidth, int tDepth, CellOrder tOrder = CellOrder::DepthMajor>
struct CellFormat {
  static const int kWidth = tWidth;
  static const int kDepth = tDepth;
  static const CellOrder kOrder = tOrder;

  static const int kSize = kWidth * kDepth;
};

template <typename tCellFormat, int tCells>
struct KernelSideFormat {
  typedef tCellFormat Cell;
  static const int kCells = tCells;
  static const int kWidth = kCells * Cell::kWidth;
  static const int kDepth = Cell::kDepth;
  typedef std::uint8_t Scalar;
};

template <typename tCellFormat, int tCells>
struct KernelSideFormatInt8 : KernelSideFormat<tCellFormat, tCells> {
  typedef std::int8_t Scalar;
};

// KernelFormat describes fully the input data layout that a kernel expects.
// It consists of two KernelSideFormat's, one for LHS and one for RHS.
template <typename tLhs, typename tRhs>
struct KernelFormat {
  typedef tLhs Lhs;
  typedef tRhs Rhs;

  static_assert(Lhs::Cell::kDepth == Rhs::Cell::kDepth, "");
  static const int kDepth = Lhs::Cell::kDepth;
  static const int kRows = Lhs::Cell::kWidth * Lhs::kCells;
  static const int kCols = Rhs::Cell::kWidth * Rhs::kCells;
};

inline const char* CellOrderName(CellOrder o) {
  switch (o) {
    case CellOrder::DepthMajor:
      return "DepthMajor";
    case CellOrder::WidthMajor:
      return "WidthMajor";
    case CellOrder::Diagonal:
      return "Diagonal";
    default:
      assert(false);
      return nullptr;
  }
}

// Returns the offset into a cell, at which a given coefficient is stored.
template <typename CellFormat>
inline int OffsetIntoCell(int w, int d) {
  const int size = CellFormat::kWidth;
  switch (CellFormat::kOrder) {
    case CellOrder::DepthMajor:
      return w + d * CellFormat::kWidth;
    case CellOrder::WidthMajor:
      return d + w * CellFormat::kDepth;
    case CellOrder::Diagonal:
      assert(CellFormat::kWidth == CellFormat::kDepth);
      return ((size + w - d) * size + d) % (size * size);
    default:
      assert(false);
      return 0;
  }
}


struct KernelBase {
  virtual const char* Name() const = 0;

  virtual void Run(std::int32_t* dst_ptr, std::size_t dst_row_stride,
                   std::size_t dst_col_stride, const std::uint8_t* lhs_ptr,
                   const std::uint8_t* rhs_ptr, std::size_t start_depth,
                   std::size_t run_depth) const = 0;

  virtual ~KernelBase() {}
};

template <typename KernelScalarType>
struct ZeroPointInputValue {};

template <>
struct ZeroPointInputValue<std::uint8_t> {
  static constexpr std::uint8_t kValue = 0;
};

template <>
struct ZeroPointInputValue<std::int8_t> {
  static constexpr std::uint8_t kValue = 128;
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_KERNEL_H_
