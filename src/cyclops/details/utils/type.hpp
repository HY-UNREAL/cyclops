#pragma once

#include <Eigen/Sparse>

namespace cyclops {
  using EigenCRSMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  using EigenCCSMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;

  /*
   * Copy-pasted from https://en.cppreference.com/w/cpp/utility/variant/visit.
   * Helper type for the visitor #4.
   */
  template <class... Ts>
  struct overloaded: Ts... {
    using Ts::operator()...;
  };

  // Explicit deduction guide (not needed as of C++20).
  template <class... Ts>
  overloaded(Ts...) -> overloaded<Ts...>;
}  // namespace cyclops
