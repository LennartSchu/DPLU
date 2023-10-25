#pragma once

#include <Eigen/Core>
#include <complex>

// some definitions
using Real = double;

using SparseMat = Eigen::SparseMatrix<Real>;
using SparseMatC = Eigen::SparseMatrix<std::complex<Real>>;
using Vec = Eigen::Matrix<Real, 3, 1>;
using VecC = Eigen::Matrix<std::complex<Real>, 3, 1>;
using namespace std::complex_literals;

template <typename T>
std::enable_if_t<
    std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>, bool>
assertEqual(const Eigen::Matrix<T, 3, 1>& lhs,
            const Eigen::Matrix<T, 3, 1>& rhs) {
  for (int i = 0; i < 3; i++) {
    if (lhs[i] != rhs[i]) return false;
  }

  return true;
}

// TODO: write test fixture