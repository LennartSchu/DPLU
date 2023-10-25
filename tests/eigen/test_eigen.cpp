#include <spdlog/spdlog.h>

#include <Eigen/SparseLU>
#include <complex>

#include "../test_utils.h"

namespace {
constexpr Real one = 1.;
constexpr int size = 3;
}  // namespace

bool testSparseLUReal() {
  Vec b;
  SparseMat A(size, size);
  Eigen::SparseLU<SparseMat, Eigen::COLAMDOrdering<int>> solver;

  for (int i = 0; i < size; i++) A.insert(i, i) = -one;

  solver.analyzePattern(A);
  solver.factorize(A);

  for (int i = 0; i < size; i++) b.coeffRef(i) = static_cast<Real>(i * one);

  const Vec x = solver.solve(b);

  const Vec solution = -b;

  return assertEqual(x, solution);
}

bool testSparseLUComplex() {
  VecC b;
  SparseMatC A(size, size);

  Eigen::SparseLU<SparseMatC, Eigen::COLAMDOrdering<int>> solver;

  for (int i = 0; i < size; i++) A.insert(i, i) = -one + 1.0i;

  solver.analyzePattern(A);
  solver.factorize(A);

  for (int i = 0; i < size; i++) b.coeffRef(i) = i * one - 1.0i;

  const VecC x = solver.solve(b);

  VecC solution;
  solution.coeffRef(0) = -0.5 + 0.5i;
  solution.coeffRef(1) = -1.0 + 0.0i;
  solution.coeffRef(2) = -1.5 - 0.5i;

  return assertEqual(x, solution);
}

int main() {
  std::vector<bool> results;
  bool result = testSparseLUReal();

  if (result) spdlog::info("SparseLU Real passed");

  results.push_back(result);

  result = testSparseLUComplex();

  if (result) spdlog::info("SparseLU Complex passed");

  results.push_back(result);

  result = std::all_of(results.begin(), results.end(),
                       [](const bool res) { return res; });
  return result ? 0 : 1;
}