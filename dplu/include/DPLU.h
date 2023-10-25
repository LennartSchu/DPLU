#pragma once

#include <Eigen/SparseLU>

template <typename MatrixType, typename OrderingType>
class DPLU : public Eigen::SparseLU<MatrixType, OrderingType> {};