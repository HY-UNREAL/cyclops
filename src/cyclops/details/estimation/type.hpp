#pragma once

#include <Eigen/Dense>

#include <functional>
#include <map>
#include <set>

namespace ceres {
  struct Problem;

  namespace internal {
    struct ResidualBlock;
  }
}  // namespace ceres

namespace cyclops::estimation {
  struct Node;
  using NodeSet = std::set<Node>;
  using NodeSetCRef = std::reference_wrapper<NodeSet const>;

  struct Factor;

  using FactorID = uint64_t;
  using FactorPtr = ceres::internal::ResidualBlock*;
  using FactorEntry = std::tuple<FactorPtr, Factor>;

  using FactorSet = std::map<FactorID, FactorEntry>;

  using Parameter = double*;

  struct GaussianPrior {
    Eigen::MatrixXd jacobian;
    Eigen::VectorXd residual;

    NodeSet input_nodes;
    std::vector<double> nominal_parameters;
  };
}  // namespace cyclops::estimation
