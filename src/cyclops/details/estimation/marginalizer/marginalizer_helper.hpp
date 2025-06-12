#pragma once

#include "cyclops/details/estimation/type.hpp"

#include <Eigen/Dense>

#include <vector>
#include <set>

namespace cyclops::estimation {
  struct GaussianPrior;

  struct FactorGraphInstance;
  struct StateVariableReadAccessor;

  struct MarginalizationSubgraph {
    NodeSet drop_nodes;
    NodeSet keep_nodes;
    FactorSet factors;
  };

  GaussianPrior evaluateGaussianPrior(
    FactorGraphInstance& graph, StateVariableReadAccessor const& state_accessor,
    MarginalizationSubgraph const& drop_subgraph);
}  // namespace cyclops::estimation
