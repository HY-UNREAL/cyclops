#pragma once

#include <Eigen/Dense>

namespace cyclops {
  struct Qcqp1Solution {
    bool success;
    double multiplier;
    Eigen::Vector3d x;
  };

  Qcqp1Solution solveNormConstrainedQcqp1(
    Eigen::Matrix3d const& H, Eigen::Vector3d const& b, double norm_sqr,
    double multiplier_min, size_t max_iterations = 100,
    double constraint_violation_tolerance = 1e-6,
    double multiplier_safeguard_margin = 1e-6);
}  // namespace cyclops
