#pragma once

#include <Eigen/Dense>
#include <optional>

namespace cyclops::initializer {
  struct ImuMatchAnalysis;

  Eigen::MatrixXd evaluateImuMatchHessian(
    ImuMatchAnalysis const& analysis, double scale,
    Eigen::VectorXd const& inertial_state, Eigen::VectorXd const& visual_state);
}  // namespace cyclops::initializer
