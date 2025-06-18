#pragma once

#include <Eigen/Dense>
#include <optional>

namespace cyclops::initializer {
  struct ImuTranslationMatchAnalysis;

  Eigen::MatrixXd evaluateImuTranslationMatchHessian(
    ImuTranslationMatchAnalysis const& analysis, double scale,
    Eigen::VectorXd const& inertial_state, Eigen::VectorXd const& visual_state);
}  // namespace cyclops::initializer
