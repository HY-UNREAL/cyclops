#pragma once

#include <Eigen/Dense>
#include <optional>

namespace cyclops {
  struct CyclopsConfig;
}  // namespace cyclops

namespace cyclops::initializer {
  struct ImuTranslationMatchAnalysis;
  struct ImuMatchScaleSampleSolution;

  struct ImuTranslationMatchUncertainty {
    double final_cost_significant_probability;
    double scale_log_deviation;
    Eigen::Vector2d gravity_tangent_deviation;
    Eigen::Vector3d bias_deviation;
    Eigen::VectorXd body_velocity_deviation;
    Eigen::VectorXd translation_scale_symmetric_deviation;
  };

  std::optional<ImuTranslationMatchUncertainty>
  analyzeImuTranslationMatchUncertainty(
    ImuTranslationMatchAnalysis const& analysis,
    ImuMatchScaleSampleSolution const& solution);
}  // namespace cyclops::initializer
