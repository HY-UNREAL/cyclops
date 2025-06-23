#pragma once

#include <Eigen/Dense>
#include <optional>

namespace cyclops {
  struct CyclopsConfig;
}  // namespace cyclops

namespace cyclops::initializer {
  struct ImuMatchAnalysis;
  struct ImuMatchScaleSampleSolution;

  struct ImuMatchUncertainty {
    double final_cost_significant_probability;
    double scale_log_deviation;
    Eigen::Vector2d gravity_tangent_deviation;
    Eigen::Vector3d bias_deviation;
    Eigen::VectorXd body_velocity_deviation;
    Eigen::VectorXd translation_scale_symmetric_deviation;
  };

  std::optional<double> analyzeImuMatchCostProbability(
    int residual_dimension, int parameter_dimension, double cost);
  std::optional<ImuMatchUncertainty> analyzeImuMatchUncertainty(
    int frames_count, Eigen::MatrixXd const& hessian, double cost_p_value);

  std::optional<ImuMatchUncertainty> analyzeImuMatchUncertainty(
    ImuMatchAnalysis const& analysis,
    ImuMatchScaleSampleSolution const& solution);
}  // namespace cyclops::initializer
