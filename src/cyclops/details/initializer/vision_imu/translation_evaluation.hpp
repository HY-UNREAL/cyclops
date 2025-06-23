#pragma once

#include <Eigen/Dense>
#include <optional>

namespace cyclops {
  struct CyclopsConfig;
}

namespace cyclops::initializer {
  struct ImuMatchAnalysis;
  struct ImuMatchAnalysisCache;

  struct ImuMatchScaleEvaluation {
    double multiplier;
    double cost;

    /*
     * concatenation of gravity, acc bias error, and IMU body velocities.
     */
    Eigen::VectorXd inertial_solution;

    /*
     * perturbations of vision position estimation, omitting the first frame.
     * defined exponentially ignoring scale gauge; `p_i = p_hat_i + R_i * dp_i`.
     */
    Eigen::VectorXd visual_solution;
  };

  class ImuMatchScaleEvaluationContext {
  private:
    double const gravity_norm;
    ImuMatchAnalysis const& analysis;
    ImuMatchAnalysisCache const& cache;

  public:
    ImuMatchScaleEvaluationContext(
      double gravity_norm, ImuMatchAnalysis const& analysis,
      ImuMatchAnalysisCache const& cache);

    std::optional<ImuMatchScaleEvaluation> evaluate(double scale) const;
    double evaluateDerivative(
      ImuMatchScaleEvaluation const& evaluation, double scale) const;
  };
}  // namespace cyclops::initializer
