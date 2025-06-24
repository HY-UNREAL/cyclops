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
     * Concatenation of the gravity, acc bias error, and IMU body velocities,
     *
     *                                  [  g  ]
     *                                  [ b_a ]
     *                            x_I = [ v_1 ]
     *                                  [ ... ]
     *                                  [ v_N ].
     */
    Eigen::VectorXd inertial_solution;

    // Perturbations of the MSfM position estimation, omitting the first frame.
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
