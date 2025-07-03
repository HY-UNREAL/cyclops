#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>

namespace cyclops {
  struct CyclopsConfig;
}

namespace cyclops::initializer {
  struct BundleAdjustmentOptimizationContext;

  struct BundleAdjustmentSolutionUncertainty {
    double p_value;
    double inlier_ratio;

    int n_inliers;
    int n_outliers;
    int n_gyro_motion_mismatch;

    bool gyro_bias_prior_mismatch;

    Eigen::MatrixXd motion_information;
  };

  struct BundleAdjustmentSolutionAcceptance {
    bool accept;
    BundleAdjustmentSolutionUncertainty uncertainty;
  };

  class BundleAdjustmentAcceptDiscriminator {
  public:
    virtual ~BundleAdjustmentAcceptDiscriminator() = default;

    virtual BundleAdjustmentSolutionAcceptance evaluate(
      ceres::Solver::Summary const& summary,
      BundleAdjustmentOptimizationContext& context) = 0;

    static std::unique_ptr<BundleAdjustmentAcceptDiscriminator> Create(
      std::shared_ptr<CyclopsConfig const> config);
  };
}  // namespace cyclops::initializer
