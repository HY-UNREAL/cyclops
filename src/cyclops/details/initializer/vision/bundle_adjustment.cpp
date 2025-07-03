#include "cyclops/details/initializer/vision/bundle_adjustment.hpp"
#include "cyclops/details/initializer/vision/bundle_adjustment_context.hpp"
#include "cyclops/details/initializer/vision/bundle_adjustment_acceptance.hpp"
#include "cyclops/details/initializer/vision/bundle_adjustment_states.hpp"
#include "cyclops/details/initializer/vision/type.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <ceres/ceres.h>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

#include <map>
#include <optional>

namespace cyclops::initializer {
  class BundleAdjustmentSolverImpl: public BundleAdjustmentSolver {
  private:
    std::unique_ptr<BundleAdjustmentAcceptDiscriminator> _accept_discriminator;
    std::shared_ptr<CyclopsConfig const> _config;

  public:
    explicit BundleAdjustmentSolverImpl(
      std::unique_ptr<BundleAdjustmentAcceptDiscriminator> accept_discriminator,
      std::shared_ptr<CyclopsConfig const> config);

    std::optional<BundleAdjustmentSolution> solve(
      MultiViewGeometry const& guess,
      std::map<FrameID, std::map<LandmarkID, FeaturePoint>> const& features,
      std::map<FrameID, GyroMotionConstraint> const& gyro_motion) override;
  };

  BundleAdjustmentSolverImpl::BundleAdjustmentSolverImpl(
    std::unique_ptr<BundleAdjustmentAcceptDiscriminator> accept_discriminator,
    std::shared_ptr<CyclopsConfig const> config)
      : _accept_discriminator(std::move(accept_discriminator)),
        _config(config) {
  }

  std::optional<BundleAdjustmentSolution> BundleAdjustmentSolverImpl::solve(
    MultiViewGeometry const& guess,
    std::map<FrameID, std::map<LandmarkID, FeaturePoint>> const& features,
    std::map<FrameID, GyroMotionConstraint> const& gyro_motion) {
    auto context = BundleAdjustmentOptimizationContext(*_config, guess);
    if (!context.construct(features, gyro_motion))
      return std::nullopt;

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;

    auto const& msfm_config = _config->initialization.vision.multiview;
    options.max_solver_time_in_seconds =
      msfm_config.bundle_adjustment_max_solver_time;
    options.max_num_iterations = msfm_config.bundle_adjustment_max_iterations;

    ceres::Solve(options, &context.problem(), &summary);
    __logger__->info("Finished bundle adjustment: {}", summary.BriefReport());
    __logger__->info(
      "Gyro bias: {}", context.state().gyro_bias.value().transpose());

    auto acceptance = _accept_discriminator->evaluate(summary, context);
    auto const& uncertainty = acceptance.uncertainty;

    return BundleAdjustmentSolution {
      .acceptable = acceptance.accept,
      .solution_significant_probability = uncertainty.p_value,
      .measurement_inlier_ratio = uncertainty.inlier_ratio,

      .n_inliers = uncertainty.n_inliers,
      .n_outliers = uncertainty.n_outliers,

      .camera_motions = context.state().cameraMotions(),
      .motion_information_weight = uncertainty.motion_information,
      .gyro_bias = context.state().gyro_bias.value(),
      .landmarks = context.state().landmarkPositions(),
    };
  }

  std::unique_ptr<BundleAdjustmentSolver> BundleAdjustmentSolver::Create(
    std::shared_ptr<CyclopsConfig const> config) {
    return std::make_unique<BundleAdjustmentSolverImpl>(
      BundleAdjustmentAcceptDiscriminator::Create(config), config);
  }
}  // namespace cyclops::initializer
