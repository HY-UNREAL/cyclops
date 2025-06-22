#include "cyclops/details/initializer/vision/bundle_adjustment.hpp"
#include "cyclops/details/initializer/vision/bundle_adjustment_factors.hpp"
#include "cyclops/details/initializer/vision/bundle_adjustment_states.hpp"
#include "cyclops/details/initializer/vision/type.hpp"

#include "cyclops/details/estimation/ceres/manifold.se3.hpp"

#include "cyclops/details/utils/type.hpp"
#include "cyclops/details/utils/math.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <ceres/ceres.h>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

#include <range/v3/all.hpp>

#include <map>
#include <memory>
#include <vector>
#include <optional>

namespace cyclops::initializer {
  using ceres::AutoDiffCostFunction;
  using ceres::AutoDiffLocalParameterization;

  using Eigen::MatrixXd;
  using Eigen::Quaterniond;
  using Eigen::Vector2d;
  using Eigen::Vector3d;

  namespace views = ranges::views;

  using MultiviewImageFrame = std::map<LandmarkID, FeaturePoint>;
  using MultiViewImageData = std::map<FrameID, MultiviewImageFrame>;

  class BundleAdjustmentConstructor {
  private:
    CyclopsConfig const& _config;
    std::shared_ptr<BundleAdjustmentOptimizationState> _state;

    ceres::Problem _problem;

    std::map<LandmarkID, double*> _landmark_parameters;
    std::map<FrameID, double*> _frame_parameters;
    std::vector<ceres::ResidualBlockId> _residuals;

    bool constructCameraMotionStates();

    int constructLandmarkFactors(MultiViewImageData const& data);
    int constructImuRotationPriorFactors(
      std::map<FrameID, TwoViewImuRotationConstraint> const& imu_prior);
    bool constructVirtualScaleGaugeFactor();

    struct SolutionEvaluation {
      double cost;
      std::vector<double> residuals;
      EigenCRSMatrix jacobian;
    };
    SolutionEvaluation evaluateFinalSolution();
    MatrixXd getCameraMotionFisherInformation(EigenCRSMatrix const& jacobian);

    int countFeatureOutliers(
      int n_features, std::vector<double> const& residuals) const;
    int countGyroMotionMismatch(
      int n_features, int n_frames, std::vector<double> const& residuals) const;
    bool detectGyroBiasPriorMismatch(
      int n_features, int n_frames, std::vector<double> const& residuals) const;

    struct SolutionAcceptance {
      bool accept;

      double p_value;
      double inlier_ratio;
      int n_rotation_mismatch;
    };
    SolutionAcceptance determineAcceptance(
      ceres::Solver::Summary const& summary,
      SolutionEvaluation const& evaluation, int n_features, int n_frames) const;

  public:
    BundleAdjustmentConstructor(
      CyclopsConfig const& config,
      std::shared_ptr<BundleAdjustmentOptimizationState> state);

    std::optional<MSfMSolution> solve(
      MultiViewImageData const& data,
      std::map<FrameID, TwoViewImuRotationConstraint> const& imu_prior);
  };

  bool BundleAdjustmentConstructor::constructCameraMotionStates() {
    _problem.AddParameterBlock(_state->gyro_bias.data(), 3);

    for (auto& [frame_id, x] : _state->camera_motions) {
      auto parameterization = new AutoDiffLocalParameterization<
        estimation::ExponentialSE3Plus<false, false>, 7, 6>;
      _problem.AddParameterBlock(x.data(), 7, parameterization);
      _frame_parameters.emplace(frame_id, x.data());
    }
    return true;
  }

  int BundleAdjustmentConstructor::constructLandmarkFactors(
    MultiViewImageData const& data) {
    auto const& vision_initializer_config = _config.initialization.vision;
    auto const& robust_kernel_radius =
      vision_initializer_config.bundle_adjustment_robust_kernel_radius;

    int n_landmark_measurements = 0;

    for (auto const& [frame_id, image_frame] : data) {
      auto maybe_x = _frame_parameters.find(frame_id);
      if (maybe_x == _frame_parameters.end()) {
        __logger__->error(
          "Uninitialized motion frame in vision-only bundle adjustment",
          frame_id);
        return -1;
      }
      auto& [_, x] = *maybe_x;

      for (auto const& [landmark_id, feature] : image_frame) {
        auto maybe_f = _state->landmark_positions.find(landmark_id);
        if (maybe_f == _state->landmark_positions.end())
          continue;
        auto& [_, f] = *maybe_f;

        auto cost = new LandmarkProjectionCost(feature);
        auto loss = new ceres::HuberLoss(robust_kernel_radius);

        _residuals.emplace_back(
          _problem.AddResidualBlock(cost, loss, x, f.data()));
        _landmark_parameters.emplace(landmark_id, f.data());

        n_landmark_measurements++;
      }
    }
    return n_landmark_measurements;
  }

  int BundleAdjustmentConstructor::constructImuRotationPriorFactors(
    std::map<FrameID, TwoViewImuRotationConstraint> const& imu_prior) {
    int n_frames = 0;

    for (auto const& prior : imu_prior | views::values) {
      auto maybe_x_init = _frame_parameters.find(prior.init_frame_id);
      auto maybe_x_term = _frame_parameters.find(prior.term_frame_id);

      if (
        maybe_x_init == _frame_parameters.end() ||
        maybe_x_term == _frame_parameters.end()) {
        continue;
      }
      auto& [_1, x_init] = *maybe_x_init;
      auto& [_2, x_term] = *maybe_x_term;

      auto factor = new AutoDiffCostFunction<
        BundleAdjustmentCameraRotationPriorCost, 3, 7, 7, 3>(
        new BundleAdjustmentCameraRotationPriorCost(prior.rotation));
      _residuals.emplace_back(_problem.AddResidualBlock(
        factor, nullptr, x_init, x_term, _state->gyro_bias.data()));

      n_frames++;
    }

    auto bias_prior = 1.0 / _config.noise.gyr_bias_prior_stddev;
    auto cost =
      new AutoDiffCostFunction<BundleAdjustmentGyroBiasZeroPriorCost, 3, 3>(
        new BundleAdjustmentGyroBiasZeroPriorCost(bias_prior));
    _residuals.emplace_back(
      _problem.AddResidualBlock(cost, nullptr, _state->gyro_bias.data()));

    return n_frames;
  }

  bool BundleAdjustmentConstructor::constructVirtualScaleGaugeFactor() {
    auto const& msfm_config = _config.initialization.vision.multiview;
    auto const& deviation = msfm_config.scale_gauge_soft_constraint_deviation;

    auto maybe_normalized_frame_pair = _state->normalize();
    if (!maybe_normalized_frame_pair)
      return false;
    auto& [x0, xn] = *maybe_normalized_frame_pair;

    auto x0_ptr = x0.get().data();
    auto xn_ptr = xn.get().data();

    auto factor = new AutoDiffCostFunction<
      BundleAdjustmentScaleConstraintVirtualCost, 1, 7, 7>(
      new BundleAdjustmentScaleConstraintVirtualCost(1 / deviation));
    _residuals.emplace_back(
      _problem.AddResidualBlock(factor, nullptr, x0_ptr, xn_ptr));
    _problem.SetParameterBlockConstant(x0_ptr);

    return true;
  }

  BundleAdjustmentConstructor::SolutionEvaluation
  BundleAdjustmentConstructor::evaluateFinalSolution() {
    ceres::Problem::EvaluateOptions opt;

    auto bs = views::single(_state->gyro_bias.data());
    auto fs = _landmark_parameters | views::values;
    auto xs = _frame_parameters | views::values;
    opt.parameter_blocks = views::concat(bs, fs, xs) | ranges::to_vector;
    opt.residual_blocks = _residuals;

    double cost;
    std::vector<double> residuals;
    ceres::CRSMatrix jacobian;
    _problem.Evaluate(opt, &cost, &residuals, nullptr, &jacobian);

    return SolutionEvaluation {
      .cost = cost,
      .residuals = std::move(residuals),
      .jacobian = Eigen::Map<EigenCRSMatrix>(
        jacobian.num_rows, jacobian.num_cols, jacobian.values.size(),
        jacobian.rows.data(), jacobian.cols.data(), jacobian.values.data()),
    };
  }

  MatrixXd BundleAdjustmentConstructor::getCameraMotionFisherInformation(
    EigenCRSMatrix const& jacobian) {
    // Marginalize gyro bias + landmark position states
    auto m = 3 + _landmark_parameters.size() * 3;
    auto k = _frame_parameters.size() * 6;

    EigenCRSMatrix J_m = jacobian.middleCols(0, m);
    EigenCRSMatrix J_k = jacobian.middleCols(m, k);
    MatrixXd const H_kk = J_k.transpose() * J_k;

    EigenCCSMatrix const H_mm = J_m.transpose() * J_m;
    MatrixXd const H_km = J_k.transpose() * J_m;
    MatrixXd const H_mk = H_km.transpose();

    Eigen::SimplicialLDLT<EigenCCSMatrix> H_mm__inv(H_mm);
    MatrixXd H_km__H_mm__inv__H_mk = H_km * H_mm__inv.solve(H_mk);
    return H_kk - H_km__H_mm__inv__H_mk;
  }

  int BundleAdjustmentConstructor::countFeatureOutliers(
    int n_features, std::vector<double> const& residuals) const {
    auto const& vision_config = _config.initialization.vision;

    auto rho = vision_config.bundle_adjustment_robust_kernel_radius;
    auto rho_square = rho * rho;

    int n_outliers = 0;
    for (auto i = 0; i < n_features; i++) {
      auto r1 = residuals.at(2 * i);
      auto r2 = residuals.at(2 * i + 1);

      auto s = r1 * r1 + r2 * r2;
      if (s >= rho_square)
        n_outliers++;
    }
    return n_outliers;
  }

  int BundleAdjustmentConstructor::countGyroMotionMismatch(
    int n_features, int n_frames, std::vector<double> const& residuals) const {
    auto const& vision_config = _config.initialization.vision;
    auto min_p_value = vision_config.acceptance_test.gyro_motion_min_p_value;
    auto n_mismatch = 0;

    for (int i = 0; i < n_frames; i++) {
      auto r1 = residuals.at(2 * n_features + 3 * i);
      auto r2 = residuals.at(2 * n_features + 3 * i + 1);
      auto r3 = residuals.at(2 * n_features + 3 * i + 2);

      auto p_value = 1.0 - chiSquaredCdf(3, r1 * r1 + r2 * r2 + r3 * r3);
      __logger__->debug("MSfM IMU rotation prior p-value: {}%", p_value * 100);

      if (p_value < min_p_value) {
        __logger__->info(
          "MSfM rotation mismatch to IMU prior at {}-th motion.", i);
        __logger__->info("{}% < {}%", p_value * 100, min_p_value * 100);
        n_mismatch++;
      }
    }
    return n_mismatch;
  }

  bool BundleAdjustmentConstructor::detectGyroBiasPriorMismatch(
    int n_features, int n_frames, std::vector<double> const& residuals) const {
    auto const& vision_config = _config.initialization.vision;
    auto min_p_value = vision_config.acceptance_test.gyro_bias_min_p_value;
    auto r1 = residuals.at(2 * n_features + 3 * n_frames);
    auto r2 = residuals.at(2 * n_features + 3 * n_frames + 1);
    auto r3 = residuals.at(2 * n_features + 3 * n_frames + 2);

    auto p_value = 1.0 - chiSquaredCdf(3, r1 * r1 + r2 * r2 + r3 * r3);
    __logger__->debug("MSfM gyro bias prior p-value: {}%", p_value * 100);

    if (p_value < min_p_value) {
      __logger__->info("MSfM gyro bias zero prior mismatch");
      __logger__->info("{}% < {}%", p_value * 100, min_p_value * 100);
      return true;
    }
    return false;
  }

  BundleAdjustmentConstructor::SolutionAcceptance
  BundleAdjustmentConstructor::determineAcceptance(
    ceres::Solver::Summary const& summary, SolutionEvaluation const& evaluation,
    int n_features, int n_frames) const {
    auto n_residuals = static_cast<int>(summary.num_residuals);
    auto n_parameters = static_cast<int>(summary.num_effective_parameters);

    auto degrees_of_freedom = n_residuals - n_parameters;
    auto p_value = 1.0 - chiSquaredCdf(degrees_of_freedom, summary.final_cost);

    auto n_outliers = countFeatureOutliers(n_features, evaluation.residuals);
    auto outlier_ratio = static_cast<double>(n_outliers) / n_features;
    auto inlier_ratio = 1.0 - outlier_ratio;

    auto n_mismatch =
      countGyroMotionMismatch(n_features, n_frames, evaluation.residuals);
    auto bias_mismatch =
      detectGyroBiasPriorMismatch(n_features, n_frames, evaluation.residuals);

    __logger__->debug("MSfM solution statistics:");
    __logger__->debug("  p-value: {}%", p_value * 100);
    __logger__->debug("  inlier ratio: {}%", inlier_ratio * 100);
    __logger__->debug("  IMU rotation prior mismatch: {}", n_mismatch);

    if (n_mismatch > 0) {
      __logger__->info("MSfM solution rejected. Reason: gyro prior mismatch.");
      return {false, p_value, inlier_ratio, n_mismatch};
    }

    if (bias_mismatch) {
      __logger__->info(
        "MSfM solution rejected. Reason: gyro bias prior mismatch.");
      return {false, p_value, inlier_ratio, n_mismatch};
    }

    auto const& accept_config = _config.initialization.vision.acceptance_test;
    auto accept_min_p_value = accept_config.min_significant_probability;
    auto accept_min_inlier_ratio = accept_config.min_inlier_ratio;

    if (p_value < accept_min_p_value) {
      __logger__->info(
        "MSfM solution rejected. Reason: insignificant final cost.");
      __logger__->debug("{}% < {}%", p_value * 100, accept_min_p_value * 100);
      return {false, p_value, inlier_ratio, n_mismatch};
    }

    if (inlier_ratio < accept_min_inlier_ratio) {
      __logger__->info("MSfM solution rejected. Reason: low inlier ratio.");
      __logger__->debug(
        "{}% < {}%", inlier_ratio * 100, accept_min_inlier_ratio * 100);
      return {false, p_value, inlier_ratio, n_mismatch};
    }

    return {true, p_value, inlier_ratio, n_mismatch};
  }

  std::optional<MSfMSolution> BundleAdjustmentConstructor::solve(
    MultiViewImageData const& data,
    std::map<FrameID, TwoViewImuRotationConstraint> const& imu_prior) {
    if (!constructCameraMotionStates()) {
      __logger__->error("BA camera motion state construction failed.");
      return std::nullopt;
    }

    auto n_features = constructLandmarkFactors(data);
    if (n_features < 0) {
      __logger__->error("BA landmark factor construction failed.");
      return std::nullopt;
    }
    auto n_frames = constructImuRotationPriorFactors(imu_prior);

    if (!constructVirtualScaleGaugeFactor()) {
      __logger__->error("BA virtual scale gauge factor construction failed.");
      return std::nullopt;
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;

    auto const& msfm_config = _config.initialization.vision.multiview;
    options.max_solver_time_in_seconds =
      msfm_config.bundle_adjustment_max_solver_time;
    options.max_num_iterations = msfm_config.bundle_adjustment_max_iterations;

    ceres::Solve(options, &_problem, &summary);
    __logger__->info("Finished bundle adjustment: {}", summary.BriefReport());
    __logger__->info("Gyro bias: {}", _state->gyro_bias.value().transpose());

    auto evaluation = evaluateFinalSolution();
    auto [accept, p_value, inlier_ratio, n_mismatch] =
      determineAcceptance(summary, evaluation, n_features, n_frames);

    return MSfMSolution {
      .acceptable = accept,
      .solution_significant_probability = p_value,
      .measurement_inlier_ratio = inlier_ratio,

      .geometry = _state->asMultiViewGeometry(),
      .motion_information_weight =
        getCameraMotionFisherInformation(evaluation.jacobian),
      .gyro_bias = _state->gyro_bias.value(),
    };
  }

  BundleAdjustmentConstructor::BundleAdjustmentConstructor(
    CyclopsConfig const& config,
    std::shared_ptr<BundleAdjustmentOptimizationState> state)
      : _config(config), _state(state) {
  }

  std::optional<MSfMSolution> solveBundleAdjustment(
    CyclopsConfig const& config, MultiViewGeometry const& guess,
    MultiViewImageData const& data,
    std::map<FrameID, TwoViewImuRotationConstraint> const& imu_prior) {
    auto state = std::make_shared<BundleAdjustmentOptimizationState>(guess);
    auto context = BundleAdjustmentConstructor(config, state);

    return context.solve(data, imu_prior);
  }
}  // namespace cyclops::initializer
