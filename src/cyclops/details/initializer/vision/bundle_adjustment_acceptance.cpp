#include "cyclops/details/initializer/vision/bundle_adjustment_acceptance.hpp"
#include "cyclops/details/initializer/vision/bundle_adjustment_context.hpp"
#include "cyclops/details/initializer/vision/bundle_adjustment_states.hpp"

#include "cyclops/details/utils/type.hpp"
#include "cyclops/details/utils/math.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>
#include <range/v3/all.hpp>

namespace cyclops::initializer {
  using Context = BundleAdjustmentOptimizationContext;

  class BundleAdjustmentAcceptDiscriminatorImpl:
      public BundleAdjustmentAcceptDiscriminator {
  private:
    std::shared_ptr<CyclopsConfig const> _config;

    std::tuple<double, std::vector<double>, EigenCRSMatrix> evaluateSolution(
      Context& context);
    Eigen::MatrixXd getMotionInformation(
      Context& context, EigenCRSMatrix const& jacobian);

    double evaluateOutlierRatio(
      Context& context, std::vector<double> const& residuals) const;
    int countGyroMotionMismatch(
      Context& context, std::vector<double> const& residuals) const;
    bool detectGyroBiasPriorMismatch(
      Context& context, std::vector<double> const& residuals) const;

    bool determineAcceptance(
      BundleAdjustmentSolutionUncertainty const& uncertainty) const;

  public:
    BundleAdjustmentSolutionAcceptance evaluate(
      ceres::Solver::Summary const& summary, Context& context) override;

    explicit BundleAdjustmentAcceptDiscriminatorImpl(
      std::shared_ptr<CyclopsConfig const> config);
  };

  static EigenCRSMatrix asEigenMatrix(ceres::CRSMatrix const& matrix) {
    return Eigen::Map<EigenCRSMatrix const>(
      matrix.num_rows, matrix.num_cols, matrix.values.size(),
      matrix.rows.data(), matrix.cols.data(), matrix.values.data());
  }

  std::tuple<double, std::vector<double>, EigenCRSMatrix>
  BundleAdjustmentAcceptDiscriminatorImpl::evaluateSolution(Context& context) {
    ceres::Problem::EvaluateOptions opt;

    namespace views = ranges::views;
    auto bs = views::single(context.state().gyro_bias.data());
    auto fs = context.landmarks() | views::values;
    auto xs = context.frames() | views::values;
    opt.parameter_blocks = views::concat(bs, fs, xs) | ranges::to_vector;
    opt.residual_blocks = context.residuals();

    double cost;
    std::vector<double> residuals;
    ceres::CRSMatrix jacobian;
    context.problem().Evaluate(opt, &cost, &residuals, nullptr, &jacobian);

    return std::make_tuple(cost, std::move(residuals), asEigenMatrix(jacobian));
  }

  Eigen::MatrixXd BundleAdjustmentAcceptDiscriminatorImpl::getMotionInformation(
    Context& context, EigenCRSMatrix const& jacobian) {
    // Marginalize gyro bias + landmark position states
    auto m = 3 + context.landmarks().size() * 3;
    auto k = context.frames().size() * 6;

    EigenCRSMatrix J_m = jacobian.middleCols(0, m);
    EigenCRSMatrix J_k = jacobian.middleCols(m, k);
    Eigen::MatrixXd const H_kk = J_k.transpose() * J_k;

    EigenCCSMatrix const H_mm = J_m.transpose() * J_m;
    Eigen::MatrixXd const H_km = J_k.transpose() * J_m;
    Eigen::MatrixXd const H_mk = H_km.transpose();

    Eigen::SimplicialLDLT<EigenCCSMatrix> H_mm__inv(H_mm);
    Eigen::MatrixXd H_km__H_mm__inv__H_mk = H_km * H_mm__inv.solve(H_mk);
    return H_kk - H_km__H_mm__inv__H_mk;
  }

  double BundleAdjustmentAcceptDiscriminatorImpl::evaluateOutlierRatio(
    Context& context, std::vector<double> const& residuals) const {
    int n_features = context.nLandmarkMeasurements();

    auto const& vision_config = _config->initialization.vision;
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

    return static_cast<double>(n_outliers) / n_features;
  }

  int BundleAdjustmentAcceptDiscriminatorImpl::countGyroMotionMismatch(
    Context& context, std::vector<double> const& residuals) const {
    auto const& vision_config = _config->initialization.vision;
    auto min_p_value = vision_config.acceptance_test.gyro_motion_min_p_value;
    auto n_mismatch = 0;

    int n_motions = context.nGyroMotionConstraints();
    int n_features = context.nLandmarkMeasurements();

    for (int i = 0; i < n_motions; i++) {
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

  bool BundleAdjustmentAcceptDiscriminatorImpl::detectGyroBiasPriorMismatch(
    Context& context, std::vector<double> const& residuals) const {
    int n_motions = context.nGyroMotionConstraints();
    int n_features = context.nLandmarkMeasurements();

    auto const& vision_config = _config->initialization.vision;
    auto min_p_value = vision_config.acceptance_test.gyro_bias_min_p_value;

    auto r1 = residuals.at(2 * n_features + 3 * n_motions);
    auto r2 = residuals.at(2 * n_features + 3 * n_motions + 1);
    auto r3 = residuals.at(2 * n_features + 3 * n_motions + 2);

    auto p_value = 1.0 - chiSquaredCdf(3, r1 * r1 + r2 * r2 + r3 * r3);
    __logger__->debug("MSfM gyro bias prior p-value: {}%", p_value * 100);

    if (p_value < min_p_value) {
      __logger__->info("MSfM gyro bias zero prior mismatch");
      __logger__->info("{}% < {}%", p_value * 100, min_p_value * 100);
      return true;
    }
    return false;
  }

  bool BundleAdjustmentAcceptDiscriminatorImpl::determineAcceptance(
    BundleAdjustmentSolutionUncertainty const& uncertainty) const {
    auto n_mismatch = uncertainty.n_gyro_motion_mismatch;
    auto bias_mismatch = uncertainty.gyro_bias_prior_mismatch;
    auto p_value = uncertainty.p_value;
    auto inlier_ratio = uncertainty.inlier_ratio;

    __logger__->debug("MSfM solution uncertainty:");
    __logger__->debug("  p-value: {}%", p_value * 100);
    __logger__->debug("  inlier ratio: {}%", uncertainty.inlier_ratio * 100);
    __logger__->debug("  Gyro motion mismatch: {}", n_mismatch);

    auto report = [](auto reason) {
      __logger__->info("MSfM solution rejected. Reason: {}.", reason);
    };
    auto reportPercentage = [](auto a, auto b) {
      __logger__->debug("{}% < {}%", a * 100, b * 100);
    };

    if (n_mismatch > 0) {
      report("gyro motion mismatch");
      return false;
    }

    if (bias_mismatch) {
      report("gyro bias mismatch");
      return false;
    }

    auto const& accept_config = _config->initialization.vision.acceptance_test;
    auto accept_min_p_value = accept_config.min_significant_probability;
    auto accept_min_inlier_ratio = accept_config.min_inlier_ratio;

    if (p_value < accept_min_p_value) {
      report("insignificant final cost");
      reportPercentage(p_value, accept_min_p_value);
      return false;
    }

    if (inlier_ratio < accept_min_inlier_ratio) {
      report("low inlier ratio");
      reportPercentage(inlier_ratio, accept_min_inlier_ratio);
      return false;
    }

    return true;
  }

  BundleAdjustmentSolutionAcceptance
  BundleAdjustmentAcceptDiscriminatorImpl::evaluate(
    ceres::Solver::Summary const& summary, Context& context) {
    auto [cost, residuals, jacobian] = evaluateSolution(context);

    auto n_residuals = static_cast<int>(summary.num_residuals);
    auto n_parameters = static_cast<int>(summary.num_effective_parameters);
    auto degrees_of_freedom = n_residuals - n_parameters;
    auto p_value = 1.0 - chiSquaredCdf(degrees_of_freedom, summary.final_cost);

    auto outlier_ratio = evaluateOutlierRatio(context, residuals);
    auto inlier_ratio = 1.0 - outlier_ratio;

    auto n_mismatch = countGyroMotionMismatch(context, residuals);
    auto bias_mismatch = detectGyroBiasPriorMismatch(context, residuals);

    auto motion_information = getMotionInformation(context, jacobian);

    auto uncertainty = BundleAdjustmentSolutionUncertainty {
      .p_value = p_value,
      .inlier_ratio = inlier_ratio,
      .n_gyro_motion_mismatch = n_mismatch,
      .gyro_bias_prior_mismatch = bias_mismatch,
      .motion_information = motion_information,
    };
    return {determineAcceptance(uncertainty), uncertainty};
  }

  BundleAdjustmentAcceptDiscriminatorImpl::
    BundleAdjustmentAcceptDiscriminatorImpl(
      std::shared_ptr<CyclopsConfig const> config)
      : _config(config) {
  }

  std::unique_ptr<BundleAdjustmentAcceptDiscriminator>
  BundleAdjustmentAcceptDiscriminator::Create(
    std::shared_ptr<CyclopsConfig const> config) {
    return std::make_unique<BundleAdjustmentAcceptDiscriminatorImpl>(config);
  }
}  // namespace cyclops::initializer
