#include "cyclops/details/initializer/vision/bundle_adjustment_context.hpp"
#include "cyclops/details/initializer/vision/bundle_adjustment_factors.hpp"
#include "cyclops/details/initializer/vision/bundle_adjustment_states.hpp"
#include "cyclops/details/initializer/vision/type.hpp"

#include "cyclops/details/estimation/ceres/manifold.se3.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <ceres/ceres.h>
#include <spdlog/spdlog.h>

#include <range/v3/all.hpp>

#include <vector>

namespace cyclops::initializer {
  using ceres::AutoDiffCostFunction;
  using ceres::AutoDiffLocalParameterization;

  namespace views = ranges::views;

  using MultiViewImageData =
    std::map<FrameID, std::map<LandmarkID, FeaturePoint>>;
  using MultiViewGyroMotionData = std::map<FrameID, GyroMotionConstraint>;

  struct BundleAdjustmentOptimizationContext::Impl {
    CyclopsConfig const& _config;
    BundleAdjustmentOptimizationState _state;

    ceres::Problem _problem;
    std::map<LandmarkID, double*> _landmarks;
    std::map<FrameID, double*> _frames;
    std::vector<ceres::ResidualBlockId> _residuals;

    int _n_landmark_measurements = 0;
    int _n_gyro_motion_constraints = 0;

    bool constructMotionStates();
    bool constructLandmarkFactors(MultiViewImageData const& features);
    bool constructGyroMotionFactors(
      MultiViewGyroMotionData const& gyro_motions);
    bool constructScaleGaugeFactor();

  public:
    Impl(CyclopsConfig const& config, MultiViewGeometry const& initial_guess);
    bool construct(
      MultiViewImageData const& features,
      MultiViewGyroMotionData const& gyro_motions);
  };

  bool BundleAdjustmentOptimizationContext::Impl::constructMotionStates() {
    _problem.AddParameterBlock(_state.gyro_bias.data(), 3);

    for (auto& [frame_id, x] : _state.camera_motions) {
      auto parameterization = new AutoDiffLocalParameterization<
        estimation::ExponentialSE3Plus<false, false>, 7, 6>;
      _problem.AddParameterBlock(x.data(), 7, parameterization);
      _frames.emplace(frame_id, x.data());
    }
    return true;
  }

  bool BundleAdjustmentOptimizationContext::Impl::constructLandmarkFactors(
    MultiViewImageData const& features) {
    auto const& vision_config = _config.initialization.vision;
    auto const& kernel_radius =
      vision_config.bundle_adjustment_robust_kernel_radius;

    for (auto const& [frame_id, image_frame] : features) {
      auto maybe_x = _frames.find(frame_id);
      if (maybe_x == _frames.end()) {
        __logger__->error(
          "Uninitialized motion frame in vision-only bundle adjustment",
          frame_id);
        return false;
      }
      auto& [_, x] = *maybe_x;

      for (auto const& [landmark_id, feature] : image_frame) {
        auto maybe_f = _state.landmark_positions.find(landmark_id);
        if (maybe_f == _state.landmark_positions.end())
          continue;
        auto& [_, f] = *maybe_f;

        auto cost = new LandmarkProjectionCost(feature);
        auto loss = new ceres::HuberLoss(kernel_radius);

        _residuals.emplace_back(
          _problem.AddResidualBlock(cost, loss, x, f.data()));
        _landmarks.emplace(landmark_id, f.data());

        _n_landmark_measurements++;
      }
    }
    return true;
  }

  bool BundleAdjustmentOptimizationContext::Impl::constructGyroMotionFactors(
    MultiViewGyroMotionData const& gyro_motions) {
    for (auto const& motion : gyro_motions | views::values) {
      auto maybe_x_init = _frames.find(motion.init_frame_id);
      auto maybe_x_term = _frames.find(motion.term_frame_id);

      if (maybe_x_init == _frames.end() || maybe_x_term == _frames.end()) {
        continue;
      }
      auto& [_1, x_init] = *maybe_x_init;
      auto& [_2, x_term] = *maybe_x_term;

      auto factor = new AutoDiffCostFunction<
        BundleAdjustmentCameraRotationPriorCost, 3, 7, 7, 3>(
        new BundleAdjustmentCameraRotationPriorCost(motion));
      _residuals.emplace_back(_problem.AddResidualBlock(
        factor, nullptr, x_init, x_term, _state.gyro_bias.data()));

      _n_gyro_motion_constraints++;
    }

    auto bias_prior = 1.0 / _config.noise.gyr_bias_prior_stddev;
    auto cost =
      new AutoDiffCostFunction<BundleAdjustmentGyroBiasZeroPriorCost, 3, 3>(
        new BundleAdjustmentGyroBiasZeroPriorCost(bias_prior));
    _residuals.emplace_back(
      _problem.AddResidualBlock(cost, nullptr, _state.gyro_bias.data()));

    return true;
  }

  bool BundleAdjustmentOptimizationContext::Impl::constructScaleGaugeFactor() {
    auto const& msfm_config = _config.initialization.vision.multiview;
    auto const& deviation = msfm_config.scale_gauge_soft_constraint_deviation;

    auto maybe_normalized_frame_pair = _state.normalize();
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

  bool BundleAdjustmentOptimizationContext::Impl::construct(
    MultiViewImageData const& features,
    MultiViewGyroMotionData const& gyro_motions) {
    if (!constructMotionStates()) {
      __logger__->error("BA camera motion state construction failed.");
      return false;
    }

    if (!constructLandmarkFactors(features)) {
      __logger__->error("BA landmark factor construction failed.");
      return false;
    }

    if (!constructGyroMotionFactors(gyro_motions)) {
      __logger__->error(
        "BA gyro motion constraint factor construction failed.");
      return false;
    }

    if (!constructScaleGaugeFactor()) {
      __logger__->error("BA virtual scale gauge factor construction failed.");
      return false;
    }

    return true;
  }

  BundleAdjustmentOptimizationContext::Impl::Impl(
    CyclopsConfig const& config, MultiViewGeometry const& geometry_guess)
      : _config(config), _state(geometry_guess) {
  }

  BundleAdjustmentOptimizationContext::BundleAdjustmentOptimizationContext(
    CyclopsConfig const& config, MultiViewGeometry const& geometry_guess)
      : _pimpl(std::make_unique<Impl>(config, geometry_guess)) {
  }

  BundleAdjustmentOptimizationContext::~BundleAdjustmentOptimizationContext() =
    default;

  ceres::Problem& BundleAdjustmentOptimizationContext::problem() {
    return _pimpl->_problem;
  }

  BundleAdjustmentOptimizationState&
  BundleAdjustmentOptimizationContext::state() {
    return _pimpl->_state;
  }

  std::map<LandmarkID, double*>&
  BundleAdjustmentOptimizationContext::landmarks() {
    return _pimpl->_landmarks;
  }

  std::map<FrameID, double*>& BundleAdjustmentOptimizationContext::frames() {
    return _pimpl->_frames;
  }

  std::vector<ceres::ResidualBlockId>&
  BundleAdjustmentOptimizationContext::residuals() {
    return _pimpl->_residuals;
  }

  int BundleAdjustmentOptimizationContext::nLandmarkMeasurements() const {
    return _pimpl->_n_landmark_measurements;
  }

  int BundleAdjustmentOptimizationContext::nGyroMotionConstraints() const {
    return _pimpl->_n_gyro_motion_constraints;
  }

  bool BundleAdjustmentOptimizationContext::construct(
    MultiViewImageData const& features,
    MultiViewGyroMotionData const& gyro_motions) {
    return _pimpl->construct(features, gyro_motions);
  }
}  // namespace cyclops::initializer
