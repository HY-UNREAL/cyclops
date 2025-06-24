#include "cyclops/details/initializer/vision/hypothesis.hpp"
#include "cyclops/details/initializer/vision/triangulation.hpp"
#include "cyclops/details/initializer/vision/twoview.hpp"

#include "cyclops/details/utils/math.hpp"
#include "cyclops/details/utils/vision.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>
#include <range/v3/all.hpp>

namespace cyclops::initializer {
  namespace views = ranges::views;

  class TwoViewMotionHypothesisSelectorImpl:
      public TwoViewMotionHypothesisSelector {
  private:
    std::shared_ptr<CyclopsConfig const> _config;

    bool testMotionIMURotationPrior(
      RotationPositionPair const& motion,
      TwoViewImuRotationData const& imu_prior) const;
    bool testTriangulationSuccess(
      TwoViewTriangulation const& triangulation, int inliers) const;

  public:
    explicit TwoViewMotionHypothesisSelectorImpl(
      std::shared_ptr<CyclopsConfig const> config);
    void reset() override;

    std::vector<TwoViewGeometry> selectPossibleMotions(
      MotionHypotheses const& motions, TwoViewFeatureSet const& image_data,
      InlierSet const& inliers, TwoViewImuRotationData const& prior) override;
  };

  TwoViewMotionHypothesisSelectorImpl::TwoViewMotionHypothesisSelectorImpl(
    std::shared_ptr<CyclopsConfig const> config)
      : _config(config) {
  }

  void TwoViewMotionHypothesisSelectorImpl::reset() {
    // Does nothing
  }

  static Eigen::Vector3d so3Log(Eigen::Matrix3d const& R) {
    auto w = Eigen::AngleAxisd(R);
    return w.angle() * w.axis();
  }

  bool TwoViewMotionHypothesisSelectorImpl::testMotionIMURotationPrior(
    RotationPositionPair const& motion,
    TwoViewImuRotationData const& imu_prior) const {
    auto const& vision_config = _config->initialization.vision;
    auto const& hypothesis_config = vision_config.two_view.motion_hypothesis;
    auto min_p_value = hypothesis_config.min_imu_rotation_consistency_p_value;

    auto R_hat = imu_prior.value.matrix().eval();
    auto v = so3Log(R_hat.transpose() * motion.rotation);

    auto llt = Eigen::LDLT<Eigen::Matrix3d>(imu_prior.covariance);
    auto error = v.dot(llt.solve(v));

    auto p_value = 1.0 - chiSquaredCdf(3, error);

    return p_value > min_p_value;
  }

  bool TwoViewMotionHypothesisSelectorImpl::testTriangulationSuccess(
    TwoViewTriangulation const& triangulation, int inliers) const {
    auto const& config =
      _config->initialization.vision.two_view.motion_hypothesis;

    auto const success = triangulation.landmarks.size();
    auto const min_success = config.min_triangulation_success;

    __logger__->trace("Success: {}, min success: {}", success, min_success);
    __logger__->trace("Expected inliers: {}", triangulation.expected_inliers);

    return success >= min_success;
  }

  std::vector<TwoViewGeometry>
  TwoViewMotionHypothesisSelectorImpl::selectPossibleMotions(
    MotionHypotheses const& motions, TwoViewFeatureSet const& features,
    InlierSet const& inliers, TwoViewImuRotationData const& prior) {
    __logger__->debug("Selecting best two-view motion hypothesis");
    __logger__->debug("Motion candidates: {}", motions.size());

    auto motions_filtered =  //
      motions | views::filter([&](auto const& motion) {
        return testMotionIMURotationPrior(motion, prior);
      }) |
      ranges::to_vector;

    if (motions_filtered.empty()) {
      __logger__->warn("Visual motion does not align with the IMU rotation");
      __logger__->info(
        "Suggestion: ensure that the IMU-camera extrinsic is correct.");
    }
    __logger__->debug(
      "Candidates that match IMU rotation: {}", motions_filtered.size());

    auto motion_triangulations =
      motions_filtered | views::transform([&](auto const& motion) {
        auto triangulation = triangulateTwoViewFeaturePairs(
          _config->initialization.vision, features, inliers, motion);
        return std::make_tuple(motion, triangulation);
      }) |
      ranges::to_vector;

    auto successed_motion_triangulations =
      motion_triangulations | views::filter([&](auto const& pair) {
        auto const& [motion, triangulation] = pair;
        return testTriangulationSuccess(triangulation, inliers.size());
      }) |
      ranges::to_vector;
    __logger__->debug(
      "Triangulation successes: {}", successed_motion_triangulations.size());

    return  //
      successed_motion_triangulations | views::transform([](auto const& pair) {
        auto const& [motion, triangulation] = pair;
        auto const& [R, p] = motion;
        return TwoViewGeometry {
          .camera_motion = SE3Transform {p, Eigen::Quaterniond(R)},
          .landmarks = triangulation.landmarks,
        };
      }) |
      ranges::to_vector;
  }

  std::unique_ptr<TwoViewMotionHypothesisSelector>
  TwoViewMotionHypothesisSelector::Create(
    std::shared_ptr<CyclopsConfig const> config) {
    return std::make_unique<TwoViewMotionHypothesisSelectorImpl>(config);
  }
}  // namespace cyclops::initializer
