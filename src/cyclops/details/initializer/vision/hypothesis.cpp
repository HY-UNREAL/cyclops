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

    std::tuple<bool, double> testMotionIMURotationPrior(
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

  std::tuple<bool, double>
  TwoViewMotionHypothesisSelectorImpl::testMotionIMURotationPrior(
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

    return std::make_tuple(p_value > min_p_value, p_value);
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

    auto gyro_tests =  //
      motions | views::transform([&](auto const& motion) {
        return testMotionIMURotationPrior(motion, prior);
      }) |
      ranges::to_vector;

    auto _1 = [&](auto const& pair) { return std::get<0>(pair); };
    if (!ranges::any_of(gyro_tests, _1)) {
      __logger__->warn("Visual motion does not align with the IMU rotation");
      __logger__->info(
        "Suggestion: ensure that the IMU-camera extrinsic is correct.");
    }
    __logger__->debug(
      "Candidates that match IMU rotation: {}",
      ranges::count_if(gyro_tests, _1));

    auto triangulations =
      views::zip(motions, gyro_tests) |
      views::transform(
        [&](auto const& pair) -> std::optional<TwoViewTriangulation> {
          auto const& [motion, test] = pair;
          auto const& [acceptable, _] = test;

          if (!acceptable)
            return std::nullopt;
          return triangulateTwoViewFeaturePairs(
            _config->initialization.vision, features, inliers, motion);
        }) |
      ranges::to_vector;

    return  //
      views::zip(motions, gyro_tests, triangulations) |
      views::transform([&](auto const& pair) {
        auto const& [motion, gyro_test, triangulation] = pair;
        auto const& [R, p] = motion;
        auto [gyro_acceptable, gyro_p_value] = gyro_test;

        auto triangulation_acceptable = triangulation.has_value() &&
          testTriangulationSuccess(*triangulation, inliers.size());

        return TwoViewGeometry {
          .acceptable = gyro_acceptable && triangulation_acceptable,
          .gyro_prior_test_passed = gyro_acceptable,
          .triangulation_test_passed = triangulation_acceptable,
          .gyro_prior_p_value = gyro_p_value,

          .camera_motion = SE3Transform {p, Eigen::Quaterniond(R)},
          .landmarks =
            triangulation ? triangulation->landmarks : LandmarkPositions {},
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
