#include "cyclops_tests/data/imu.hpp"
#include "cyclops_tests/data/landmark.hpp"
#include "cyclops_tests/data/rotation.hpp"
#include "cyclops_tests/default.hpp"
#include "cyclops_tests/random.hpp"
#include "cyclops_tests/range.ipp"

#include "cyclops/details/initializer/vision.hpp"

#include "cyclops/details/initializer/vision/multiview.hpp"
#include "cyclops/details/initializer/vision/type.hpp"
#include "cyclops/details/telemetry/initializer.hpp"
#include "cyclops/details/utils/math.hpp"

#include "cyclops/details/config.hpp"

#include <range/v3/all.hpp>
#include <doctest/doctest.h>

namespace cyclops::initializer {
  namespace views = ranges::views;

  using cyclops::telemetry::InitializerTelemetry;

  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  static auto positionSignal(Timestamp t) {
    auto x = 3 * (1 - std::cos(t));
    return Vector3d(x, 0, 0);
  }

  static auto orientationSignal(Timestamp t) -> Quaterniond {
    auto theta = atan2(1, cos(t));
    auto q0 = makeDefaultCameraRotation();
    return Eigen::AngleAxisd(theta, Vector3d::UnitZ()) * q0;
  }

  TEST_CASE("Test vision initializer") {
    auto rgen = std::make_shared<std::mt19937>(2021052403);

    auto pose_signal = PoseSignal {
      .position = positionSignal,
      .orientation = orientationSignal,
    };
    auto extrinsic = SE3Transform::Identity();
    auto timestamps = linspace(0, M_PI_2, 16) | ranges::to_vector;
    auto timestamp_lookup =
      makeDictionary<FrameID, Timestamp>(views::enumerate(timestamps));

    auto motion_frames = timestamp_lookup | views::keys | ranges::to_vector;

    auto landmarks = generateLandmarks(
      *rgen, {200, Vector3d(3, 3, 0), Vector3d(1, 1, 1).asDiagonal()});

    auto multiview_image_data = makeLandmarkMultiviewObservation(
      pose_signal, extrinsic, landmarks, timestamp_lookup);
    auto multiview_rotation_prior =
      makeMultiViewRotationPrior(pose_signal, extrinsic, timestamp_lookup);

    auto config = makeDefaultConfig();
    config->initialization.vision.feature_point_isotropic_noise = 0.005;
    auto solver = VisionInitializer::Create(
      config, rgen, InitializerTelemetry::CreateDefault());
    auto solutions =
      solver->solve(multiview_image_data, multiview_rotation_prior);
    REQUIRE_FALSE(solutions.empty());

    auto const& best_solution = *std::max_element(
      solutions.begin(), solutions.end(), [](auto const& a, auto const& b) {
        return a.landmarks.size() < b.landmarks.size();
      });
    auto const& camera_motions = best_solution.camera_motions;

    REQUIRE(camera_motions.size() != 0);
    REQUIRE(
      (camera_motions | views::keys | ranges::to_vector) == motion_frames);

    auto init_frame = motion_frames.front();
    auto last_frame = motion_frames.back();

    auto distance = [](auto const& a, auto const& b) { return (a - b).norm(); };
    auto result_travel = distance(
      camera_motions.at(last_frame).translation,
      camera_motions.at(init_frame).translation);

    auto init_time = timestamp_lookup.at(init_frame);
    auto last_time = timestamp_lookup.at(last_frame);
    auto truth_travel =
      distance(positionSignal(last_time), positionSignal(init_time));

    REQUIRE(result_travel != 0);
    REQUIRE(truth_travel != 0);

    auto scale = truth_travel / result_travel;
    for (auto const& [frame_id, time] : timestamp_lookup) {
      auto const& obtained_motion = camera_motions.at(frame_id);
      auto truth_motion = compose(
        inverse(pose_signal.evaluate(init_time)), pose_signal.evaluate(time));

      auto const& q_truth = truth_motion.rotation;
      auto const& q_result = obtained_motion.rotation;
      CAPTURE(q_truth.coeffs().transpose());
      CAPTURE(q_result.coeffs().transpose());
      CHECK(q_truth.isApprox(q_result, 0.01));

      auto const& p_truth = truth_motion.translation;
      auto const& p_result = obtained_motion.translation;
      CAPTURE(p_truth.transpose());
      CAPTURE(p_result.transpose());
      CHECK(p_truth.isApprox(p_result * scale, 0.05));
    }
  }
}  // namespace cyclops::initializer
