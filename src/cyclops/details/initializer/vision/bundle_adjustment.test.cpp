#include "cyclops/details/initializer/vision/bundle_adjustment.cpp"
#include "cyclops/details/initializer/vision/type.hpp"
#include "cyclops/details/utils/math.hpp"

#include "cyclops/details/config.hpp"

#include "cyclops_tests/data/landmark.hpp"
#include "cyclops_tests/default.hpp"
#include "cyclops_tests/random.hpp"
#include "cyclops_tests/range.ipp"

#include <range/v3/all.hpp>
#include <doctest/doctest.h>

namespace cyclops::initializer {
  namespace views = ranges::views;

  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  static auto constexpr n_frames = 8;

  static auto positionSignal(Timestamp t) {
    auto x = 3 * (1 - std::cos(t));
    return Vector3d(x, 0, 0);
  }

  static auto orientationSignal(Timestamp t) -> Quaterniond {
    auto theta = atan2(1, cos(t));
    auto q0 = makeDefaultCameraRotation();
    return Eigen::AngleAxisd(theta, Vector3d::UnitZ()) * q0;
  }

  static auto makeMultiViewLandmarkObservation(
    std::mt19937& rgen, PoseSignal pose_signal,
    std::map<FrameID, Timestamp> motion_timestamps) {
    auto landmarks = generateLandmarks(
      rgen, {200, Vector3d(3, 3, 0), Vector3d(1, 1, 1).asDiagonal()});
    auto image_data = makeLandmarkMultiviewObservation(
      pose_signal, SE3Transform::Identity(), landmarks, motion_timestamps);
    return std::make_tuple(landmarks, image_data);
  }

  static auto makeMultiViewGeometryGuess(
    std::mt19937& rgen, std::map<FrameID, Timestamp> motion_timestamps,
    PoseSignal pose_signal, LandmarkPositions const& landmarks) {
    REQUIRE_FALSE(motion_timestamps.empty());
    auto [_, init_time] = *motion_timestamps.begin();
    auto init_pose = pose_signal.evaluate(init_time);

    auto camera_motions_estimated =  //
      motion_timestamps | views::transform([&](auto _) {
        auto [frame_id, time] = _;
        auto [p, q] = compose(inverse(init_pose), pose_signal.evaluate(time));
        auto p_perturbed = perturbate((2 * p).eval(), 0.1, rgen);
        auto q_perturbed = perturbate(q, 0.1, rgen);

        return std::make_pair(frame_id, SE3Transform {p, q});
      }) |
      ranges::to<std::map<FrameID, SE3Transform>>;

    auto landmarks_estimated =  //
      landmarks | views::transform([&](auto const& id_landmark) {
        auto const& [landmark_id, landmark] = id_landmark;
        auto const& [p0, q0] = init_pose;
        auto f = (q0.conjugate() * (landmark - p0)).eval();

        return std::make_pair(
          landmark_id, perturbate((2 * f).eval(), 0.1, rgen));
      }) |
      ranges::to<LandmarkPositions>;

    return MultiViewGeometry {camera_motions_estimated, landmarks_estimated};
  }

  TEST_CASE("Bundle adjustment") {
    std::mt19937 rgen(20240513007);

    auto timestamps = linspace(0, M_PI_2, n_frames) | ranges::to_vector;
    auto motion_frames = views::ints(0, n_frames) | ranges::to_vector;
    auto motion_timestamps =
      makeDictionary<FrameID, Timestamp>(views::zip(motion_frames, timestamps));
    auto pose_signal = PoseSignal {positionSignal, orientationSignal};

    auto [landmarks, image_data] =
      makeMultiViewLandmarkObservation(rgen, pose_signal, motion_timestamps);
    auto geometry_guess = makeMultiViewGeometryGuess(
      rgen, motion_timestamps, pose_signal, landmarks);

    auto noise = SensorStatistics {
      .acc_white_noise = 0.05,
      .gyr_white_noise = 0.001,
      .acc_random_walk = 0.00001,
      .gyr_random_walk = 0.00001,
      .acc_bias_prior_stddev = 0.2,
      .gyr_bias_prior_stddev = 0.1,
    };
    auto extrinsic = SensorExtrinsics {
      .imu_camera_time_delay = 0.,
      .imu_camera_transform = SE3Transform::Identity(),
    };
    std::shared_ptr config = CyclopsConfig::CreateDefault(noise, extrinsic);

    config->initialization.vision.multiview.bundle_adjustment_max_solver_time =
      10;

    auto bundle_adjustment_solver = BundleAdjustmentSolver::Create(config);

    auto maybe_solution =
      bundle_adjustment_solver->solve(geometry_guess, image_data, {});
    REQUIRE(maybe_solution.has_value());

    auto const& camera_motions = maybe_solution->geometry.camera_motions;
    REQUIRE(camera_motions.size() == n_frames);
    REQUIRE(
      (camera_motions | views::keys | ranges::to<std::set>) ==
      (motion_timestamps | views::keys | ranges::to<std::set>));

    THEN("The resulting Fisher information is positive semidefinite") {
      auto const& H = maybe_solution->motion_information_weight;
      REQUIRE(H.rows() == n_frames * 6);
      REQUIRE(H.cols() == n_frames * 6);
      REQUIRE(H.isApprox(H.transpose()));

      auto lambda = H.selfadjointView<Eigen::Upper>().eigenvalues().eval();

      // 1e-6: Margin to avoid the numerical inaccuracy
      REQUIRE(lambda.x() > -1e-6);

      AND_THEN("The resulting Fisher information experiences 6-DoF symmetry") {
        REQUIRE(lambda.head(6).norm() < 1e-6);
        REQUIRE(lambda.head(7).norm() > 1e-6);
      }
    }

    auto distance = [](auto const& a, auto const& b) { return (a - b).norm(); };

    auto init_time = timestamps.front();
    auto last_time = timestamps.back();
    auto true_travel =
      distance(positionSignal(init_time), positionSignal(last_time));

    auto init_frame = motion_frames.front();
    auto last_frame = motion_frames.back();
    auto result_travel = distance(
      camera_motions.at(init_frame).translation,
      camera_motions.at(last_frame).translation);
    auto scale = true_travel / result_travel;

    for (auto const& [frame_id, time] : motion_timestamps) {
      auto result_motion = compose(
        inverse(camera_motions.at(init_frame)), camera_motions.at(frame_id));
      auto true_motion = compose(
        inverse(pose_signal.evaluate(init_time)), pose_signal.evaluate(time));

      auto const& q_true = true_motion.rotation;
      auto const& p_true = true_motion.translation;
      auto const& q_result = result_motion.rotation;
      auto const& p_result = result_motion.translation;

      CAPTURE(frame_id);
      CAPTURE(q_true.coeffs().transpose());
      CAPTURE(q_result.coeffs().transpose());
      CHECK(q_true.isApprox(q_result, 1e-6));

      CAPTURE(p_true.transpose());
      CAPTURE(scale * p_result.transpose());
      CHECK(p_true.isApprox(p_result * scale, 1e-6));
    }
  }
}  // namespace cyclops::initializer
