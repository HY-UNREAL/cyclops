#include "cyclops_tests/data/imu.hpp"
#include "cyclops_tests/data/landmark.hpp"
#include "cyclops_tests/default.hpp"
#include "cyclops_tests/random.hpp"
#include "cyclops_tests/range.ipp"
#include "cyclops_tests/signal.ipp"

#include "cyclops/details/initializer/vision/type.hpp"
#include "cyclops/details/initializer/vision_imu.hpp"
#include "cyclops/details/measurement/preintegration.hpp"
#include "cyclops/details/utils/math.hpp"
#include "cyclops/details/config.hpp"

#include "cyclops/details/telemetry/initializer.hpp"

#include <range/v3/all.hpp>

#include <doctest/doctest.h>

namespace cyclops::initializer {
  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  namespace views = ranges::views;

  static auto constexpr n_frames = 12;

  static auto makeBodyCameraMotionData(
    PoseSignal pose_signal, SE3Transform const& extrinsic, double scale) {
    auto velocity_signal = numericDerivative<Vector3d>(pose_signal.position);

    auto timestamps = linspace(0, M_PI, n_frames) | ranges::to_vector;
    auto motion_frame_ids = views::ints(0, n_frames) | ranges::to_vector;
    auto motion_timestamp_lookup =  //
      views::zip(motion_frame_ids, timestamps) | views::transform([](auto _) {
        auto [id, time] = _;
        return std::make_pair(id, time);
      }) |
      ranges::to<std::map<FrameID, Timestamp>>;

    auto body_motions =  //
      motion_frame_ids | views::transform([&](auto frame_id) {
        auto time = motion_timestamp_lookup.at(frame_id);

        auto x0 = pose_signal.evaluate(timestamps.front());
        auto [p, q] = compose(inverse(x0), pose_signal.evaluate(time));
        auto v = (x0.rotation.conjugate() * velocity_signal(time)).eval();

        return std::make_pair(frame_id, ImuMotionState {q, p, v});
      }) |
      ranges::to<std::map<FrameID, ImuMotionState>>;

    auto camera_motions =  //
      body_motions | views::transform([&](auto const& lookup) {
        auto const& [frame_id, imu_pose] = lookup;
        auto const& [q, p, v] = imu_pose;
        auto camera_pose = compose({p, q}, extrinsic);
        camera_pose.translation *= 1 / scale;

        return std::make_pair(frame_id, camera_pose);
      }) |
      ranges::to<std::map<FrameID, SE3Transform>>;

    return std::make_tuple(
      motion_timestamp_lookup, body_motions, camera_motions);
  }

  TEST_CASE("IMU bootstrap solver") {
    auto rgen = std::make_shared<std::mt19937>(20240513007);
    auto config = makeDefaultConfig();
    auto const& extrinsic = config->extrinsics.imu_camera_transform;

    GIVEN("Sinusoidal position signal") {
      auto rotation_axis =
        perturbate(Vector3d::Zero().eval(), 1, *rgen).normalized().eval();
      auto pose_signal = PoseSignal {
        .position = [](Timestamp t) -> Vector3d {
          auto x = t - std::sin(4 * M_PI * t) / (8 * M_PI);
          return Vector3d(x, 0, 0);
        },
        .orientation = [rotation_axis](Timestamp t) -> Quaterniond {
          auto theta = t * t / 2 + M_PI_2;
          return Quaterniond(Eigen::AngleAxisd(theta, rotation_axis));
        },
      };

      GIVEN("Monocular camera motion sensed in the half-scale gauge") {
        auto scale_solution = 2.0;
        auto [motion_timestamps, body_motions, camera_motions] =
          makeBodyCameraMotionData(pose_signal, extrinsic, scale_solution);

        GIVEN("The vision SfM and the IMU sensing without noise") {
          auto msfm = MSfMSolution {
            .acceptable = true,
            .solution_significant_probability = 1.0,
            .measurement_inlier_ratio = 1.0,
            .geometry = {camera_motions, {}},
            .motion_information_weight =
              1e4 * Eigen::MatrixXd::Identity(n_frames * 6, n_frames * 6),
          };
          auto imu_motions = makeImuMotions(pose_signal, motion_timestamps);
          auto imu_motion_refs = imu_motions |
            views::transform([](auto const& _) { return std::ref(_); }) |
            ranges::to_vector;

          WHEN("Solved IMU matching") {
            auto telemetry = telemetry::InitializerTelemetry::CreateDefault();
            auto solver =
              VisionImuInitializer::Create(config, std::move(telemetry));
            auto maybe_result = solver->solve(msfm, imu_motion_refs);
            REQUIRE(static_cast<bool>(maybe_result));

            THEN("Solution is correct up to numerical accuracy") {
              auto const& result = *maybe_result;
              REQUIRE(result.size() == 1);

              CHECK(
                result.front().solution.scale ==
                doctest::Approx(scale_solution).epsilon(1e-3));
            }
          }
        }
      }
    }
  }
}  // namespace cyclops::initializer
