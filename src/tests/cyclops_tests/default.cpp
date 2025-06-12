#include "cyclops_tests/default.hpp"

#include "cyclops_tests/data/typefwd.hpp"
#include "cyclops_tests/data/landmark.hpp"

#include "cyclops/details/config.hpp"

namespace cyclops {
  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  Quaterniond makeDefaultCameraRotation() {
    Eigen::Matrix3d result;

    // clang-format off
    result <<
      +0, +0, +1,
      -1, +0, +0,
      +0, -1, +0;
    // clang-format on

    return Eigen::Quaterniond(result);
  }

  SE3Transform makeDefaultImuCameraExtrinsic() {
    return {
      .translation = Vector3d(0.1, 0, 0),
      .rotation = makeDefaultCameraRotation(),
    };
  }

  std::shared_ptr<CyclopsConfig> makeDefaultConfig() {
    return CyclopsConfig::CreateDefault(
      SensorStatistics {
        .acc_white_noise = 0.01,
        .gyr_white_noise = 0.003,
        .acc_random_walk = 0.00001,
        .gyr_random_walk = 0.00001,
        .acc_bias_prior_stddev = 0.1,
        .gyr_bias_prior_stddev = 0.1,
      },
      SensorExtrinsics {
        .imu_camera_time_delay = 0,
        .imu_camera_transform = makeDefaultImuCameraExtrinsic(),
      });
  }

  LandmarkGenerationArguments makeDefaultLandmarkSet() {
    return LandmarkGenerationArguments {
      LandmarkGenerationArgument {
        .count = 30,
        .center = Vector3d(1.5, 0.5, 0.8),
        .concentration = Vector3d(0.5, 1.0, 0.8).asDiagonal(),
      },
      LandmarkGenerationArgument {
        .count = 30,
        .center = Vector3d(2.0, 2.8, 0.6),
        .concentration = Vector3d(0.5, 0.5, 0.6).asDiagonal(),
      },
      LandmarkGenerationArgument {
        .count = 30,
        .center = Vector3d(4.5, 0.75, 0.5),
        .concentration = Vector3d(0.5, 0.75, 0.5).asDiagonal(),
      },
      LandmarkGenerationArgument {
        .count = 30,
        .center = Vector3d(3.5, 3.25, 0.75),
        .concentration = Vector3d(0.5, 0.75, 0.75).asDiagonal(),
      },
      LandmarkGenerationArgument {
        .count = 30,
        .center = Vector3d(0.0, 5.0, 0.5),
        // clang-format off
        .concentration = (Eigen::Matrix3d() <<
          -0.5, -2.0, 0,
          +0.5, -2.0, 0,
          +0.0, +0.0, 1
        ).finished(),
        // clang-format on
      },
    };
  }
}  // namespace cyclops
