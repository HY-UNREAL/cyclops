#include "cyclops/details/initializer/vision_imu.hpp"
#include "cyclops/details/initializer/vision_imu/motion_prior.hpp"
#include "cyclops/details/initializer/vision_imu/solver.hpp"
#include "cyclops/details/initializer/vision_imu/solver.imu_only.hpp"
#include "cyclops/details/initializer/vision_imu/type.hpp"

#include "cyclops/details/initializer/vision/type.hpp"
#include "cyclops/details/telemetry/initializer.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>
#include <range/v3/all.hpp>

namespace cyclops::initializer {
  using measurement::ImuMotionRefs;

  class VisionImuInitializerImpl: public VisionImuInitializer {
  private:
    std::unique_ptr<ImuMatchSolver> _matcher;
    std::shared_ptr<CyclopsConfig const> _config;

  public:
    VisionImuInitializerImpl(
      std::unique_ptr<ImuMatchSolver> matcher,
      std::shared_ptr<CyclopsConfig const> config);
    void reset() override;

    std::optional<std::vector<ImuMatchResult>> solve(
      MSfMSolution const& msfm, ImuMotionRefs const& imu_motions) override;
  };

  VisionImuInitializerImpl::VisionImuInitializerImpl(
    std::unique_ptr<ImuMatchSolver> matcher,
    std::shared_ptr<CyclopsConfig const> config)
      : _matcher(std::move(matcher)), _config(config) {
  }

  void VisionImuInitializerImpl::reset() {
    _matcher->reset();
  }

  std::optional<std::vector<ImuMatchResult>> VisionImuInitializerImpl::solve(
    MSfMSolution const& msfm, ImuMotionRefs const& imu_motions) {
    auto const& camera_motions = msfm.camera_motions;
    auto solvable_imu_motions =
      imu_motions | ranges::views::filter([&](auto const& motion_ref) {
        auto const& motion = motion_ref.get();
        return  //
          camera_motions.find(motion.from) != camera_motions.end() &&
          camera_motions.find(motion.to) != camera_motions.end();
      }) |
      ranges::to<ImuMotionRefs>;

    auto n_imu = static_cast<int>(solvable_imu_motions.size());
    auto n_sfm = static_cast<int>(camera_motions.size());
    if (n_sfm != n_imu + 1) {
      __logger__->error("Unmatching number of motion frames (SfM vs IMU)");
      __logger__->error(
        "SfM motion frames ({}) != IMU motion frames + 1 ({} + 1)",  //
        n_sfm, n_imu);
      return std::nullopt;
    }

    auto const& extrinsic = _config->extrinsics.imu_camera_transform;
    auto camera_prior = makeImuMatchCameraMotionPrior(msfm, extrinsic);
    return _matcher->solve(solvable_imu_motions, camera_prior);
  }

  std::unique_ptr<VisionImuInitializer> VisionImuInitializer::Create(
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<telemetry::InitializerTelemetry> telemetry) {
    auto matcher = config->initialization.imu.imu_only
      ? ImuOnlyMatchSolver::Create(config, telemetry)
      : ImuMatchSolver::Create(config, telemetry);

    return std::make_unique<VisionImuInitializerImpl>(
      std::move(matcher), config);
  }
}  // namespace cyclops::initializer
