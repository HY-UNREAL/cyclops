#include "cyclops/details/initializer/vision_imu.hpp"
#include "cyclops/details/initializer/vision_imu/camera_motion_prior.hpp"
#include "cyclops/details/initializer/vision_imu/rotation.hpp"
#include "cyclops/details/initializer/vision_imu/translation.hpp"
#include "cyclops/details/initializer/vision_imu/translation.imu_only.hpp"

#include "cyclops/details/initializer/vision/type.hpp"
#include "cyclops/details/telemetry/initializer.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>
#include <range/v3/all.hpp>

namespace cyclops::initializer {
  namespace views = ranges::views;

  class ImuMatchSolverImpl: public ImuMatchSolver {
  private:
    std::unique_ptr<ImuRotationMatchSolver> _rotation_solver;
    std::unique_ptr<ImuTranslationMatchSolver> _translation_solver;

    std::shared_ptr<CyclopsConfig const> _config;

  public:
    ImuMatchSolverImpl(
      std::unique_ptr<ImuRotationMatchSolver> rotation_solver,
      std::unique_ptr<ImuTranslationMatchSolver> translation_solver,
      std::shared_ptr<CyclopsConfig const> config);
    void reset() override;

    std::optional<ImuMatchSolution> solve(
      MSfMSolution const& msfm,
      measurement::ImuMotionRefs const& imu_motions) override;
  };

  ImuMatchSolverImpl::ImuMatchSolverImpl(
    std::unique_ptr<ImuRotationMatchSolver> rotation_solver,
    std::unique_ptr<ImuTranslationMatchSolver> translation_solver,
    std::shared_ptr<CyclopsConfig const> config)
      : _rotation_solver(std::move(rotation_solver)),
        _translation_solver(std::move(translation_solver)),
        _config(config) {
  }

  void ImuMatchSolverImpl::reset() {
    _rotation_solver->reset();
    _translation_solver->reset();
  }

  std::optional<ImuMatchSolution> ImuMatchSolverImpl::solve(
    MSfMSolution const& msfm, measurement::ImuMotionRefs const& imu_motions) {
    auto const& camera_motions = msfm.geometry.camera_motions;
    auto solvable_imu_motions =  //
      imu_motions | views::filter([&](auto const& motion_ref) {
        auto const& motion = motion_ref.get();
        return  //
          camera_motions.find(motion.from) != camera_motions.end() &&
          camera_motions.find(motion.to) != camera_motions.end();
      }) |
      ranges::to<measurement::ImuMotionRefs>;

    auto n_imu = static_cast<int>(solvable_imu_motions.size());
    auto n_sfm = static_cast<int>(camera_motions.size());
    if (n_sfm != n_imu + 1) {
      __logger__->error("Unmatching number of motion frames (SfM vs IMU)");
      __logger__->error(
        "SfM motion frames ({}) != IMU motion frames + 1 ({} + 1)",  //
        n_sfm, n_imu);
      return std::nullopt;
    }
    auto [rotation_prior, translation_prior] =
      makeImuMatchCameraMotionPrior(msfm);

    auto rotation_match =
      _rotation_solver->solve(solvable_imu_motions, rotation_prior);
    if (!rotation_match.has_value()) {
      __logger__->info("IMU match rotation solver failed.");
      return std::nullopt;
    }

    auto translation_match = _translation_solver->solve(
      solvable_imu_motions, *rotation_match, translation_prior);
    if (!translation_match.has_value()) {
      __logger__->info("IMU match translation solver failed.");
      return std::nullopt;
    }
    return ImuMatchSolution {*rotation_match, *translation_match};
  }

  std::unique_ptr<ImuMatchSolver> ImuMatchSolver::Create(
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<telemetry::InitializerTelemetry> telemetry) {
    auto rotation_solver = ImuRotationMatchSolver::Create(config);
    auto translation_solver = config->initialization.imu.imu_only
      ? ImuOnlyTranslationMatchSolver::Create(config, telemetry)
      : ImuTranslationMatchSolver::Create(config, telemetry);

    return std::make_unique<ImuMatchSolverImpl>(
      std::move(rotation_solver), std::move(translation_solver), config);
  }
}  // namespace cyclops::initializer
