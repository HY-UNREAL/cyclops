#include "cyclops/details/initializer/vision_imu/acceptance.imu_only.hpp"
#include "cyclops/details/initializer/vision_imu/camera_motion_prior.hpp"
#include "cyclops/details/initializer/vision_imu/translation.hpp"
#include "cyclops/details/initializer/vision_imu/uncertainty.hpp"

#include "cyclops/details/telemetry/initializer.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>

namespace cyclops::initializer {
  using telemetry::InitializerTelemetry;
  using RejectReason = InitializerTelemetry::ImuMatchCandidateRejectReason;

  static InitializerTelemetry::ImuMatchSolutionPoint makeTelemetrySolutionPoint(
    ImuMatchSolution const& match) {
    return {
      .scale = match.scale,
      .cost = match.cost,

      .gravity = match.gravity,
      .acc_bias = match.acc_bias,
      .gyr_bias = match.gyr_bias,
      .imu_orientations = match.body_orientations,
      .imu_body_velocities = match.body_velocities,
      .sfm_positions = match.sfm_positions,
    };
  }

  static bool checkPercentThreshold(
    std::string tag, double value, double threshold) {
    if (value > threshold) {
      __logger__->info(
        "IMU match: {} is uncertain. estimated uncertainty: {}% > {}%.", tag,
        100 * value, 100 * threshold);
      return true;
    }
    return false;
  }

  static auto maxVelocity(ImuMatchSolution const& solution) {
    double result = 1e-6;
    for (auto const& [_, v] : solution.body_velocities)
      result = std::max<double>(v.norm(), result);
    return result;
  }

  class ImuOnlyMatchAcceptDiscriminatorImpl:
      public ImuOnlyMatchAcceptDiscriminator {
  private:
    std::shared_ptr<CyclopsConfig const> _config;
    std::shared_ptr<InitializerTelemetry> _telemetry;

    std::optional<InitializerTelemetry::ImuMatchUncertainty>
    makeSolutionUncertaintyReport(
      ImuMatchCandidate const& match_candidate) const;

    InitializerTelemetry::ImuMatchReject makeRejectReport(
      RejectReason reason, ImuMatchCandidate const& match_candidate) const;

  public:
    ImuOnlyMatchAcceptDiscriminatorImpl(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<InitializerTelemetry> telemetry)
        : _config(config), _telemetry(telemetry) {
    }
    void reset() override;

    bool determineAccept(
      ImuMatchCandidate const& match_candidate) const override;
  };

  std::optional<InitializerTelemetry::ImuMatchUncertainty>
  ImuOnlyMatchAcceptDiscriminatorImpl::makeSolutionUncertaintyReport(
    ImuMatchCandidate const& match_candidate) const {
    auto const& [solution, maybe_uncertainty] = match_candidate;
    if (!maybe_uncertainty.has_value())
      return std::nullopt;

    auto const& uncertainty = maybe_uncertainty.value();

    auto gravity_norm = _config->gravity_norm;
    auto sigma_g = uncertainty.gravity_tangent_deviation(0) / gravity_norm;
    auto sigma_v =
      uncertainty.body_velocity_deviation(1) / maxVelocity(solution);

    return InitializerTelemetry::ImuMatchUncertainty {
      .final_cost_significant_probability =
        uncertainty.final_cost_significant_probability,
      .scale_log_deviation = uncertainty.scale_log_deviation,
      .gravity_max_deviation = sigma_g,
      .bias_max_deviation = uncertainty.bias_deviation(0),
      .body_velocity_max_deviation = sigma_v,
      .scale_symmetric_translation_error_max_deviation =
        uncertainty.translation_scale_symmetric_deviation(0),
    };
  }

  void ImuOnlyMatchAcceptDiscriminatorImpl::reset() {
    _telemetry->reset();
  }

  InitializerTelemetry::ImuMatchReject
  ImuOnlyMatchAcceptDiscriminatorImpl::makeRejectReport(
    RejectReason reason, ImuMatchCandidate const& match_candidate) const {
    auto const& [solution, _] = match_candidate;

    return InitializerTelemetry::ImuMatchReject {
      .reason = reason,
      .solution = makeTelemetrySolutionPoint(solution),
      .uncertainty = makeSolutionUncertaintyReport(match_candidate),
    };
  }

  bool ImuOnlyMatchAcceptDiscriminatorImpl::determineAccept(
    ImuMatchCandidate const& match_candidate) const {
    auto report = [&](auto reason) {
      _telemetry->onImuMatchReject(makeRejectReport(reason, match_candidate));
    };
    auto const& [solution, uncertainty] = match_candidate;

    if (!uncertainty.has_value()) {
      report(RejectReason::UNCERTAINTY_EVALUATION_FAILED);
      return false;
    }

    if (solution.scale <= 0) {
      report(RejectReason::SCALE_LESS_THAN_ZERO);
      return false;
    }

    auto const& threshold = _config->initialization.imu.acceptance_test;

    auto sigma_s = uncertainty->scale_log_deviation;
    auto epsilon_s = threshold.max_scale_log_deviation;
    if (checkPercentThreshold("Scale", sigma_s, epsilon_s)) {
      report(RejectReason::UNDERINFORMATIVE_PARAMETER);
      return false;
    }

    auto gravity = _config->gravity_norm;
    auto sigma_g = uncertainty->gravity_tangent_deviation(0) / gravity;
    auto epsilon_g = threshold.max_normalized_gravity_deviation;
    if (checkPercentThreshold("Gravity direction", sigma_g, epsilon_g)) {
      report(RejectReason::UNDERINFORMATIVE_PARAMETER);
      return false;
    }

    auto sigma_v =
      uncertainty->body_velocity_deviation(1) / maxVelocity(solution);
    auto epsilon_v = threshold.max_normalized_velocity_deviation;
    if (checkPercentThreshold("Velocity", sigma_v, epsilon_v)) {
      report(RejectReason::UNDERINFORMATIVE_PARAMETER);
      return false;
    }

    _telemetry->onImuMatchAccept({
      .solution = makeTelemetrySolutionPoint(solution),
      .uncertainty = makeSolutionUncertaintyReport(match_candidate).value(),
    });
    return true;
  }

  std::unique_ptr<ImuOnlyMatchAcceptDiscriminator>
  ImuOnlyMatchAcceptDiscriminator::Create(
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<InitializerTelemetry> telemetry) {
    return std::make_unique<ImuOnlyMatchAcceptDiscriminatorImpl>(
      config, telemetry);
  }
}  // namespace cyclops::initializer
