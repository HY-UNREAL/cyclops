#include "cyclops/details/initializer/vision_imu/acceptance.hpp"
#include "cyclops/details/initializer/vision_imu/rotation.hpp"
#include "cyclops/details/initializer/vision_imu/translation.hpp"
#include "cyclops/details/initializer/vision_imu/uncertainty.hpp"

#include "cyclops/details/telemetry/initializer.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <range/v3/all.hpp>
#include <spdlog/spdlog.h>

namespace cyclops::initializer {
  using telemetry::InitializerTelemetry;
  using RejectReason = InitializerTelemetry::ImuMatchCandidateRejectReason;

  namespace views = ranges::views;

  using namespace std::placeholders;

  static InitializerTelemetry::ImuMatchSolutionPoint makeTelemetrySolutionPoint(
    ImuRotationMatch const& rotation_match,
    ImuTranslationMatchSolution const& translation_match) {
    return {
      .scale = translation_match.scale,
      .cost = translation_match.cost,

      .gravity = translation_match.gravity,
      .acc_bias = translation_match.acc_bias,
      .gyr_bias = rotation_match.gyro_bias,
      .imu_orientations = rotation_match.body_orientations,
      .imu_body_velocities = translation_match.imu_body_velocities,
      .sfm_positions = translation_match.sfm_positions,
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

  static auto maxVelocity(ImuTranslationMatchSolution const& solution) {
    double result = 1e-6;
    for (auto const& [_, v] : solution.imu_body_velocities)
      result = std::max<double>(v.norm(), result);
    return result;
  }

  class ImuTranslationMatchAcceptDiscriminatorImpl:
      public ImuTranslationMatchAcceptDiscriminator {
  private:
    std::shared_ptr<CyclopsConfig const> _config;
    std::shared_ptr<telemetry::InitializerTelemetry> _telemetry;

    std::optional<InitializerTelemetry::ImuMatchUncertainty>
    makeSolutionUncertaintyReport(
      ImuTranslationMatchCandidate const& candidate) const;

    InitializerTelemetry::ImuMatchReject makeRejectReport(
      RejectReason reason, ImuRotationMatch const& rotation_match,
      ImuTranslationMatchCandidate const& candidate) const;

    bool determineCandidateAcceptance(
      ImuRotationMatch const& rotation_match,
      ImuTranslationMatchCandidate const& candidate) const;
    bool determineSolutionAcceptance(
      ImuRotationMatch const& rotation_match,
      ImuTranslationMatchCandidate const& candidate) const;

  public:
    explicit ImuTranslationMatchAcceptDiscriminatorImpl(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry)
        : _config(config), _telemetry(telemetry) {
    }
    void reset() override;

    std::optional<ImuTranslationMatch> determineAcceptance(
      ImuRotationMatch const& rotation_match,
      std::vector<ImuTranslationMatchCandidate> const& candidates)
      const override;
  };

  std::optional<InitializerTelemetry::ImuMatchUncertainty>
  ImuTranslationMatchAcceptDiscriminatorImpl::makeSolutionUncertaintyReport(
    ImuTranslationMatchCandidate const& candidate) const {
    auto const& [solution, maybe_uncertainty] = candidate;
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

  void ImuTranslationMatchAcceptDiscriminatorImpl::reset() {
    _telemetry->reset();
  }

  InitializerTelemetry::ImuMatchReject
  ImuTranslationMatchAcceptDiscriminatorImpl::makeRejectReport(
    RejectReason reason, ImuRotationMatch const& rotation_match,
    ImuTranslationMatchCandidate const& candidate) const {
    auto const& [solution, _] = candidate;

    return InitializerTelemetry::ImuMatchReject {
      .reason = reason,
      .solution = makeTelemetrySolutionPoint(rotation_match, solution),
      .uncertainty = makeSolutionUncertaintyReport(candidate),
    };
  }

  bool ImuTranslationMatchAcceptDiscriminatorImpl::determineCandidateAcceptance(
    ImuRotationMatch const& rotation_match,
    ImuTranslationMatchCandidate const& candidate) const {
    auto report = [&](auto reason) {
      _telemetry->onImuMatchCandidateReject(
        makeRejectReport(reason, rotation_match, candidate));
    };

    auto const& [solution, uncertainty] = candidate;

    if (!uncertainty.has_value()) {
      report(RejectReason::UNCERTAINTY_EVALUATION_FAILED);
      return false;
    }

    if (solution.scale <= 0) {
      report(RejectReason::SCALE_LESS_THAN_ZERO);
      return false;
    }

    auto P = uncertainty->final_cost_significant_probability;
    auto rho = _config->initialization.imu.candidate_test.cost_significance;
    if (P < rho) {
      report(RejectReason::COST_PROBABILITY_INSIGNIFICANT);
      return false;
    }

    return true;
  }

  bool ImuTranslationMatchAcceptDiscriminatorImpl::determineSolutionAcceptance(
    ImuRotationMatch const& rotation_match,
    ImuTranslationMatchCandidate const& candidate) const {
    auto report = [&](auto reason) {
      _telemetry->onImuMatchReject(
        makeRejectReport(reason, rotation_match, candidate));
    };

    auto const& [solution, uncertainty] = candidate;
    if (solution.scale <= 0) {
      report(RejectReason::SCALE_LESS_THAN_ZERO);
      return false;
    }

    auto const& threshold = _config->initialization.imu.acceptance_test;

    auto P = uncertainty->final_cost_significant_probability;
    auto rho = threshold.translation_match_min_p_value;
    if (P < rho) {
      report(RejectReason::COST_PROBABILITY_INSIGNIFICANT);
      return false;
    }

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
      .solution = makeTelemetrySolutionPoint(rotation_match, solution),
      .uncertainty = makeSolutionUncertaintyReport(candidate).value(),
    });
    return true;
  }

  std::optional<ImuTranslationMatch>
  ImuTranslationMatchAcceptDiscriminatorImpl::determineAcceptance(
    ImuRotationMatch const& rotation_match,
    std::vector<ImuTranslationMatchCandidate> const& candidates) const {
    using This = ImuTranslationMatchAcceptDiscriminatorImpl;

    auto acceptables =  //
      candidates |
      views::filter(std::bind(
        &This::determineCandidateAcceptance, this, rotation_match, _1)) |
      ranges::to_vector;

    if (acceptables.size() != 1) {
      auto solutions =  //
        acceptables | views::transform([&](auto const& _) {
          auto const& [solution, uncertainty] = _;
          return makeTelemetrySolutionPoint(rotation_match, solution);
        }) |
        ranges::to_vector;
      auto uncertainties =  //
        acceptables |
        views::transform(
          std::bind(&This::makeSolutionUncertaintyReport, this, _1)) |
        views::transform([](auto const& _) { return _.value(); }) |
        ranges::to_vector;

      _telemetry->onImuMatchAmbiguity({solutions, uncertainties});
      return std::nullopt;
    }

    auto const& candidate = acceptables.front();
    auto const& [solution, uncertainty] = candidate;

    return ImuTranslationMatch {
      .accept = determineSolutionAcceptance(rotation_match, candidate),
      .solution = solution,
    };
  }

  std::unique_ptr<ImuTranslationMatchAcceptDiscriminator>
  ImuTranslationMatchAcceptDiscriminator::Create(
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<telemetry::InitializerTelemetry> telemetry) {
    return std::make_unique<ImuTranslationMatchAcceptDiscriminatorImpl>(
      config, telemetry);
  }
}  // namespace cyclops::initializer
