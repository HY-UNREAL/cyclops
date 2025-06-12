#include "cyclops/details/estimation/sanity.hpp"

#include "cyclops/details/telemetry/optimizer.hpp"
#include "cyclops/details/utils/math.hpp"

#include "cyclops/details/config.hpp"

namespace cyclops::estimation {
  class EstimationSanityDiscriminatorImpl:
      public EstimationSanityDiscriminator {
  private:
    std::shared_ptr<CyclopsConfig const> _config;
    std::shared_ptr<telemetry::OptimizerTelemetry> _telemetry;

    bool _sanity = true;
    int _landmark_update_failure_count = 0;
    int _final_cost_sanity_failure_count = 0;

    bool updateFinalCostSanityBadness(double cost_significant_probability);
    bool updateLandmarkSanityBadness(double accept_rate, int n_accepts);

  public:
    EstimationSanityDiscriminatorImpl(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<telemetry::OptimizerTelemetry> telemetry);

    void reset() override;

    void update(
      LandmarkSanityStatistics const& landmark_sanity,
      OptimizerSanityStatistics const& optimizer_sanity) override;
    bool sanity() const override;
  };

  void EstimationSanityDiscriminatorImpl::reset() {
    _sanity = true;
    _landmark_update_failure_count = 0;
    _final_cost_sanity_failure_count = 0;

    _telemetry->reset();
  }

  static auto makeTelemetryReport(
    LandmarkSanityStatistics const& landmark_sanity,
    OptimizerSanityStatistics const& optimizer_sanity) {
    auto [cost, num_residuals, num_parameters] = optimizer_sanity;
    auto n_landmarks = landmark_sanity.landmark_observations;

    auto degrees_of_freedom = num_residuals - num_parameters;

    double landmark_accepts = landmark_sanity.landmark_accepts;
    double uninitialized_landmarks = landmark_sanity.uninitialized_landmarks;
    double depth_failures = landmark_sanity.depth_threshold_failures;
    double mnorm_failures = landmark_sanity.mnorm_threshold_failures;

    return telemetry::OptimizerTelemetry::SanityStatistics {
      .final_cost = cost,
      .final_cost_significant_probability =
        1. - chiSquaredCdf(degrees_of_freedom, cost),

      .landmark_observations = n_landmarks,
      .landmark_accept_rate = landmark_accepts / n_landmarks,
      .landmark_uninitialized_rate = uninitialized_landmarks / n_landmarks,
      .landmark_depth_threshold_failure_rate = depth_failures / n_landmarks,
      .landmark_chi_square_test_failure_rate = mnorm_failures / n_landmarks,
    };
  }

  bool EstimationSanityDiscriminatorImpl::updateFinalCostSanityBadness(
    double cost_significant_probability) {
    auto const& threshold = _config->estimation.fault_detection;

    if (threshold.max_final_cost_sanity_failures < 0) {
      _final_cost_sanity_failure_count = 0;
      return false;
    }

    auto final_cost_bad =
      cost_significant_probability < threshold.min_final_cost_p_value;
    if (final_cost_bad) {
      _final_cost_sanity_failure_count++;
    } else {
      _final_cost_sanity_failure_count = 0;
    }
    return final_cost_bad;
  }

  bool EstimationSanityDiscriminatorImpl::updateLandmarkSanityBadness(
    double accept_rate, int n_accepts) {
    auto const& threshold = _config->estimation.fault_detection;

    auto landmark_bad =
      accept_rate < threshold.min_landmark_accept_rate || n_accepts < 8;
    if (landmark_bad) {
      _landmark_update_failure_count++;
    } else {
      _landmark_update_failure_count = 0;
    }
    return landmark_bad;
  }

  void EstimationSanityDiscriminatorImpl::update(
    LandmarkSanityStatistics const& landmark_sanity,
    OptimizerSanityStatistics const& optimizer_sanity) {
    auto sanity_report = makeTelemetryReport(landmark_sanity, optimizer_sanity);
    _telemetry->onSanityStatistics(sanity_report);

    auto final_cost_bad = updateFinalCostSanityBadness(
      sanity_report.final_cost_significant_probability);
    auto landmark_bad = updateLandmarkSanityBadness(
      sanity_report.landmark_accept_rate, landmark_sanity.landmark_accepts);

    if (landmark_bad || final_cost_bad) {
      auto reason = telemetry::OptimizerTelemetry::BadReason {
        .bad_landmark_update = landmark_bad,
        .bad_final_cost = final_cost_bad,
      };
      _telemetry->onSanityBad(reason, sanity_report);
    }

    auto const& threshold = _config->estimation.fault_detection;
    auto landmark_failed =
      _landmark_update_failure_count > threshold.max_landmark_update_failures;

    auto max_final_cost_failure = threshold.max_final_cost_sanity_failures;
    auto final_cost_failed = max_final_cost_failure >= 0 &&
      _final_cost_sanity_failure_count > max_final_cost_failure;

    if (landmark_failed || final_cost_failed) {
      auto reason = telemetry::OptimizerTelemetry::FailureReason {
        .continued_bad_landmark_update = landmark_failed,
        .continued_bad_final_cost = final_cost_failed,
      };
      _telemetry->onSanityFailure(reason, sanity_report);
      _sanity = false;
    } else {
      _sanity = true;
    }
  }

  bool EstimationSanityDiscriminatorImpl::sanity() const {
    return _sanity;
  }

  EstimationSanityDiscriminatorImpl::EstimationSanityDiscriminatorImpl(
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<telemetry::OptimizerTelemetry> telemetry)
      : _config(config), _telemetry(telemetry) {
  }

  std::unique_ptr<EstimationSanityDiscriminator>
  EstimationSanityDiscriminator::Create(
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<telemetry::OptimizerTelemetry> telemetry) {
    return std::make_unique<EstimationSanityDiscriminatorImpl>(
      config, telemetry);
  }
}  // namespace cyclops::estimation
