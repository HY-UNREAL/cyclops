#pragma once

#include <memory>

namespace cyclops::telemetry {
  class OptimizerTelemetry {
  public:
    virtual ~OptimizerTelemetry() = default;
    virtual void reset();

    struct SanityStatistics {
      double final_cost;
      double final_cost_significant_probability;

      size_t landmark_observations;
      double landmark_accept_rate;
      double landmark_uninitialized_rate;
      double landmark_depth_threshold_failure_rate;
      double landmark_chi_square_test_failure_rate;
    };
    virtual void onSanityStatistics(SanityStatistics const& statistics);

    struct BadReason {
      bool bad_landmark_update;
      bool bad_final_cost;
    };
    virtual void onSanityBad(
      BadReason reason, SanityStatistics const& statistics);

    struct FailureReason {
      bool continued_bad_landmark_update;
      bool continued_bad_final_cost;
    };
    virtual void onSanityFailure(
      FailureReason reason, SanityStatistics const& statistics);

    virtual void onUserResetRequest();

    static std::unique_ptr<OptimizerTelemetry> CreateDefault();
  };
}  // namespace cyclops::telemetry
