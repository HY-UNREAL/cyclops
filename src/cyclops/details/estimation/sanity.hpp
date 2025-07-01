#pragma once

#include <memory>

namespace cyclops {
  struct CyclopsConfig;
}

namespace cyclops::telemetry {
  struct OptimizerTelemetry;
}

namespace cyclops::estimation {
  struct LandmarkSanityStatistics {
    size_t landmark_observations;
    size_t landmark_accepts;
    size_t uninitialized_landmarks;
    size_t depth_threshold_failures;
    size_t mnorm_threshold_failures;
    size_t information_strength_failures;
  };

  struct OptimizerSanityStatistics {
    double final_cost;
    int num_residuals;
    int num_parameters;
  };

  class EstimationSanityDiscriminator {
  public:
    virtual ~EstimationSanityDiscriminator() = default;

    virtual void reset() = 0;
    virtual void update(
      LandmarkSanityStatistics const& landmark_sanity,
      OptimizerSanityStatistics const& optimizer_sanity) = 0;

    virtual bool sanity() const = 0;

    static std::unique_ptr<EstimationSanityDiscriminator> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<telemetry::OptimizerTelemetry> telemetry = nullptr);
  };
}  // namespace cyclops::estimation
