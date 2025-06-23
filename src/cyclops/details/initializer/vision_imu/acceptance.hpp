#pragma once

#include <memory>
#include <optional>
#include <vector>

namespace cyclops {
  struct CyclopsConfig;
}  // namespace cyclops

namespace cyclops::telemetry {
  class InitializerTelemetry;
}

namespace cyclops::initializer {
  struct ImuMatchResult;
  struct ImuMatchSolution;
  struct ImuMatchUncertainty;

  using ImuMatchCandidate =
    std::tuple<ImuMatchSolution, std::optional<ImuMatchUncertainty>>;

  class ImuMatchAcceptDiscriminator {
  public:
    virtual ~ImuMatchAcceptDiscriminator() = default;
    virtual void reset() = 0;

    virtual std::vector<ImuMatchResult> determineAcceptance(
      std::vector<ImuMatchCandidate> const& match_candidates) const = 0;

    static std::unique_ptr<ImuMatchAcceptDiscriminator> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
