#pragma once

#include <memory>
#include <optional>

namespace cyclops {
  struct CyclopsConfig;
}  // namespace cyclops

namespace cyclops::telemetry {
  class InitializerTelemetry;
}

namespace cyclops::initializer {
  struct ImuMatchSolution;
  struct ImuMatchUncertainty;

  using ImuMatchCandidate =
    std::tuple<ImuMatchSolution, std::optional<ImuMatchUncertainty>>;

  class ImuOnlyMatchAcceptDiscriminator {
  public:
    virtual ~ImuOnlyMatchAcceptDiscriminator() = default;
    virtual void reset() = 0;

    virtual bool determineAccept(
      ImuMatchCandidate const& match_candidate) const = 0;

    static std::unique_ptr<ImuOnlyMatchAcceptDiscriminator> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
