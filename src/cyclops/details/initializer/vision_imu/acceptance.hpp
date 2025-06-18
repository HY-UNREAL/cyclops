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
  struct ImuRotationMatch;
  struct ImuTranslationMatch;
  struct ImuTranslationMatchSolution;
  struct ImuTranslationMatchUncertainty;

  using ImuTranslationMatchCandidate = std::tuple<
    ImuTranslationMatchSolution, std::optional<ImuTranslationMatchUncertainty>>;

  class ImuTranslationMatchAcceptDiscriminator {
  public:
    virtual ~ImuTranslationMatchAcceptDiscriminator() = default;
    virtual void reset() = 0;

    virtual std::optional<ImuTranslationMatch> determineAcceptance(
      ImuRotationMatch const& rotation_match,
      std::vector<ImuTranslationMatchCandidate> const& candidates) const = 0;

    static std::unique_ptr<ImuTranslationMatchAcceptDiscriminator> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
