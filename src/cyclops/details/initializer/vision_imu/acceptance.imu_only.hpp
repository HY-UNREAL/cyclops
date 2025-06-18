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
  struct ImuRotationMatch;
  struct ImuTranslationMatchSolution;
  struct ImuTranslationMatchUncertainty;

  using ImuTranslationMatchCandidate = std::tuple<
    ImuTranslationMatchSolution, std::optional<ImuTranslationMatchUncertainty>>;

  class ImuOnlyTranslationMatchAcceptDiscriminator {
  public:
    virtual ~ImuOnlyTranslationMatchAcceptDiscriminator() = default;
    virtual void reset() = 0;

    virtual bool determineAccept(
      ImuRotationMatch const& rotation_match,
      ImuTranslationMatchCandidate const& translation_candidate) const = 0;

    static std::unique_ptr<ImuOnlyTranslationMatchAcceptDiscriminator> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
