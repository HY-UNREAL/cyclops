#pragma once

#include <memory>

namespace cyclops {
  struct CyclopsConfig;
}  // namespace cyclops

namespace cyclops::initializer {
  struct ImuTranslationMatchSolution;
  struct ImuTranslationMatchUncertainty;

  class ImuTranslationMatchAcceptDiscriminator {
  public:
    virtual ~ImuTranslationMatchAcceptDiscriminator() = default;
    virtual void reset() = 0;

    enum AcceptDecision {
      ACCEPT,
      REJECT_COST_PROBABILITY_INSIGNIFICANT,
      REJECT_UNDERINFORMATIVE_PARAMETER,
      REJECT_SCALE_LESS_THAN_ZERO,
    };

    virtual AcceptDecision determineCandidate(
      ImuTranslationMatchSolution const& solution,
      ImuTranslationMatchUncertainty const& uncertainty) const = 0;
    virtual AcceptDecision determineAccept(
      ImuTranslationMatchSolution const& solution,
      ImuTranslationMatchUncertainty const& uncertainty) const = 0;

    static std::unique_ptr<ImuTranslationMatchAcceptDiscriminator> Create(
      std::shared_ptr<CyclopsConfig const> config);
  };
}  // namespace cyclops::initializer
