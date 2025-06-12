#pragma once

#include <random>
#include <vector>
#include <memory>

namespace cyclops {
  struct CyclopsConfig;
}

namespace cyclops::initializer {
  struct TwoViewGeometry;
  struct TwoViewCorrespondenceData;

  class TwoViewVisionGeometrySolver {
  public:
    virtual ~TwoViewVisionGeometrySolver() = default;
    virtual void reset() = 0;

    // returns a sequence of possible solutions.
    virtual std::vector<TwoViewGeometry> solve(
      TwoViewCorrespondenceData const& two_view_correspondence) = 0;

    static std::unique_ptr<TwoViewVisionGeometrySolver> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<std::mt19937> rgen);
  };
}  // namespace cyclops::initializer
