#pragma once

#include "cyclops/details/type.hpp"

#include <set>
#include <map>
#include <memory>
#include <vector>

namespace cyclops {
  struct CyclopsConfig;
  struct RotationPositionPair;
}  // namespace cyclops

namespace cyclops::initializer {
  struct TwoViewFeaturePair;
  struct TwoViewGeometry;
  struct TwoViewImuRotationData;

  class TwoViewMotionHypothesisSelector {
  public:
    using MotionHypotheses = std::vector<RotationPositionPair>;

    using TwoViewFeatureSet = std::map<LandmarkID, TwoViewFeaturePair>;
    using InlierSet = std::set<LandmarkID>;

    virtual ~TwoViewMotionHypothesisSelector() = default;
    virtual void reset() = 0;

    virtual std::vector<TwoViewGeometry> selectPossibleMotions(
      MotionHypotheses const& motions, TwoViewFeatureSet const& image_data,
      InlierSet const& inliers, TwoViewImuRotationData const& gyro_prior) = 0;

    static std::unique_ptr<TwoViewMotionHypothesisSelector> Create(
      std::shared_ptr<CyclopsConfig const> config);
  };
}  // namespace cyclops::initializer
