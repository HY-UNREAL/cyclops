#pragma once

#include "cyclops/details/type.hpp"

#include <random>
#include <vector>
#include <memory>

namespace cyclops {
  struct CyclopsConfig;
}

namespace cyclops::initializer {
  struct TwoViewFeaturePair {
    Eigen::Vector2d feature_1;
    Eigen::Vector2d feature_2;
  };

  struct TwoViewImuRotationData {
    Eigen::Quaterniond value;
    Eigen::Matrix3d covariance;
  };

  struct TwoViewCorrespondenceData {
    TwoViewImuRotationData rotation_prior;
    std::map<LandmarkID, TwoViewFeaturePair> features;
  };

  struct TwoViewGeometry {
    SE3Transform camera_motion;
    LandmarkPositions landmarks;
  };

  class TwoViewVisionGeometrySolver {
  public:
    virtual ~TwoViewVisionGeometrySolver() = default;
    virtual void reset() = 0;

    // Returns a sequence of possible solutions.
    virtual std::vector<TwoViewGeometry> solve(
      TwoViewCorrespondenceData const& two_view_correspondence) = 0;

    static std::unique_ptr<TwoViewVisionGeometrySolver> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<std::mt19937> rgen);
  };
}  // namespace cyclops::initializer
