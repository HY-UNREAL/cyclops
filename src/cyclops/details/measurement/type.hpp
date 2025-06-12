#pragma once

#include "cyclops/details/type.hpp"

#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace cyclops::estimation {
  struct Node;
}  // namespace cyclops::estimation

namespace cyclops::measurement {
  struct ImuPreintegration;

  struct ImuMotion {
    FrameID from;
    FrameID to;
    std::unique_ptr<ImuPreintegration> data;
  };
  using ImuMotions = std::vector<ImuMotion>;

  using ImuMotionRef = std::reference_wrapper<ImuMotion const>;
  using ImuMotionRefs = std::vector<ImuMotionRef>;

  using FeatureTrack = std::map<FrameID, FeaturePoint>;
  using FeatureTracks = std::map<LandmarkID, FeatureTrack>;
}  // namespace cyclops::measurement
