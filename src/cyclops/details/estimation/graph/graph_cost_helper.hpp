#pragma once

#include "cyclops/details/estimation/graph/node.hpp"
#include "cyclops/details/measurement/type.hpp"

#include <set>
#include <variant>

namespace ceres {
  struct Problem;
}

namespace cyclops::measurement {
  struct ImuPreintegration;
}  // namespace cyclops::measurement

namespace cyclops {
  struct CyclopsConfig;
}  // namespace cyclops

namespace cyclops::estimation {
  class FactorGraphStateNodeMap;

  struct LandmarkAcceptance {
    struct Accepted {
      int observation_count;
      size_t accepted_count;
    };

    struct Uninitialized {};

    struct NoInlier {
      int observation_count;
      int depth_threshold_failure_count;
      int mahalanobis_norm_test_failure_count;
    };

    struct DeficientInformation {
      int observation_count;
      double information_index;
    };

    std::variant<Accepted, Uninitialized, NoInlier, DeficientInformation>
      variant;
  };

  class FactorGraphCostUpdater {
  private:
    struct Impl;
    std::unique_ptr<Impl> _pimpl;

  public:
    FactorGraphCostUpdater(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<FactorGraphStateNodeMap> node_map);
    ~FactorGraphCostUpdater();

    bool addImuCost(
      ceres::Problem& problem, measurement::ImuMotion const& imu_motion);
    bool addBiasPriorCost(ceres::Problem& problem, FrameID frame_id);

    LandmarkAcceptance addLandmarkCostBatch(
      ceres::Problem& problem, std::set<FrameID> const& solvable_motions,
      LandmarkID feature_id, measurement::FeatureTrack const& track);
  };
}  // namespace cyclops::estimation
