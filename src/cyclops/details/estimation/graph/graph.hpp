#pragma once

#include "cyclops/details/estimation/type.hpp"

#include "cyclops/details/measurement/type.hpp"
#include "cyclops/details/utils/type.hpp"
#include "cyclops/details/type.hpp"

#include <ceres/ceres.h>

#include <memory>
#include <vector>
#include <tuple>

namespace cyclops {
  struct CyclopsConfig;
}

namespace cyclops::estimation {
  struct PriorNode;
  struct NeighborQueryResult;
  struct LandmarkAcceptance;
  struct GaussianPrior;

  struct FactorGraphCostUpdater;
  struct FactorGraphStateNodeMap;

  class FactorGraphInstance {
  private:
    struct Impl;
    std::unique_ptr<Impl> _impl;

  public:
    FactorGraphInstance(
      std::unique_ptr<FactorGraphCostUpdater> cost_helper,
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<FactorGraphStateNodeMap> node_map);
    ~FactorGraphInstance();

    void fixGauge(FrameID frame_id);
    bool addFrameStateBlock(FrameID frame_id);
    bool addLandmarkStateBlock(LandmarkID landmark_id);

    void addImuCost(measurement::ImuMotion const& imu_motion);
    void addBiasPriorCost(FrameID frame_id);

    LandmarkAcceptance addLandmarkCost(
      std::set<FrameID> const& solvable_motions, LandmarkID feature_id,
      measurement::FeatureTrack const& track);

    void setPriorCost(GaussianPrior const& priors);

    std::optional<NodeSetCRef> queryNeighbors(Node const& node) const;
    NeighborQueryResult queryNeighbors(NodeSet const& nodes) const;

    std::optional<PriorNode> const& prior() const;

    std::string report() const;
    ceres::Solver::Summary solve();
    std::tuple<EigenCRSMatrix, Eigen::VectorXd> evaluate(
      std::vector<Node> const& nodes, std::vector<FactorPtr> const& factors);
  };
}  // namespace cyclops::estimation
