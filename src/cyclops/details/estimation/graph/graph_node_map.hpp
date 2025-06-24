#pragma once

#include "cyclops/details/estimation/type.hpp"
#include "cyclops/details/type.hpp"

#include <map>
#include <memory>
#include <optional>

namespace cyclops::estimation {
  struct StateVariableWriteAccessor;

  struct GraphNodeContext {
    using Ref = std::reference_wrapper<GraphNodeContext>;
    using CRef = std::reference_wrapper<GraphNodeContext const>;

    using MaybeRef = std::optional<Ref>;
    using MaybeCRef = std::optional<CRef>;

    Parameter parameter;
    NodeSet neighbors;
    FactorSet factors;
  };

  struct PriorNode {
    FactorID id;
    FactorPtr ptr;
    NodeSet input_nodes;
  };

  struct NeighborQueryResult {
    // Neighbor nodes.
    NodeSet nodes;

    // Factors between the queried nodes and the neighbor nodes.
    FactorSet factors;
  };

  class FactorGraphStateNodeMap {
  private:
    std::shared_ptr<StateVariableWriteAccessor> _state;

    FactorID _last_factor_id = 0;
    std::map<Node, GraphNodeContext> _node_contexts;
    std::optional<PriorNode> _prior;

  public:
    explicit FactorGraphStateNodeMap(
      std::shared_ptr<StateVariableWriteAccessor> state);
    ~FactorGraphStateNodeMap();

    bool createFrameNode(ceres::Problem& problem, FrameID frame_id);
    bool createLandmarkNode(ceres::Problem& problem, LandmarkID landmark_id);

    FactorID createPriorFactor(
      ceres::Problem& problem, FactorPtr ptr, NodeSet const& nodes);
    FactorID createFactor(
      FactorEntry factor_entry,
      std::vector<std::pair<Node, GraphNodeContext::Ref>> const& nodes);

    GraphNodeContext::MaybeRef findContext(Node const& node);
    GraphNodeContext::MaybeCRef findContext(Node const& node) const;

    std::optional<NodeSetCRef> queryNeighbors(Node const& node) const;
    NeighborQueryResult queryNeighbors(NodeSet const& nodes) const;

    std::optional<PriorNode> const& getPrior() const;

    std::vector<Node> allNodes() const;
    FactorSet allFactors() const;
  };
}  // namespace cyclops::estimation
