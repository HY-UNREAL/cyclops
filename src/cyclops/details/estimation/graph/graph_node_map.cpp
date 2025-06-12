#include "cyclops/details/estimation/graph/graph_node_map.hpp"
#include "cyclops/details/estimation/graph/factor.hpp"
#include "cyclops/details/estimation/graph/node.hpp"

#include "cyclops/details/estimation/ceres/manifold.se3.hpp"
#include "cyclops/details/estimation/state/accessor_write.hpp"
#include "cyclops/details/logging.hpp"

#include <ceres/ceres.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <range/v3/all.hpp>

namespace cyclops::estimation {
  namespace views = ranges::views;

  FactorGraphStateNodeMap::FactorGraphStateNodeMap(
    std::shared_ptr<StateVariableWriteAccessor> state)
      : _state(state) {
  }

  FactorGraphStateNodeMap::~FactorGraphStateNodeMap() = default;

  bool FactorGraphStateNodeMap::createFrameNode(
    ceres::Problem& problem, FrameID frame_id) {
    auto maybe_x = _state->motionFrame(frame_id);
    if (!maybe_x) {
      __logger__->error(
        "Tried to add uninitialized motion state for frame {}.", frame_id);
      return false;
    }

    auto x = maybe_x->get().data();
    _node_contexts.emplace(
      node::makeFrame(frame_id), GraphNodeContext {x, {}, {}});

    problem.AddParameterBlock(
      x, 10,
      new ceres::AutoDiffLocalParameterization<
        ExponentialSE3Plus<false>, 10, 9>);

    auto b = x + 10;
    _node_contexts.emplace(
      node::makeBias(frame_id), GraphNodeContext {b, {}, {}});

    problem.AddParameterBlock(b, 6);

    return true;
  }

  bool FactorGraphStateNodeMap::createLandmarkNode(
    ceres::Problem& problem, LandmarkID landmark_id) {
    auto maybe_f = _state->landmark(landmark_id);
    if (!maybe_f)
      return false;

    auto f = maybe_f->get().data();
    _node_contexts.emplace(
      node::makeLandmark(landmark_id), GraphNodeContext {f, {}, {}});
    problem.AddParameterBlock(f, 3);

    return true;
  }

  GraphNodeContext::MaybeRef FactorGraphStateNodeMap::findContext(
    Node const& node) {
    auto maybe_context = _node_contexts.find(node);
    if (maybe_context == _node_contexts.end())
      return std::nullopt;

    auto& [_, context] = *maybe_context;
    return context;
  }

  GraphNodeContext::MaybeCRef FactorGraphStateNodeMap::findContext(
    Node const& node) const {
    auto maybe_context = _node_contexts.find(node);
    if (maybe_context == _node_contexts.end())
      return std::nullopt;

    auto const& [_, context] = *maybe_context;
    return context;
  }

  FactorID FactorGraphStateNodeMap::createPriorFactor(
    ceres::Problem& problem, FactorPtr ptr, NodeSet const& nodes) {
    if (_prior) {
      __logger__->warn("Warning: setting prior twice");
      problem.RemoveResidualBlock(_prior->ptr);
    }
    _last_factor_id++;
    _prior = {_last_factor_id, ptr, nodes};
    return _last_factor_id;
  }

  FactorID FactorGraphStateNodeMap::createFactor(
    FactorEntry factor_entry,
    std::vector<std::pair<Node, GraphNodeContext::Ref>> const& nodes) {
    auto pairs =
      views::cartesian_product(
        views::iota(0, (int)nodes.size()), views::iota(0, (int)nodes.size())) |
      views::filter([](auto ab) {
        auto [a, b] = ab;
        return a != b;
      }) |
      ranges::to_vector;
    for (auto const& [i, j] : pairs) {
      auto const& [n1, ctxt1] = nodes.at(i);
      auto const& [n2, ctxt2] = nodes.at(j);
      ctxt1.get().neighbors.insert(n2);
      ctxt2.get().neighbors.insert(n1);
    }

    _last_factor_id++;
    for (auto const& ctxt : nodes | views::values)
      ctxt.get().factors.emplace(_last_factor_id, factor_entry);
    return _last_factor_id;
  }

  std::optional<NodeSetCRef> FactorGraphStateNodeMap::queryNeighbors(
    Node const& node) const {
    auto maybe_context = findContext(node);
    if (!maybe_context)
      return std::nullopt;
    return maybe_context->get().neighbors;
  }

  NeighborQueryResult FactorGraphStateNodeMap::queryNeighbors(
    NodeSet const& nodes) const {
    NeighborQueryResult result;
    for (auto const& node : nodes) {
      auto maybe_context = findContext(node);
      if (!maybe_context) {
        __logger__->error("Queried non-existing node: {}", node);
        continue;
      }
      auto const& context = maybe_context->get();
      result.nodes.insert(context.neighbors.begin(), context.neighbors.end());
      result.factors.insert(context.factors.begin(), context.factors.end());
    }
    return result;
  }

  std::optional<PriorNode> const& FactorGraphStateNodeMap::getPrior() const {
    return _prior;
  }

  std::vector<Node> FactorGraphStateNodeMap::allNodes() const {
    return _node_contexts | views::keys | ranges::to_vector;
  }

  FactorSet FactorGraphStateNodeMap::allFactors() const {
    FactorSet result;
    for (auto const& ctxt : _node_contexts | views::values)
      result.insert(ctxt.factors.begin(), ctxt.factors.end());
    return result;
  }
}  // namespace cyclops::estimation
