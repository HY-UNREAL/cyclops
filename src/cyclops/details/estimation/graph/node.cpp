#include "cyclops/details/estimation/graph/node.hpp"
#include "cyclops/details/utils/type.hpp"

namespace cyclops::estimation {
  bool operator<(Node::Frame const& a, Node::Frame const& b) {
    return a.id < b.id;
  }

  bool operator<(Node::Bias const& a, Node::Bias const& b) {
    return a.id < b.id;
  }

  bool operator<(Node::Landmark const& a, Node::Landmark const& b) {
    return a.id < b.id;
  }

  bool Node::operator<(Node const& other) const {
    return this->variant < other.variant;
  }

  bool operator==(Node::Frame const& a, Node::Frame const& b) {
    return a.id == b.id;
  }

  bool operator==(Node::Bias const& a, Node::Bias const& b) {
    return a.id == b.id;
  }

  bool operator==(Node::Landmark const& a, Node::Landmark const& b) {
    return a.id == b.id;
  }

  bool Node::operator==(Node const& other) const {
    return this->variant == other.variant;
  }

  int Node::Frame::dimension() const {
    return 10;
  }

  int Node::Frame::localDimension() const {
    return 9;
  }

  int Node::Bias::dimension() const {
    return 6;
  }

  int Node::Bias::localDimension() const {
    return 6;
  }

  int Node::Landmark::dimension() const {
    return 3;
  }

  int Node::Landmark::localDimension() const {
    return 3;
  }

  int Node::dimension() const {
    return std::visit([](auto const& _) { return _.dimension(); }, variant);
  }

  int Node::localDimension() const {
    return std::visit(
      [](auto const& _) { return _.localDimension(); }, variant);
  }

  std::ostream& operator<<(std::ostream& o, Node const& node) {
    std::visit(
      overloaded {
        [&o](Node::Frame frame) { o << "frame(" << frame.id << ")"; },
        [&o](Node::Bias frame) { o << "bias(" << frame.id << ")"; },
        [&o](Node::Landmark node) { o << "landmark(" << node.id << ")"; },
      },
      node.variant);
    return o;
  }
}  // namespace cyclops::estimation
