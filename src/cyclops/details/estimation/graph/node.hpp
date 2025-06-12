#pragma once

#include "cyclops/details/type.hpp"

#include <ostream>
#include <variant>

namespace cyclops::estimation {
  struct Node {
    struct Frame {
      FrameID id;

      int dimension() const;
      int localDimension() const;
    };

    struct Bias {
      FrameID id;

      int dimension() const;
      int localDimension() const;
    };

    struct Landmark {
      LandmarkID id;

      int dimension() const;
      int localDimension() const;
    };

    std::variant<Frame, Bias, Landmark> variant;

    bool operator<(Node const& other) const;
    bool operator==(Node const& other) const;

    int dimension() const;
    int localDimension() const;

    template <size_t i>
    using VariantTypeOfIndex =
      std::variant_alternative_t<i, decltype(Node::variant)>;
  };

  namespace node {
    static inline Node makeFrame(FrameID id) {
      return {Node::Frame {id}};
    }

    static inline Node makeBias(FrameID id) {
      return {Node::Bias {id}};
    }

    static inline Node makeLandmark(LandmarkID id) {
      return {Node::Landmark {id}};
    }

    template <typename type>
    static bool is(Node const& node) {
      return std::holds_alternative<type>(node.variant);
    }
  }  // namespace node

  bool operator<(Node::Frame const& a, Node::Frame const& b);
  bool operator<(Node::Bias const& a, Node::Bias const& b);
  bool operator<(Node::Landmark const& a, Node::Landmark const& b);
  bool operator==(Node::Frame const& a, Node::Frame const& b);
  bool operator==(Node::Bias const& a, Node::Bias const& b);
  bool operator==(Node::Landmark const& a, Node::Landmark const& b);

  std::ostream& operator<<(std::ostream&, Node const&);
}  // namespace cyclops::estimation
