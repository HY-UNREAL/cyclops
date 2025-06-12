#pragma once

#include "cyclops/details/type.hpp"
#include <variant>

namespace cyclops::estimation {
  struct Factor {
    struct Imu {
      FrameID from;
      FrameID to;
    };

    struct BiasWalk {
      FrameID from;
      FrameID to;
    };

    struct BiasPrior {
      FrameID frame;
    };

    struct Feature {
      FrameID frame;
      LandmarkID landmark;
    };

    struct Prior {};

    std::variant<Imu, BiasWalk, BiasPrior, Feature, Prior> variant;

    bool operator<(Factor const& other) const;
    bool operator==(Factor const& other) const;
  };

  namespace factor {
    static inline Factor makeImu(FrameID from, FrameID to) {
      return {Factor::Imu {from, to}};
    }

    static inline Factor makeBiasWalk(FrameID from, FrameID to) {
      return {Factor::BiasWalk {from, to}};
    }

    static inline Factor makeBiasPrior(FrameID frame) {
      return {Factor::BiasPrior {frame}};
    }

    static inline Factor makeFeature(FrameID frame, LandmarkID landmark) {
      return {Factor::Feature {frame, landmark}};
    }

    static inline Factor makePrior() {
      return {Factor::Prior {}};
    }

    template <typename type>
    static bool is(Factor const& factor) {
      return std::holds_alternative<type>(factor.variant);
    }
  }  // namespace factor

  bool operator<(Factor::Imu const& a, Factor::Imu const& b);
  bool operator<(Factor::BiasWalk const& a, Factor::BiasWalk const& b);
  bool operator<(Factor::BiasPrior const& a, Factor::BiasPrior const& b);
  bool operator<(Factor::Feature const& a, Factor::Feature const& b);
  bool operator<(Factor::Prior const& a, Factor::Prior const& b);

  bool operator==(Factor::Imu const& a, Factor::Imu const& b);
  bool operator==(Factor::BiasWalk const& a, Factor::BiasWalk const& b);
  bool operator==(Factor::BiasPrior const& a, Factor::BiasPrior const& b);
  bool operator==(Factor::Feature const& a, Factor::Feature const& b);
  bool operator==(Factor::Prior const& a, Factor::Prior const& b);

  std::ostream& operator<<(std::ostream&, Factor const&);
}  // namespace cyclops::estimation
