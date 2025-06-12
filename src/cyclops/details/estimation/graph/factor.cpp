#include "cyclops/details/estimation/graph/factor.hpp"
#include "cyclops/details/utils/type.hpp"

namespace cyclops::estimation {
  bool operator<(Factor::Imu const& a, Factor::Imu const& b) {
    if (a.from < b.from)
      return true;
    if (a.from > b.from)
      return false;
    return a.to < b.to;
  }

  bool operator<(Factor::BiasWalk const& a, Factor::BiasWalk const& b) {
    if (a.from < b.from)
      return true;
    if (a.from > b.from)
      return false;
    return a.to < b.to;
  }

  bool operator<(Factor::BiasPrior const& a, Factor::BiasPrior const& b) {
    return a.frame < b.frame;
  }

  bool operator<(Factor::Feature const& a, Factor::Feature const& b) {
    if (a.frame < b.frame)
      return true;
    if (a.frame > b.frame)
      return false;
    return a.landmark < b.landmark;
  }

  bool operator<(Factor::Prior const& a, Factor::Prior const& b) {
    return false;
  }

  bool operator==(Factor::Imu const& a, Factor::Imu const& b) {
    return a.from == b.from && a.to == b.to;
  }

  bool operator==(Factor::BiasWalk const& a, Factor::BiasWalk const& b) {
    return a.from == b.from && a.to == b.to;
  }

  bool operator==(Factor::BiasPrior const& a, Factor::BiasPrior const& b) {
    return a.frame == b.frame;
  }

  bool operator==(Factor::Feature const& a, Factor::Feature const& b) {
    return a.frame == b.frame && a.landmark == b.landmark;
  }

  bool operator==(Factor::Prior const& a, Factor::Prior const& b) {
    return true;
  }

  bool Factor::operator<(Factor const& other) const {
    return this->variant < other.variant;
  }

  bool Factor::operator==(Factor const& other) const {
    return this->variant == other.variant;
  }

  std::ostream& operator<<(std::ostream& ostr, Factor const& factor) {
    auto visitor = overloaded {
      [&ostr](Factor::Imu const& _) {
        ostr << "IMU [" << _.from << ", " << _.to << "]";
      },
      [&ostr](Factor::BiasWalk const& _) {
        ostr << "IMU walk [" << _.from << ", " << _.to << "]";
      },
      [&ostr](Factor::BiasPrior const& _) {
        ostr << "IMU bias prior [" << _.frame << "]";
      },
      [&ostr](Factor::Feature const& _) {
        ostr << "Feature [" << _.frame << ", " << _.landmark << "]";
      },
      [&ostr](Factor::Prior const& _) { ostr << "Prior"; },
    };
    std::visit(visitor, factor.variant);
    return ostr;
  }
}  // namespace cyclops::estimation
