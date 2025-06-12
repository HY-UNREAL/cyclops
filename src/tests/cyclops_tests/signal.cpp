#include "cyclops_tests/signal.hpp"

namespace cyclops {
  SE3Transform PoseSignal::evaluate(Timestamp t) const {
    return {position(t), orientation(t)};
  }
}  // namespace cyclops
