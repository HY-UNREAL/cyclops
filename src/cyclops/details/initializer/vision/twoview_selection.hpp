#pragma once

#include "cyclops/details/type.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <utility>

namespace cyclops::initializer {
  struct TwoViewCorrespondenceData;
  struct MultiViewCorrespondences;

  std::optional<std::reference_wrapper<
    std::pair<FrameID const, TwoViewCorrespondenceData> const>>
  selectBestTwoViewPair(MultiViewCorrespondences const& multiviews);
}  // namespace cyclops::initializer
