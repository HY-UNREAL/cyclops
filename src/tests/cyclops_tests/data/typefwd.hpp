#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <vector>

namespace cyclops {
  struct ImuMockup;
  struct LandmarkGenerationArgument;

  using ImuMockupSequence = std::map<Timestamp, ImuMockup>;
  using LandmarkGenerationArguments = std::vector<LandmarkGenerationArgument>;
}  // namespace cyclops
