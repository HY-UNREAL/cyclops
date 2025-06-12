#pragma once

#include "cyclops/details/utils/block_meta.hpp"
#include "cyclops/details/type.hpp"

#include <map>

namespace cyclops::estimation {
  using MotionFrameParameterBlock = block_meta::block_cascade<
    block_meta::orientation, block_meta::position, block_meta::velocity,
    block_meta::bias_acc, block_meta::bias_gyr>;
  using LandmarkParameterBlock =
    block_meta::block_cascade<block_meta::landmark_position>;

  using MotionFrameParameterBlocks =
    std::map<FrameID, MotionFrameParameterBlock>;
  using LandmarkParameterBlocks = std::map<LandmarkID, LandmarkParameterBlock>;

  namespace buffer::motion_frame {
    SE3Transform getSE3Transform(double const* frame_ptr);
  }  // namespace buffer::motion_frame

  SE3Transform getSE3Transform(MotionFrameParameterBlock const& frame);

  ImuMotionState getMotionState(MotionFrameParameterBlock const& frame);
  Eigen::Vector3d getAccBias(MotionFrameParameterBlock const& frame);
  Eigen::Vector3d getGyrBias(MotionFrameParameterBlock const& frame);

  Eigen::Vector3d getPosition(LandmarkParameterBlock const& block);
}  // namespace cyclops::estimation
