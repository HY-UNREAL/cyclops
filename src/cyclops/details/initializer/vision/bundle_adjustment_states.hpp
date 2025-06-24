#pragma once

#include "cyclops/details/utils/block_meta.hpp"
#include "cyclops/details/type.hpp"

#include <Eigen/Dense>
#include <functional>

namespace cyclops {
  struct SE3Transform;
}  // namespace cyclops

namespace cyclops::initializer {
  struct MultiViewGeometry;

  class BundleAdjustmentGyroBiasStateBlock {
  private:
    block_meta::block_cascade<block_meta::bias_gyr> _data_block;

  public:
    explicit BundleAdjustmentGyroBiasStateBlock();

    Eigen::Map<Eigen::Vector3d> value();
    Eigen::Map<Eigen::Vector3d const> value() const;

    double* data();
    double const* data() const;
  };

  class BundleAdjustmentCameraMotionStateBlock {
  private:
    block_meta::block_cascade<block_meta::orientation, block_meta::position>
      _data_block;

  public:
    explicit BundleAdjustmentCameraMotionStateBlock(SE3Transform const& guess);

    Eigen::Map<Eigen::Quaterniond> orientation();
    Eigen::Map<Eigen::Quaterniond const> orientation() const;

    Eigen::Map<Eigen::Vector3d> position();
    Eigen::Map<Eigen::Vector3d const> position() const;

    double* data();
    double const* data() const;

    static Eigen::Map<Eigen::Quaterniond> orientation(double* data);
    static Eigen::Map<Eigen::Quaterniond const> orientation(double const* data);
    static Eigen::Map<Eigen::Vector3d> position(double* data);
    static Eigen::Map<Eigen::Vector3d const> position(double const* data);

    SE3Transform asSE3Transform() const;
  };

  class BundleAdjustmentLandmarkPositionStateBlock {
  private:
    std::array<double, 3> _data_block;

  public:
    explicit BundleAdjustmentLandmarkPositionStateBlock(
      Eigen::Vector3d const& guess);

    Eigen::Map<Eigen::Vector3d> position();
    Eigen::Map<Eigen::Vector3d const> position() const;
    double* data();
    double const* data() const;

    static Eigen::Map<Eigen::Vector3d> position(double* data);
    static Eigen::Map<Eigen::Vector3d const> position(double const* data);

    Eigen::Vector3d asVector3() const;
  };

  struct BundleAdjustmentOptimizationState {
    using BiasBlock = BundleAdjustmentGyroBiasStateBlock;
    using MotionBlock = BundleAdjustmentCameraMotionStateBlock;
    using LandmarkBlock = BundleAdjustmentLandmarkPositionStateBlock;

    explicit BundleAdjustmentOptimizationState(
      MultiViewGeometry const& initial_guess);

    BiasBlock gyro_bias;
    std::map<FrameID, MotionBlock> camera_motions;
    std::map<LandmarkID, LandmarkBlock> landmark_positions;

    using MotionBlockRef = std::reference_wrapper<MotionBlock>;
    using MotionBlockRefPair = std::tuple<MotionBlockRef, MotionBlockRef>;

    std::optional<MotionBlockRefPair> normalize();

    std::map<LandmarkID, Eigen::Vector3d> landmarkPositions() const;
    std::map<FrameID, SE3Transform> cameraMotions() const;
  };
}  // namespace cyclops::initializer
