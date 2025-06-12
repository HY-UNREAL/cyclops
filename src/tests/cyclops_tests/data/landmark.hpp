#pragma once

#include "cyclops_tests/signal.hpp"

#include "cyclops/details/measurement/type.hpp"
#include "cyclops/details/type.hpp"

#include <functional>
#include <map>
#include <random>
#include <set>
#include <vector>

namespace cyclops {
  struct LandmarkGenerationArgument {
    int count;
    Eigen::Vector3d center;
    Eigen::Matrix3d concentration;
  };

  LandmarkPositions generateLandmarks(
    std::mt19937& rgen, LandmarkGenerationArgument const& arg);
  LandmarkPositions generateLandmarks(
    std::mt19937& rgen, std::vector<LandmarkGenerationArgument> const& args);
  LandmarkPositions generateLandmarks(
    std::set<LandmarkID> ids, std::function<Eigen::Vector3d(LandmarkID)> gen);

  std::map<LandmarkID, FeaturePoint> generateLandmarkObservations(
    Eigen::Matrix3d const& R, Eigen::Vector3d const& p,
    LandmarkPositions const& landmarks);
  std::map<LandmarkID, FeaturePoint> generateLandmarkObservations(
    std::mt19937& rgen, Eigen::Matrix2d const& cov, Eigen::Matrix3d const& R,
    Eigen::Vector3d const& p, LandmarkPositions const& landmarks);

  std::vector<ImageData> makeLandmarkFrames(
    PoseSignal pose_signal, SE3Transform const& extrinsic,
    LandmarkPositions const&, std::vector<Timestamp> const&);
  std::vector<ImageData> makeLandmarkFrames(
    PoseSignal pose_signal, SE3Transform const& extrinsic,
    LandmarkPositions const&, std::vector<Timestamp> const&, std::mt19937& rgen,
    Eigen::Matrix2d const& cov);

  measurement::FeatureTracks makeLandmarkTracks(
    PoseSignal pose_signal, SE3Transform const& extrinsic,
    LandmarkPositions const&, std::map<FrameID, Timestamp> const&);
  measurement::FeatureTracks makeLandmarkTracks(
    PoseSignal pose_signal, SE3Transform const& extrinsic,
    LandmarkPositions const&, std::map<FrameID, Timestamp> const&,
    std::mt19937& rgen, Eigen::Matrix2d const& cov);

  std::map<FrameID, std::map<LandmarkID, FeaturePoint>>
  makeLandmarkMultiviewObservation(
    PoseSignal pose_signal, SE3Transform const& extrinsic,
    LandmarkPositions const&, std::map<FrameID, Timestamp> const&);
}  // namespace cyclops
