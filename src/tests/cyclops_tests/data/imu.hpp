#pragma once

#include "cyclops_tests/data/typefwd.hpp"
#include "cyclops_tests/signal.hpp"
#include "cyclops/details/measurement/type.hpp"

#include <map>
#include <memory>
#include <random>
#include <vector>

namespace cyclops::measurement {
  struct ImuPreintegration;
}

namespace cyclops {
  struct ImuData;
  struct SensorStatistics;

  struct ImuMockup {
    Eigen::Vector3d bias_acc;
    Eigen::Vector3d bias_gyr;
    ImuData measurement;
  };

  ImuMockupSequence generateImuData(
    PoseSignal pose_signal, std::vector<Timestamp> const& timestamps);
  ImuMockupSequence generateImuData(
    PoseSignal pose_signal, std::vector<Timestamp> const& timestamps,
    Eigen::Vector3d const& bias_acc, Eigen::Vector3d const& bias_gyr);
  ImuMockupSequence generateImuData(
    PoseSignal pose_signal, std::vector<Timestamp> const& timestamps,
    std::mt19937& rgen, SensorStatistics const& noise);
  ImuMockupSequence generateImuData(
    PoseSignal pose_signal, std::vector<Timestamp> const& timestamps,
    Eigen::Vector3d const& bias_acc, Eigen::Vector3d const& bias_gyr,
    std::mt19937& rgen, SensorStatistics const& noise);

  std::unique_ptr<measurement::ImuPreintegration> makeImuPreintegration(
    PoseSignal pose_signal, Timestamp t_s, Timestamp t_e);
  std::unique_ptr<measurement::ImuPreintegration> makeImuPreintegration(
    SensorStatistics const& noise, PoseSignal pose_signal, Timestamp t_s,
    Timestamp t_e);
  std::unique_ptr<measurement::ImuPreintegration> makeImuPreintegration(
    Eigen::Vector3d const& bias_acc, Eigen::Vector3d const& bias_gyr,
    PoseSignal pose_signal, Timestamp t_s, Timestamp t_e);
  std::unique_ptr<measurement::ImuPreintegration> makeImuPreintegration(
    std::mt19937& rgen, SensorStatistics const& noise,
    PoseSignal pose_signal, Timestamp t_s, Timestamp t_e);
  std::unique_ptr<measurement::ImuPreintegration> makeImuPreintegration(
    std::mt19937& rgen, SensorStatistics const& noise,
    Eigen::Vector3d const& bias_acc, Eigen::Vector3d const& bias_gyr,
    PoseSignal pose_signal, Timestamp t_s, Timestamp t_e);

  measurement::ImuMotions makeImuMotions(
    PoseSignal pose_signal, std::map<FrameID, Timestamp> const& frames);
  measurement::ImuMotions makeImuMotions(
    SensorStatistics const& noise, PoseSignal pose_signal,
    std::map<FrameID, Timestamp> const& frames);
  measurement::ImuMotions makeImuMotions(
    Eigen::Vector3d const& bias_acc, Eigen::Vector3d const& bias_gyr,
    PoseSignal pose_signal, std::map<FrameID, Timestamp> const& frames);
  measurement::ImuMotions makeImuMotions(
    std::mt19937& rgen, SensorStatistics const& noise,
    PoseSignal pose_signal, std::map<FrameID, Timestamp> const& frames);
  measurement::ImuMotions makeImuMotions(
    std::mt19937& rgen, SensorStatistics const& noise,
    Eigen::Vector3d const& bias_acc, Eigen::Vector3d const& bias_gyr,
    PoseSignal pose_signal, std::map<FrameID, Timestamp> const& frames);
}  // namespace cyclops
