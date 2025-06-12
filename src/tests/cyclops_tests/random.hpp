#pragma once

#include "cyclops/details/estimation/state/state_block.hpp"

#include <Eigen/Dense>
#include <random>

namespace cyclops {
  struct ImuMotionState;

  double perturbate(double const& x, double const& s, std::mt19937& rgen);

  Eigen::Vector2d perturbate(
    Eigen::Vector2d const& x, double const& S, std::mt19937& rgen);
  Eigen::Vector2d perturbate(
    Eigen::Vector2d const& x, Eigen::Matrix2d const& S, std::mt19937& rgen);

  Eigen::Vector3d perturbate(
    Eigen::Vector3d const& x, double const& S, std::mt19937& rgen);
  Eigen::Vector3d perturbate(
    Eigen::Vector3d const& x, Eigen::Matrix3d const& S, std::mt19937& rgen);
  Eigen::Quaterniond perturbate(
    Eigen::Quaterniond const& q, double const& S, std::mt19937& rgen);
  Eigen::Quaterniond perturbate(
    Eigen::Quaterniond const& q, Eigen::Matrix3d const& S, std::mt19937& rgen);

  ImuMotionState perturbate(
    ImuMotionState const& x, double S, std::mt19937& rgen);

  estimation::MotionFrameParameterBlock makePerturbatedFrameState(
    ImuMotionState const& x, double perturbation, std::mt19937& rgen);
  estimation::LandmarkParameterBlock makePerturbatedLandmarkState(
    Eigen::Vector3d const& landmark, double perturbation, std::mt19937& rgen);
}  // namespace cyclops
