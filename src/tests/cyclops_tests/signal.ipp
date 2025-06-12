#pragma once

#include "cyclops_tests/signal.hpp"

namespace cyclops {
  template <typename vector_t>
  static auto numericDerivative(
    std::function<vector_t(Timestamp)> f, double const h = 1e-6) {
    return [f, h](Timestamp const t) -> vector_t {
      return (f(t + h) - f(t - h)) / (2 * h);
    };
  }

  template <>
  auto numericDerivative<Eigen::Quaterniond>(
    std::function<Eigen::Quaterniond(Timestamp)> f, double const h) {
    return [f, h](Timestamp const t) -> Eigen::Vector3d {
      auto const q1 = f(t - h);
      auto const q2 = f(t + h);
      return (q1.inverse() * q2).vec() / h;
    };
  }

  template <typename vector_t>
  static auto numericSecondDerivative(
    std::function<vector_t(Timestamp)> f, double const h = 1e-6) {
    return [f, h](Timestamp const t) -> vector_t {
      return (f(t + h) - 2 * f(t) + f(t - h)) / h / h;
    };
  }

  static QuaternionSignal yawRotation(ScalarSignal phi) {
    return [phi](Timestamp t) -> Eigen::Quaterniond {
      return Eigen::Quaterniond(
        Eigen::AngleAxisd(phi(t), Eigen::Vector3d::UnitZ()));
    };
  }

  static QuaternionSignal operator>>=(
    QuaternionSignal q1, QuaternionSignal q2) {
    return
      [q1, q2](Timestamp t) -> Eigen::Quaterniond { return q1(t) * q2(t); };
  }

  static QuaternionSignal just(Eigen::Quaterniond const& q) {
    return [q](Timestamp _) -> Eigen::Quaterniond { return q; };
  }
}  // namespace cyclops
