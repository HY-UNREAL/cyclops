#pragma once

#include "cyclops/details/initializer/vision/type.hpp"

#include "cyclops/details/utils/math.hpp"
#include "cyclops/details/type.hpp"

#include <ceres/ceres.h>

namespace cyclops::initializer {
  struct LandmarkProjectionCost: public ceres::SizedCostFunction<2, 7, 3> {
    Eigen::Vector2d const u;
    Eigen::Matrix2d const weight_sqrt;

    LandmarkProjectionCost(FeaturePoint const& feature);
    bool Evaluate(
      double const* const* parameters, double* residuals,
      double** jacobians) const override;
  };

  struct BundleAdjustmentScaleConstraintVirtualCost {
    double const weight;
    explicit BundleAdjustmentScaleConstraintVirtualCost(double weight)
        : weight(weight) {
    }

    template <typename scalar_t>
    bool operator()(
      scalar_t const* const x0, scalar_t const* const xn,
      scalar_t* const r) const {
      using Vector3 = Eigen::Matrix<scalar_t, 3, 1>;

      Vector3 const p0(x0 + 4);
      Vector3 const pn(xn + 4);

      *r = ((pn - p0).norm() - scalar_t(1.0)) * scalar_t(weight);
      return true;
    }
  };

  struct BundleAdjustmentGyroBiasZeroPriorCost {
    double const _weight;

    explicit BundleAdjustmentGyroBiasZeroPriorCost(double weight)
        : _weight(weight) {
    }

    template <typename scalar_t>
    bool operator()(scalar_t const* const b_w, scalar_t* const r) const {
      using Vector3 = Eigen::Matrix<scalar_t, 3, 1>;
      (Eigen::Map<Vector3>(r)) = static_cast<scalar_t>(_weight) * Vector3(b_w);
      return true;
    }
  };

  struct BundleAdjustmentCameraRotationPriorCost {
    TwoViewImuRotationData const& _prior;
    Eigen::Matrix3d const _weight;

    explicit BundleAdjustmentCameraRotationPriorCost(
      TwoViewImuRotationData const& prior);

    template <typename scalar_t>
    bool operator()(
      scalar_t const* const x_prev, scalar_t const* const x_next,
      scalar_t const* const b_w, scalar_t* const residual) const {
      using Vector3 = Eigen::Matrix<scalar_t, 3, 1>;
      using Quaternion = Eigen::Quaternion<scalar_t>;

      auto r = Eigen::Map<Vector3>(residual);

      auto q_prev = Quaternion(x_prev);
      auto q_next = Quaternion(x_next);

      Quaternion y_q = q_prev.conjugate() * q_next;
      Quaternion y_q_hat = _prior.value.cast<scalar_t>();

      auto G_R = _prior.gyro_bias_jacobian.cast<scalar_t>().eval();
      auto delta_b_w =
        (Vector3(b_w) - _prior.gyro_bias_nominal.cast<scalar_t>()).eval();

      Vector3 u = so3Logmap(y_q_hat.conjugate() * y_q) - G_R * delta_b_w;
      r = _weight.cast<scalar_t>() * u;

      return true;
    }
  };
}  // namespace cyclops::initializer
