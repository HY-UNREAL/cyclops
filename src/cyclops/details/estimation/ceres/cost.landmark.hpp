#pragma once

#include "cyclops/details/type.hpp"
#include <Eigen/Dense>

namespace cyclops::estimation {
  struct LandmarkProjectionCostEvaluator {
    Eigen::Vector2d const u;
    Eigen::Matrix2d const weight_sqrt;

    SE3Transform const& extrinsic;

    LandmarkProjectionCostEvaluator(
      FeaturePoint const& feature, SE3Transform const& extrinsic)
        : u(feature.point),
          weight_sqrt(Eigen::LLT<Eigen::Matrix2d>(feature.weight).matrixU()),
          extrinsic(extrinsic) {
    }

    template <typename scalar_t, int dim>
    using Vector = Eigen::Matrix<scalar_t, dim, 1>;

    template <typename scalar_t>
    auto computeCameraPoint(
      scalar_t const* const x_b, scalar_t const* const f) const {
      using Quaternion = Eigen::Quaternion<scalar_t>;
      using Vector3 = Vector<scalar_t, 3>;
      Quaternion const q_b(x_b);
      Quaternion const q_bc = extrinsic.rotation.cast<scalar_t>();
      Quaternion const q_c = q_b * q_bc;

      Vector3 const p_b(x_b + 4);
      Vector3 const p_bc = extrinsic.translation.cast<scalar_t>();
      Vector3 const p_c = p_b + q_b * p_bc;

      return (q_c.inverse() * (Vector3(f) - p_c)).eval();
    }

    template <typename scalar_t>
    auto computeProjectionError(Vector<scalar_t, 3> const& z) const {
      auto const d_min = scalar_t(1e-2);
      auto const d = z.z() < d_min ? d_min : z.z();

      auto const& S = weight_sqrt;
      auto u_hat = (z.template head<2>() / d).eval();
      return (S.cast<scalar_t>() * (u_hat - u.cast<scalar_t>())).eval();
    }

    template <typename scalar_t>
    bool operator()(
      scalar_t const* const x_b, scalar_t const* const f,
      scalar_t* const r) const {
      (Eigen::Map<Vector<scalar_t, 2>>(r)) =
        computeProjectionError(computeCameraPoint(x_b, f));
      return true;
    }
  };
}  // namespace cyclops::estimation
