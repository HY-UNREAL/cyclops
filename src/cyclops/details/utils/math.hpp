#pragma once

#include "cyclops/details/type.hpp"

namespace cyclops {
  static inline SE3Transform inverse(SE3Transform const& x) {
    auto const& p = x.translation;
    auto const& q = x.rotation;
    return SE3Transform {-(q.inverse() * p), q.inverse()};
  }

  static inline SE3Transform compose(
    SE3Transform const& a, SE3Transform const& b) {
    auto const& p_a = a.translation;
    auto const& p_b = b.translation;
    auto const& q_a = a.rotation;
    auto const& q_b = b.rotation;
    return SE3Transform {p_a + q_a * p_b, q_a * q_b};
  }

  // Templatize scalar type to handle the automatic differentiation.
  template <typename scalar_t>
  static auto skew3d(Eigen::Matrix<scalar_t, 3, 1> const& w) {
    auto x = w.x();
    auto y = w.y();
    auto z = w.z();
    auto _0 = scalar_t(0);

    // clang-format off
    Eigen::Matrix<scalar_t, 3, 3> result;
    result <<
      _0, -z, +y,
      +z, _0, -x,
      -y, +x, _0;
    // clang-format on
    return result;
  }

  template <typename scalar_t>
  static auto so3Logmap(Eigen::Quaternion<scalar_t> const& q) {
    auto const v_norm = q.vec().norm();
    auto const w = q.w();
    auto const _2 = scalar_t(2.);

    if (v_norm < scalar_t(1e-10))
      return (_2 * q.vec()).eval();

    if (v_norm < scalar_t(1e-6)) {
      auto const _1 = scalar_t(1.);
      auto const _3 = scalar_t(3.);
      return (_2 * q.vec() / w * (_1 - v_norm * v_norm / _3 / w / w)).eval();
    }
    auto u = q.vec().normalized().eval();
    auto theta = atan2(v_norm, w);
    return (_2 * u * theta).eval();
  }

  /*
   * [1] J. Sola, et.al., "A micro Lie theory for state estimation in robotics",
   *     arxiv preprint, last accessed: Jan. 23, 2022, appendix C.
   */
  template <typename scalar_t>
  static Eigen::Matrix<scalar_t, 3, 3> so3LeftJacobianInverse(
    Eigen::Matrix<scalar_t, 3, 1> const& w) {
    using Matrix3 = Eigen::Matrix<scalar_t, 3, 3>;
    Matrix3 const S = skew3d(w);

    auto const _1 = scalar_t(1);
    auto const _2 = scalar_t(2.);
    auto const theta = w.norm();

    if (theta < scalar_t(1e-10))
      return Matrix3::Identity() - S / _2;

    if (theta < scalar_t(1e-2)) {
      auto const _6 = scalar_t(6.);
      auto const _12 = scalar_t(12.);
      auto const _80 = scalar_t(80.);
      auto const _120 = scalar_t(120.);
      auto const _2016 = scalar_t(2016.);

      auto const theta_square = theta * theta;
      auto const theta_quadro = theta * theta * theta * theta;
      auto const A = _1 / _12 - theta_square / _80 + theta_quadro / _2016;
      auto const B = _1 - theta_square / _6 + theta_quadro / _120;

      return Matrix3::Identity() - S / _2 + (A / B) * S * S;
    }
    auto A = sin(theta) / theta;
    auto B = (_1 - cos(theta)) / theta / theta;
    auto C = (_1 - A / B / _2) / theta / theta;

    return Matrix3::Identity() - S / _2 + C * S * S;
  }

  double chiSquaredCdf(int degrees_of_freedom, double x);
}  // namespace cyclops
